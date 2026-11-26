"""Client and typed parsers for the AESO public API (Phase 5).

Security
--------
* The key is read from the environment variable ``AESO_API_KEY`` (or a local,
  git-ignored ``.env`` file). It is never hard-coded, printed, logged, cached
  or written to disk. It is sent only as the ``API-KEY`` request header.
* Log lines contain the endpoint and the date range / number of assets, never
  headers.

Behaviour
---------
* One shared ``requests.Session`` with urllib3 retries (exponential backoff on
  429 and 5xx, honouring Retry-After) and a polite pause between calls.
* Every chunk's parsed response is cached as parquet under ``data/raw/<endpoint>/``;
  reruns skip cached chunks.
* All timestamps are keyed on the UTC field (``begin_datetime_utc`` /
  ``begin_date_utc``): local "MPT" time repeats an hour on fall-back days
  (before Alberta moved to year-round UTC-6 on 2026-11-01).

Endpoint quirks found by the probe (docs/aeso_api_probe.md)
-----------------------------------------------------------
* pool price: ``endDate`` inclusive, max one year per request -> yearly chunks.
* AIL: a request whose LAST local day is a 25-hour fall-back day drops that
  day's last hour -> every chunk asks for one extra day, results are trimmed,
  de-duplicated on UTC and checked for completeness; a chunk that is still
  incomplete or errors is split in half and retried.
* metered volume: ``endDate`` EXCLUSIVE (start == end returns nothing). The
  documented limit is ~100k records per call, but the API returns HTTP 400 above
  ~10k (message: "report is available for maximum end date of ...") ->
  n_assets * n_days * 24 < 9,900, i.e. one asset-year per call.
* asset list: no fuel type; fuel comes from the Current Supply Demand endpoint.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ROOT

log = logging.getLogger("aeso")

PATHS = {
    "poolprice": "/poolprice-api/v1.1/price/poolPrice",
    "ail": "/actualforecast-api/v1/load/albertaInternalLoad",
    "assetlist": "/assetlist-api/v1/assetlist",
    "meteredvolume": "/meteredvolume-api/v1/meteredvolume/details",
    "csd_assets": "/currentsupplydemand-api/v1/csd/generation/assets/current",
}


class AESOError(RuntimeError):
    pass


class AESOSchemaError(AESOError):
    pass


# ---------------------------------------------------------------------------
# key handling
# ---------------------------------------------------------------------------
def get_api_key(env_file: Path | None = None) -> str:
    """AESO_API_KEY from the environment, else from a local .env (KEY=VALUE lines)."""
    key = os.environ.get("AESO_API_KEY", "").strip()
    if not key:
        env_file = env_file or ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "AESO_API_KEY":
                    key = value.strip().strip("'\"")
    if not key:
        raise AESOError("AESO_API_KEY is not set (environment variable or .env). "
                        "Get a key from the AESO developer portal.")
    return key


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class AESOClient:
    def __init__(self, cfg: dict, key: str | None = None, session: requests.Session | None = None):
        a = cfg["aeso"]
        self.base = a["base_url"].rstrip("/")
        self.timeout = a["timeout_s"]
        self.delay = a["polite_delay_s"]
        self._key = key or get_api_key()
        self.session = session or requests.Session()
        retry = Retry(total=a["max_retries"], backoff_factor=a["backoff_factor"],
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}),
                      respect_retry_after_header=True, raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.calls = 0
        self._last = 0.0

    def __repr__(self):  # never show the key
        return f"AESOClient(base={self.base!r})"

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        shown = {k: (v if k != "asset_ID" else f"{len(str(v).split(','))} assets") for k, v in (params or {}).items()}
        t0 = time.monotonic()
        try:
            r = self.session.get(self.base + PATHS[endpoint], params=params, timeout=self.timeout,
                                 headers={"API-KEY": self._key, "Accept": "application/json"})
        finally:
            self._last = time.monotonic()
            self.calls += 1
        log.info("GET %s %s -> HTTP %s in %.1fs", endpoint, shown, r.status_code, time.monotonic() - t0)
        if r.status_code != 200:  # the body is a short server message (it never contains our headers)
            raise AESOError(f"{endpoint} {shown}: HTTP {r.status_code}: {r.text[:300]}")
        js = r.json()
        if str(js.get("responseCode")) != "200":
            raise AESOError(f"{endpoint} {shown}: responseCode {js.get('responseCode')}")
        return js


# ---------------------------------------------------------------------------
# typed parsers (pure functions; unit-tested on handcrafted fixtures)
# ---------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce").astype(float)


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M").dt.tz_localize("UTC")


def _require(rec_list: list, fields: tuple, where: str) -> None:
    if rec_list:
        missing = [f for f in fields if f not in rec_list[0]]
        if missing:
            raise AESOSchemaError(f"{where}: missing fields {missing}; got {sorted(rec_list[0])}")


def parse_pool_price(js: dict) -> pd.DataFrame:
    recs = js["return"]["Pool Price Report"]
    f = ("begin_datetime_utc", "begin_datetime_mpt", "pool_price", "forecast_pool_price", "rolling_30day_avg")
    _require(recs, f, "pool price")
    df = pd.DataFrame(recs, columns=list(f))
    return pd.DataFrame({
        "ts_utc": _utc(df.begin_datetime_utc), "ts_mpt": df.begin_datetime_mpt.astype(str),
        "pool_price": _num(df.pool_price), "forecast_pool_price": _num(df.forecast_pool_price),
        "rolling_30day_avg": _num(df.rolling_30day_avg),
    })


def parse_ail(js: dict) -> pd.DataFrame:
    recs = js["return"]["Actual Forecast Report"]
    f = ("begin_datetime_utc", "begin_datetime_mpt", "alberta_internal_load", "forecast_alberta_internal_load")
    _require(recs, f, "AIL")
    df = pd.DataFrame(recs, columns=list(f))
    return pd.DataFrame({
        "ts_utc": _utc(df.begin_datetime_utc), "ts_mpt": df.begin_datetime_mpt.astype(str),
        "ail_mw": _num(df.alberta_internal_load), "ail_forecast_mw": _num(df.forecast_alberta_internal_load),
    })


def parse_asset_list(js: dict) -> pd.DataFrame:
    recs = js["return"]
    f = ("asset_name", "asset_ID", "asset_type", "operating_status", "pool_participant_name",
         "pool_participant_ID", "net_to_grid_asset_flag", "asset_incl_storage_flag")
    _require(recs, f, "asset list")
    df = pd.DataFrame(recs, columns=list(f)).astype(str)
    return df.drop_duplicates().reset_index(drop=True)  # NX01 appears twice, identical


def parse_metered_volume(js: dict) -> pd.DataFrame:
    rows = []
    for p in js["return"]:
        for a in p.get("asset_list", []):
            vl = a.get("metered_volume_list", [])
            _require(vl, ("begin_date_utc", "begin_date_mpt", "metered_volume"), "metered volume")
            for v in vl:
                rows.append((v["begin_date_utc"], v["begin_date_mpt"], p.get("pool_participant_ID", ""),
                             a.get("asset_ID", ""), a.get("asset_class", ""), v["metered_volume"]))
    df = pd.DataFrame(rows, columns=["utc", "mpt", "pool_participant_ID", "asset_ID", "asset_class", "mv"])
    if df.empty:
        return pd.DataFrame({"ts_utc": pd.Series(dtype="datetime64[ns, UTC]"), "ts_mpt": pd.Series(dtype=str),
                             "pool_participant_ID": pd.Series(dtype=str), "asset_ID": pd.Series(dtype=str),
                             "asset_class": pd.Series(dtype=str), "metered_volume_mwh": pd.Series(dtype=float)})
    return pd.DataFrame({"ts_utc": _utc(df.utc), "ts_mpt": df.mpt.astype(str),
                         "pool_participant_ID": df.pool_participant_ID.astype(str),
                         "asset_ID": df.asset_ID.astype(str), "asset_class": df.asset_class.astype(str),
                         "metered_volume_mwh": _num(df.mv)})


def parse_csd_assets(js: dict) -> pd.DataFrame:
    r = js["return"]
    recs = r["asset_list"]
    f = ("asset", "fuel_type", "sub_fuel_type", "maximum_capability", "net_generation")
    _require(recs, f, "CSD assets")
    df = pd.DataFrame(recs)
    out = pd.DataFrame({
        "asset_ID": df.asset.astype(str), "fuel_type": df.fuel_type.astype(str),
        "sub_fuel_type": df.sub_fuel_type.fillna("").astype(str),
        "maximum_capability_mw": pd.to_numeric(df.maximum_capability, errors="coerce").astype(float),
        "net_generation_mw": pd.to_numeric(df.net_generation, errors="coerce").astype(float),
    })
    out["snapshot_utc"] = pd.Timestamp(r.get("last_updated_datetime_utc"), tz="UTC")
    return out


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------
def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))


def local_day_start_utc(day, tz: str) -> pd.Timestamp:
    """UTC instant at which local calendar day ``day`` starts (DST-aware via tz database)."""
    return pd.Timestamp(_d(day).isoformat()).tz_localize(tz).tz_convert("UTC")


def expected_hours(start, end, tz: str) -> pd.DatetimeIndex:
    """All UTC hour-beginnings of local days start..end (inclusive)."""
    a, b = local_day_start_utc(start, tz), local_day_start_utc(_d(end) + timedelta(days=1), tz)
    return pd.date_range(a, b, freq="h", inclusive="left")


def completeness(ts_utc: pd.Series, start, end, tz: str) -> dict:
    exp = expected_hours(start, end, tz)
    have = pd.DatetimeIndex(ts_utc.dropna().unique())
    missing = exp.difference(have)
    return {"expected_hours": len(exp), "present_hours": int(have.isin(exp).sum()),
            "missing_hours": len(missing), "first_missing": [str(t) for t in missing[:10]],
            "duplicate_rows": int(ts_utc.duplicated().sum())}


def year_chunks(start, end) -> list[tuple[date, date]]:
    s, e = _d(start), _d(end)
    out = []
    while s <= e:
        ce = min(e, date(s.year, 12, 31))
        out.append((s, ce))
        s = ce + timedelta(days=1)
    return out


def day_chunks(start, end, days: int) -> list[tuple[date, date]]:
    s, e = _d(start), _d(end)
    out = []
    while s <= e:
        ce = min(e, s + timedelta(days=days - 1))
        out.append((s, ce))
        s = ce + timedelta(days=1)
    return out


def metered_batches(asset_ids: list[str], start, end, max_records: int) -> list[tuple[list[str], date, date]]:
    """(assets, first day, last day) batches with n_assets * n_days * 24 < max_records."""
    out = []
    for cs, ce in year_chunks(start, end):
        n_days = (ce - cs).days + 1
        per = max(1, (max_records - 1) // (24 * n_days))
        if per < 1:
            raise ValueError("max_records too small for one asset-year")
        ids = sorted(asset_ids)
        for i in range(0, len(ids), per):
            out.append((ids[i:i + per], cs, ce))
    return out


# ---------------------------------------------------------------------------
# cached fetchers
# ---------------------------------------------------------------------------
@dataclass
class Cache:
    root: Path

    def path(self, endpoint: str, name: str) -> Path:
        p = self.root / endpoint / f"{name}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_or_fetch(self, endpoint: str, name: str, fetch) -> tuple[pd.DataFrame, bool]:
        p = self.path(endpoint, name)
        if p.exists():
            return pd.read_parquet(p), True
        df = fetch()
        df.to_parquet(p, index=False)
        return df, False


def cache_for(cfg: dict) -> Cache:
    return Cache(ROOT / cfg["aeso"]["cache_dir"])


def fetch_pool_price(client: AESOClient, cache: Cache, start, end) -> pd.DataFrame:
    parts = []
    for cs, ce in year_chunks(start, end):  # max 1 year per request, endDate inclusive
        df, hit = cache.get_or_fetch("poolprice", f"{cs}_{ce}", lambda cs=cs, ce=ce: parse_pool_price(
            client.get("poolprice", {"startDate": cs.isoformat(), "endDate": ce.isoformat()})))
        log.info("pool price %s..%s: %d rows%s", cs, ce, len(df), " (cache)" if hit else "")
        parts.append(df)
    return pd.concat(parts).drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def fetch_ail(client: AESOClient, cache: Cache, start, end, tz: str, chunk_days: int) -> pd.DataFrame:
    parts = []

    def one(cs: date, ce: date, days: int):
        name = f"{cs}_{ce}"
        p = cache.path("ail", name)
        split_marker = p.with_suffix(".split")  # this chunk failed before: go straight to its halves
        if p.exists():
            parts.append(pd.read_parquet(p))
            return
        if split_marker.exists() and days > 1:
            half = max(1, days // 2)
            for s2, e2 in day_chunks(cs, ce, half):
                one(s2, e2, half)
            return
        try:
            # ask for one extra day: a request ending on a fall-back day drops its last hour
            raw = parse_ail(client.get("ail", {"startDate": cs.isoformat(),
                                               "endDate": (ce + timedelta(days=1)).isoformat()}))
            lo, hi = local_day_start_utc(cs, tz), local_day_start_utc(ce + timedelta(days=1), tz)
            df = raw[(raw.ts_utc >= lo) & (raw.ts_utc < hi)].drop_duplicates("ts_utc")
            comp = completeness(df.ts_utc, cs, ce, tz)
            if comp["missing_hours"] and days > 1:
                raise AESOError(f"truncated: {comp['missing_hours']} missing hours")
        except (AESOError, requests.RequestException) as exc:
            if days <= 1:
                raise
            half = max(1, days // 2)
            log.warning("AIL %s..%s failed (%s); splitting into %d-day chunks", cs, ce, exc, half)
            split_marker.touch()
            for s2, e2 in day_chunks(cs, ce, half):
                one(s2, e2, half)
            return
        df.to_parquet(p, index=False)
        log.info("AIL %s..%s: %d rows", cs, ce, len(df))
        parts.append(df)

    for cs, ce in day_chunks(start, end, chunk_days):
        one(cs, ce, chunk_days)
    return pd.concat(parts).drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)


def fetch_asset_list(client: AESOClient, cache: Cache, snapshot: str) -> pd.DataFrame:
    df, _ = cache.get_or_fetch("assetlist", f"assetlist_{snapshot}", lambda: parse_asset_list(client.get("assetlist")))
    return df


def fetch_csd_assets(client: AESOClient, cache: Cache, snapshot: str) -> pd.DataFrame:
    df, _ = cache.get_or_fetch("csd", f"csd_assets_{snapshot}", lambda: parse_csd_assets(client.get("csd_assets")))
    return df


def fetch_metered(client: AESOClient, cache: Cache, asset_ids: list[str], start, end, max_records: int) -> pd.DataFrame:
    parts = []
    for ids, cs, ce in metered_batches(asset_ids, start, end, max_records):
        h = hashlib.sha1(",".join(ids).encode()).hexdigest()[:10]
        df, hit = cache.get_or_fetch("meteredvolume", f"{cs}_{ce}_{len(ids)}assets_{h}", lambda ids=ids, cs=cs, ce=ce: (
            parse_metered_volume(client.get("meteredvolume", {
                "startDate": cs.isoformat(), "endDate": (ce + timedelta(days=1)).isoformat(),  # endDate exclusive
                "asset_ID": ",".join(ids)}))))
        log.info("metered %s..%s, %d assets: %d rows%s", cs, ce, len(ids), len(df), " (cache)" if hit else "")
        parts.append(df)
    out = pd.concat(parts) if parts else parse_metered_volume({"return": []})
    return out.drop_duplicates(["asset_ID", "ts_utc"]).sort_values(["asset_ID", "ts_utc"]).reset_index(drop=True)
