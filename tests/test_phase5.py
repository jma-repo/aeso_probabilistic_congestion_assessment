"""Phase 5 tests: AESO client, typed parsers, chunking, DST handling, caching.

No network access: a fake session / fake client reproduce the response shapes and
the quirks found by the probe (docs/aeso_api_probe.md).
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src import aeso_api as A
from src.config import load_config
from src.features import merit_order_price

TZ = "America/Edmonton"
SECRET = "test-key-should-never-appear"


# ---------------------------------------------------------------- key handling
def test_key_from_env_and_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("AESO_API_KEY", "abc")
    assert A.get_api_key(tmp_path / "none") == "abc"
    monkeypatch.delenv("AESO_API_KEY")
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nAESO_API_KEY='from-dotenv'\n")
    assert A.get_api_key(env) == "from-dotenv"
    with pytest.raises(A.AESOError, match="not set"):
        A.get_api_key(tmp_path / "missing.env")


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, str(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def mount(self, *a):
        pass

    def get(self, url, params=None, timeout=None, headers=None):
        self.requests.append((url, params, headers))
        return self.responses.pop(0)


def _cfg():
    cfg = load_config()
    cfg["aeso"]["polite_delay_s"] = 0
    return cfg


def test_client_sends_key_header_but_never_logs_it(caplog):
    payload = {"responseCode": "200", "return": {"Pool Price Report": []}}
    s = FakeSession([FakeResponse(200, payload), FakeResponse(400, {"message": "bad"})])
    c = A.AESOClient(_cfg(), key=SECRET, session=s)
    with caplog.at_level(logging.DEBUG, logger="aeso"):
        c.get("poolprice", {"startDate": "2024-01-01", "endDate": "2024-01-02"})
        with pytest.raises(A.AESOError) as e:
            c.get("meteredvolume", {"startDate": "2024-01-01", "asset_ID": "A,B,C"})
    assert s.requests[0][2]["API-KEY"] == SECRET
    assert "API-KEY" not in str(s.requests[0][1])           # not in the query string
    assert SECRET not in caplog.text and SECRET not in repr(c) and SECRET not in str(e.value)
    assert "3 assets" in caplog.text and "2024-01-01" in caplog.text  # date range + asset count are logged


# ---------------------------------------------------------------- parsers
def test_parse_pool_price_types_and_empty_values():
    js = {"return": {"Pool Price Report": [
        {"begin_datetime_utc": "2025-11-02 07:00", "begin_datetime_mpt": "2025-11-02 01:00", "pool_price": "12.5",
         "forecast_pool_price": "", "rolling_30day_avg": "40"},
        {"begin_datetime_utc": "2025-11-02 08:00", "begin_datetime_mpt": "2025-11-02 01:00", "pool_price": "13",
         "forecast_pool_price": "14", "rolling_30day_avg": "40"}]}}
    df = A.parse_pool_price(js)
    assert isinstance(df.ts_utc.dtype, pd.DatetimeTZDtype) and str(df.ts_utc.dt.tz) == "UTC" and df.ts_utc.is_unique
    assert df.ts_mpt.duplicated().any()          # local time repeats on fall-back days: UTC is the key
    assert np.isnan(df.forecast_pool_price.iloc[0]) and df.pool_price.dtype == float


def test_parser_reports_schema_changes():
    with pytest.raises(A.AESOSchemaError, match="pool_price"):
        A.parse_pool_price({"return": {"Pool Price Report": [{"begin_datetime_utc": "2025-01-01 07:00"}]}})


def test_parse_metered_volume_nested_and_empty():
    js = {"return": [{"pool_participant_ID": "P1", "asset_list": [
        {"asset_ID": "W1", "asset_class": "IPP", "metered_volume_list": [
            {"begin_date_utc": "2025-06-01 06:00", "begin_date_mpt": "2025-06-01 00:00", "metered_volume": "1.5"}]},
        {"asset_ID": "W2", "asset_class": "IPP", "metered_volume_list": []}]}]}
    df = A.parse_metered_volume(js)
    assert list(df.asset_ID) == ["W1"] and df.metered_volume_mwh.iloc[0] == 1.5
    assert A.parse_metered_volume({"return": []}).empty


def test_parse_asset_list_drops_exact_duplicates():
    rec = {"asset_name": "NX01 Nexen", "asset_ID": "NX01", "asset_type": "SOURCE", "operating_status": "Active",
           "pool_participant_name": "x", "pool_participant_ID": "y", "net_to_grid_asset_flag": "",
           "asset_incl_storage_flag": ""}
    assert len(A.parse_asset_list({"return": [rec, dict(rec)]})) == 1


# ---------------------------------------------------------------- time + chunking
def test_expected_hours_follow_dst_from_tz_database():
    assert len(A.expected_hours("2025-11-02", "2025-11-02", TZ)) == 25   # fall back
    assert len(A.expected_hours("2025-03-09", "2025-03-09", TZ)) == 23   # spring forward
    assert len(A.expected_hours("2024-01-01", "2025-12-31", TZ)) == 17544


def test_year_chunks_and_metered_batches_respect_limits():
    ch = A.year_chunks("2024-03-15", "2025-12-31")
    assert ch == [(date(2024, 3, 15), date(2024, 12, 31)), (date(2025, 1, 1), date(2025, 12, 31))]
    ids = [f"A{i:03d}" for i in range(23)]
    for lim in (9900, 90000):
        b = A.metered_batches(ids, "2024-01-01", "2025-12-31", lim)
        for assets, s, e in b:
            assert len(assets) * ((e - s).days + 1) * 24 < lim
        covered = {(a, s) for assets, s, _ in b for a in assets}
        assert covered == {(a, s) for a in ids for s in (date(2024, 1, 1), date(2025, 1, 1))}


class FakeAILClient:
    """Mimics the AIL endpoint, incl. the quirk: if the LAST requested local day is a
    fall-back day, its last hour is missing. Optionally fails for long ranges."""

    def __init__(self, fail_over_days=None):
        self.calls, self.fail = [], fail_over_days

    def get(self, endpoint, params):
        s, e = date.fromisoformat(params["startDate"]), date.fromisoformat(params["endDate"])
        self.calls.append((s, e))
        if self.fail and (e - s).days + 1 > self.fail:
            raise A.AESOError("HTTP 500")
        hrs = A.expected_hours(s, e, TZ)
        if len(A.expected_hours(e, e, TZ)) == 25:
            hrs = hrs[:-1]
        recs = [{"begin_datetime_utc": t.strftime("%Y-%m-%d %H:%M"), "begin_datetime_mpt": "x",
                 "alberta_internal_load": "10000", "forecast_alberta_internal_load": ""} for t in hrs]
        return {"responseCode": "200", "return": {"Actual Forecast Report": recs}}


@pytest.mark.parametrize("fail", [None, 10])
def test_fetch_ail_handles_fallback_quirk_and_errors(tmp_path, fail):
    client = FakeAILClient(fail_over_days=fail)
    cache = A.Cache(tmp_path)
    # chunk boundaries chosen so that 2025-11-02 (fall-back) is the last day of a chunk
    df = A.fetch_ail(client, cache, "2025-10-20", "2025-11-10", TZ, chunk_days=14)
    comp = A.completeness(df.ts_utc, "2025-10-20", "2025-11-10", TZ)
    assert comp["missing_hours"] == 0 and comp["duplicate_rows"] == 0
    n = len(client.calls)
    A.fetch_ail(client, cache, "2025-10-20", "2025-11-10", TZ, chunk_days=14)
    assert len(client.calls) == n                 # second run: everything from cache


def test_merit_order_with_must_run_shifts_only_low_prices():
    cfg = load_config()
    x = pd.DataFrame({f"load_{b}": [3000 * s] for b, s in [("NORTH", .22), ("EDM", .3), ("RD", .08), ("CAL", .3), ("SOUTH", .1)]})
    x["cf_wind_south"], x["cf_wind_central"], x["cf_solar_south"], x["pu_hydro"] = 1.0, 1.0, 0.0, 0.5
    cfg["generators"]["NORTH_cogen"]["p_min_pu"] = 0.0
    p0 = merit_order_price(cfg, x).iloc[0]
    cfg["generators"]["NORTH_cogen"]["p_min_pu"] = 0.5   # 2,000 MW must-run pushes wind to the margin
    p1 = merit_order_price(cfg, x).iloc[0]
    assert p1 <= p0 and p1 == pytest.approx(0.5)
