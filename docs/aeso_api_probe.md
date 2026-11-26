# AESO API probe (Phase 5, step 1)

One small request per endpoint, made on 2026-11-26 before writing any parser. Base URL
`https://apimgw.aeso.ca/public`, key sent in the `API-KEY` header (never stored in the repo; in the cloud session it
is injected by the network proxy). Raw responses are kept locally in `data/raw/probe/` (git-ignored).

All responses share an envelope:

```json
{"timestamp": "2026-11-26 00:14:18.431+0000", "responseCode": "200", "return": ...}
```

All numeric values arrive as **strings** (e.g. `"pool_price": "30.56"`), except in the CSD endpoint.

## 1. Pool price — `GET /poolprice-api/v1.1/price/poolPrice?startDate=&endDate=`

`return["Pool Price Report"]` → list, one record per hour:

| field | example | type after parsing |
|---|---|---|
| `begin_datetime_utc` | `"2026-11-01 06:00"` | timestamp, UTC (**the key we use**) |
| `begin_datetime_mpt` | `"2026-11-01 00:00"` | local "Mountain Prevailing Time", hour *beginning* |
| `pool_price` | `"30.56"` | float, $/MWh |
| `forecast_pool_price` | `"29.71"` | float |
| `rolling_30day_avg` | `"49.95"` | float |

* `endDate` is **inclusive** (`startDate=endDate` returns that local day).
* Timestamps are hour-*beginning*; AESO's HE01–HE24 "hour ending" label = local begin hour + 1.

## 2. Alberta Internal Load — `GET /actualforecast-api/v1/load/albertaInternalLoad?startDate=&endDate=`

`return["Actual Forecast Report"]` → list, one record per hour:

| field | example |
|---|---|
| `begin_datetime_utc` | `"2026-11-01 06:00"` |
| `begin_datetime_mpt` | `"2026-11-01 00:00"` |
| `alberta_internal_load` | `"10194"` (MW) |
| `forecast_alberta_internal_load` | `"10227"` (MW) |

* **Quirk:** a single-day request for the 25-hour fall-back day (2025-11-02) returns only **24** rows: the last
  hour (UTC 2025-11-03 06:00) is missing. A two-day request returns it. → Chunks must overlap by one day and be
  de-duplicated on the UTC timestamp, followed by a contiguity check.

## 3. Asset list — `GET /assetlist-api/v1/assetlist`

`return` → list of 3,757 records (one duplicated: `NX01` appears twice, identical):

| field | distinct values (count) |
|---|---|
| `asset_type` | `SINK` (2,343), `SOURCE` (1,414) |
| `operating_status` | `Active` (2,118), `Retired` (1,582), `Inactive` (36), `Suspended` (21) |
| `net_to_grid_asset_flag` | `""` (3,680), `Y` (77) |
| `asset_incl_storage_flag` | `""` (3,748), `Y` (9) |
| `asset_name`, `asset_ID`, `pool_participant_name`, `pool_participant_ID` | free text / IDs |

**There is no fuel-type field.** Name matching is not usable: of the wind and solar assets identified below, 34 of
50 wind and 30 of 49 solar asset names contain neither "wind" nor "solar".

## 4. Metered volume — `GET /meteredvolume-api/v1/meteredvolume/details?startDate=&endDate=&asset_ID=`

`return` → list per pool participant → `asset_list` → `metered_volume_list`:

```json
[{"pool_participant_ID": "CHDM",
  "asset_list": [{"asset_ID": "AKE1", "asset_class": "IPP",
                  "metered_volume_list": [{"begin_date_utc": "2025-06-01 06:00",
                                           "begin_date_mpt": "2025-06-01 00:00",
                                           "metered_volume": "0"}, ...]}]}]
```

* `endDate` is **exclusive**: `startDate=endDate` returns an empty list (also for all assets); `2025-06-01 →
  2025-06-02` returns the 24 hours of 2025-06-01. Omitting `endDate` returns one day.
* Field names differ from the other endpoints (`begin_date_*` vs `begin_datetime_*`).
* Data are available up to about 5 days before today.
* **Record limit is ~10,000 per call, not ~100,000.** 10 assets × 1 year and even 3 assets × 1 year return HTTP 400
  with `"With the current inputs specified the report is available for maximum end date of 2024-05-18"` (3 assets
  × 139 days ≈ 10,000 records). One asset × one (leap) year = 8,784 records works. The client therefore batches
  `n_assets × n_days × 24 < 9,900`.

## 5. (Not in the original list) Current Supply Demand — `GET /currentsupplydemand-api/v1/csd/generation/assets/current`

Used only to **classify assets by fuel**, because the asset list has no fuel type. Snapshot of currently reporting
generators (231), numbers as JSON numbers:

| `fuel_type` / `sub_fuel_type` | assets | maximum capability (MW) |
|---|---:|---:|
| WIND | 50 | 5,684 |
| SOLAR | 49 | 1,892 |
| GAS / COGENERATION | 40 | 6,125 |
| GAS / COMBINED_CYCLE | 11 | 3,974 |
| GAS / GAS_FIRED_STEAM | 8 | 3,078 |
| GAS / SIMPLE_CYCLE | 38 | 1,014 |
| HYDRO | 9 | 896 |
| ENERGY STORAGE | 11 | 270 |
| OTHER | 15 | 485 |

Fields: `asset`, `fuel_type`, `sub_fuel_type`, `maximum_capability`, `net_generation`,
`dispatched_contingency_reserve`. All 231 assets are `Active` in the asset list.

**Caveat:** like the asset list, this is a *current* snapshot. Wind/solar assets that retired before today are not
in it; historical coverage must be checked by comparing fleet totals over time.

## Time zone finding

* Responses carry both a UTC and a local (`_mpt`) timestamp. **UTC is unambiguous; local is not**: on the
  2025-11-02 fall-back day, local `01:00` appears twice (UTC 07:00 and 08:00). The parsers key on UTC.
* The container's tz database (version **2026e**) says `America/Edmonton` uses MDT until **2026-11-01 02:00** and
  a fixed **UTC−6** afterwards (no fall-back in Nov 2026). AESO's data agree: 2026-03-08 has 23 hours, 2026-11-01
  has 24 hours, and every local timestamp equals the tz-database conversion of its UTC timestamp. Conversions
  therefore use `zoneinfo` with an up-to-date tz database, never a hard-coded offset.
