from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001324404.json"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "IowaCornRevenueResearch/1.0 https://github.com/Patrickbateman-g/iowa-corn-fertilizer-revenue",
)
POINTS = {"Ames": (42.0308, -93.6319), "Sioux City": (42.5, -96.4), "Cedar Rapids": (41.9779, -91.6656)}
FIRST_YEAR, LAST_YEAR = 2006, 2026
FEATURES = ["prior_q2_usd", "rain_mm", "heavy_rain_days"]
TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues", "SalesRevenueGoodsNet")
SNAPSHOT_TAGS = TAGS + ("ManualOriginal10K", "ManualOriginal10Q")
MIN_TRAIN = 6
ALPHA = 1.0
OUTPUT = Path("/kaggle/working/iowa_corn_revenue")


def request_json(url: str, params: dict | None = None, *, delay: float = 0.2) -> dict:
    full_url = url + ("?" + urlencode(params) if params else "")
    req = Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as response:
                result = json.load(response)
            time.sleep(delay)
            return result
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code} from {url}; check access and SEC_USER_AGENT") from exc
            if attempt == 2:
                raise RuntimeError(f"Could not fetch {url}; check Kaggle internet access: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def extract_revenue(document: dict, first: int = 2005, last: int = 2026) -> pd.DataFrame:
    gaap = document.get("facts", {}).get("us-gaap", {})
    found_tags = [tag for tag in TAGS if tag in gaap]
    if not found_tags:
        raise ValueError(f"No known GAAP sales tag present. Inspect available tags: {list(gaap)[:30]}")
    selected = []
    for year in range(first, last + 1):
        start, end = f"{year}-04-01", f"{year}-06-30"
        known_by = f"{year + 1}-05-31"
        candidates = []
        for rank, tag in enumerate(TAGS):
            for fact in gaap.get(tag, {}).get("units", {}).get("USD", []):
                filed = fact.get("filed", "")
                if (fact.get("start") == start and fact.get("end") == end
                        and fact.get("form") in ("10-Q", "10-Q/A")
                        and end <= filed <= known_by and fact.get("val") is not None):
                    candidates.append((filed, rank, tag, fact))
        if not candidates:
            continue
        candidates.sort(key=lambda c: (c[0], c[1], c[3].get("accn", "")))
        earliest = candidates[0][0]
        first_filing = [c for c in candidates if c[0] == earliest]
        values = {float(c[3]["val"]) for c in first_filing}
        if len(values) != 1:
            raise ValueError(f"Conflicting earliest consolidated revenue facts in {year}: {values}; inspect 10-Q")
        filed, _, tag, fact = first_filing[0]
        selected.append({"year": year, "q2_usd": float(fact["val"]),
                         "q2_filed": filed, "q2_accession": fact.get("accn", ""), "q2_tag": tag})
    if not selected:
        raise ValueError(f"No qualifying Q2 revenue facts among tags {found_tags}")
    return pd.DataFrame(selected).set_index("year").sort_index()


def summarize_point(payload: dict, year: int) -> dict:
    expected = pd.date_range(f"{year}-04-01", f"{year}-05-20", freq="D")
    keys = [d.strftime("%Y%m%d") for d in expected]
    parameter = payload.get("properties", {}).get("parameter", {})
    rain, temp = parameter.get("PRECTOTCORR", {}), parameter.get("T2M", {})
    if not rain or not temp:
        raise ValueError(f"NASA missing required fields in {year}")
    def number(data, key):
        value = data.get(key)
        if value is None:
            return np.nan
        value = float(value)
        return np.nan if not np.isfinite(value) or value <= -900 else value
    r = np.array([number(rain, k) for k in keys])
    t = np.array([number(temp, k) for k in keys])
    if np.isnan(r).any() or np.isnan(t).any() or (r < 0).any():
        raise ValueError(f"NASA incomplete or invalid daily coverage in {year}; refusing partial aggregation")
    return {"rain_mm": float(r.sum()), "heavy_rain_days": int((r >= 25).sum()), "temp_c": float(t.mean())}


def fetch_weather(years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        values = []
        for name, (latitude, longitude) in POINTS.items():
            params = {"parameters": "PRECTOTCORR,T2M", "community": "AG", "longitude": longitude,
                      "latitude": latitude, "start": f"{year}0401", "end": f"{year}0520",
                      "format": "JSON", "time-standard": "LST"}
            payload = request_json(POWER_URL, params)
            values.append(summarize_point(payload, year))
        rows.append({"year": year, **{key: float(np.mean([p[key] for p in values])) for key in values[0]}})
    print(f"Weather aggregation complete for {len(rows)} seasons", flush=True)
    return pd.DataFrame(rows).set_index("year").sort_index()


def make_dataset(revenue: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        if year not in revenue.index or year - 1 not in revenue.index or year not in weather.index:
            continue
        current, previous = revenue.loc[year], revenue.loc[year - 1]
        cutoff = f"{year}-05-31"
        if str(previous.q2_filed) > cutoff:
            raise ValueError(f"Previous Q2 was not public by {cutoff}")
        rows.append({"year": year, "as_of": cutoff, "target_q2_usd": current.q2_usd,
                     "target_filed": current.q2_filed, "target_accession": current.q2_accession,
                     "prior_q2_usd": previous.q2_usd, "prior_filed": previous.q2_filed,
                     "prior_accession": previous.q2_accession,
                     **weather.loc[year].to_dict()})
    if not rows:
        raise ValueError("No matching weather and revenue years with a known prior Q2")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def ridge_predict(train: pd.DataFrame, row: pd.Series, columns: list[str]) -> float:
    x = train[columns].to_numpy(dtype=float)
    y = train["target_q2_usd"].to_numpy(dtype=float) / 1_000_000.0
    mu, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    xx = np.column_stack([np.ones(len(x)), (x - mu) / scale])
    one = np.r_[1.0, (row[columns].to_numpy(dtype=float) - mu) / scale]
    penalty = np.eye(xx.shape[1]) * ALPHA
    penalty[0, 0] = 0
    beta = np.linalg.solve(xx.T @ xx + penalty, xx.T @ y)
    return float(one @ beta * 1_000_000.0)


def evaluate(dataset: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for _, row in dataset.iterrows():
        year = int(row.year)
        train = dataset[(dataset.year < year) & (dataset.target_filed <= row.as_of)]
        if len(train) < MIN_TRAIN:
            continue
        stage = "development" if year <= 2021 else "holdout_2022_2025" if year <= 2025 else "holdout_2026"
        values = {
            "last_year": float(row.prior_q2_usd),
            "ridge_prior": ridge_predict(train, row, FEATURES[:1]),
            "ridge_weather": ridge_predict(train, row, FEATURES),
        }
        for model, pred in values.items():
            target = float(row.target_q2_usd)
            predictions.append({"year": year, "stage": stage, "model": model,
                                "training_rows": len(train), "target_usd": target,
                                "prediction_usd": pred, "absolute_error_usd": abs(pred - target),
                                "absolute_pct_error": 100 * abs(pred - target) / target if target > 0 else np.nan})
    return pd.DataFrame(predictions)


def verify_original_filings(revenue: pd.DataFrame) -> None:
    for year, value in ((2006, 664_800_000), (2008, 1_161_000_000), (2009, 991_000_000), (2025, 1_890_000_000)):
        if year in revenue.index and float(revenue.loc[year, "q2_usd"]) != value:
            raise ValueError(f"{year} result disagrees with manually checked original filing: {revenue.loc[year, 'q2_usd']}")


def read_snapshot(snapshot: dict | str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if isinstance(snapshot, Path):
        snapshot = json.loads(snapshot.read_text(encoding="utf-8"))
    elif isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    meta = snapshot.get("metadata", {})
    expected_points = {k: list(v) for k, v in POINTS.items()}
    if meta.get("points_lat_lon") != expected_points or meta.get("time_standard") != "LST":
        raise ValueError("Snapshot coordinates or time standard differ from pre-specified analysis")
    if meta.get("days_per_point") != 50 or meta.get("missing_daily_values") != 0:
        raise ValueError("Snapshot must document 50 complete days for every year and point")
    rev = pd.DataFrame(snapshot["revenue"])
    weather = pd.DataFrame(snapshot["weather"])
    for label, frame, fields in (("revenue", rev, {"year", "q2_usd", "q2_filed", "q2_accession", "q2_tag"}),
                                  ("weather", weather, {"year", "rain_mm", "heavy_rain_days", "temp_c"})):
        if not fields.issubset(frame.columns) or frame["year"].duplicated().any() or frame[list(fields)].isna().any().any():
            raise ValueError(f"Snapshot {label} missing required fields or has duplicate years")
    rev["year"] = rev["year"].astype(int)
    weather["year"] = weather["year"].astype(int)
    for _, row in rev.iterrows():
        if not (f"{row.year}-06-30" <= str(row.q2_filed) <= f"{row.year + 1}-05-31"):
            raise ValueError(f"Revenue fact for {row.year} was unavailable by the next prediction date")
        if row.q2_tag not in SNAPSHOT_TAGS or float(row.q2_usd) <= 0 or not row.q2_accession:
            raise ValueError(f"Invalid revenue fact for {row.year}")
    if ((weather.rain_mm < 0) | (weather.heavy_rain_days < 0) | (weather.heavy_rain_days > 50)).any():
        raise ValueError("Snapshot weather summary outside physically possible bounds")
    return rev.set_index("year").sort_index(), weather.set_index("year").sort_index(), meta


def point_sensitivity(snapshot: dict | str | Path, dataset: pd.DataFrame) -> pd.DataFrame | None:
    if isinstance(snapshot, Path):
        snapshot = json.loads(snapshot.read_text(encoding="utf-8"))
    elif isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    if "point_weather" not in snapshot:
        return None
    point = pd.DataFrame(snapshot["point_weather"])
    fields = {"year", "point", "latitude", "longitude", "days", "missing",
              "precip_mm", "heavy_days", "temperature_c"}
    if not fields.issubset(point.columns) or point[list(fields)].isna().any().any():
        raise ValueError("Point sensitivity needs complete per-cell aggregate fields")
    if point.duplicated(["year", "point"]).any():
        raise ValueError("Duplicate year/point weather summaries")
    requested = set(dataset.year.astype(int))
    point = point[point.year.isin(requested)].copy()
    for year in sorted(requested):
        rows = point[point.year == year]
        if len(rows) != 3 or set(rows.point) != set(POINTS):
            raise ValueError(f"Sensitivity requires each of the three fixed points for {year}")
        for _, item in rows.iterrows():
            latitude, longitude = POINTS[item.point]
            if (not np.isclose(float(item.latitude), latitude, atol=1e-6)
                or not np.isclose(float(item.longitude), longitude, atol=1e-6)
                or item.days != 50 or item.missing != 0
                or item.precip_mm < 0 or not (0 <= item.heavy_days <= 50)):
                raise ValueError(f"Invalid sensitivity point summary for {year}/{item.point}")
        original = dataset[dataset.year == year].iloc[0]
        for col, expected in (("precip_mm", original.rain_mm), ("heavy_days", original.heavy_rain_days),
                              ("temperature_c", original.temp_c)):
            if not np.isclose(rows[col].mean(), expected, rtol=1e-5, atol=1e-3):
                raise ValueError(f"Per-point and original statewide proxy disagree for {year}/{col}")

    results = []
    for excluded in POINTS:
        retained = point[point.point != excluded]
        two_point = retained.groupby("year", as_index=False).agg(
            rain_mm=("precip_mm", "mean"), heavy_rain_days=("heavy_days", "mean"))
        altered = dataset.drop(columns=["rain_mm", "heavy_rain_days"]).merge(two_point, on="year", validate="one_to_one")
        selected = evaluate(altered)
        selected = selected[(selected.model == "ridge_weather") & selected.stage.isin(["holdout_2022_2025", "holdout_2026"])]
        for stage, rows in selected.groupby("stage"):
            results.append({"excluded_point": excluded, "stage": stage, "n_years": len(rows),
                            "mae_usd": rows.absolute_error_usd.mean(), "mape_pct": rows.absolute_pct_error.mean()})
    return pd.DataFrame(results)


def build_report(dataset: pd.DataFrame, pred: pd.DataFrame, scores: pd.DataFrame,
                 input_meta: dict, used_snapshot: bool, sensitivity: pd.DataFrame | None = None) -> str:
    stage_names = {"development": "Development", "holdout_2022_2025": "2022–2025 holdout",
                   "holdout_2026": "2026 separate holdout"}
    model_names = {"last_year": "Previous Q2", "ridge_prior": "Revenue-only ridge",
                   "ridge_weather": "Revenue + weather ridge"}
    lines = [
        "# Iowa corn weather and CF revenue: exploratory results", "",
        "Each year, the model estimates CF Industries’ April–June revenue on May 31 using the previous Q2 revenue "
        "and Iowa weather from April 1 through May 20. Dollar amounts below are in US$ millions.", "",
        "## Out-of-sample results", "",
        "MAE is average absolute error; MAPE is average absolute percentage error. Lower is better.", "",
        "| Period | Model | Years | MAE (USD m) | MAPE |", "|---|---|---:|---:|---:|",
    ]
    for stage in stage_names:
        for model in model_names:
            match = scores[(scores.stage == stage) & (scores.model == model)]
            if not match.empty:
                s = match.iloc[0]
                pct = f"{s.mape_pct:.1f}%" if np.isfinite(s.mape_pct) else "n/a"
                lines.append(f"| {stage_names[stage]} | {model_names[model]} | {int(s.n_years)} | {s.mae_usd / 1e6:,.1f} | {pct} |")

    lines.extend(["", "## What the results show", ""])
    for stage in ("holdout_2022_2025", "holdout_2026"):
        table = scores[scores.stage == stage].set_index("model")
        needed = set(model_names)
        if needed.issubset(table.index):
            weather = float(table.loc["ridge_weather", "mae_usd"]) / 1e6
            previous = float(table.loc["last_year", "mae_usd"]) / 1e6
            prior = float(table.loc["ridge_prior", "mae_usd"]) / 1e6
            better = weather < previous and weather < prior
            result = "beat both comparisons" if better else "did not beat both comparisons"
            actual_years = sorted(set(pred.loc[pred.stage == stage, "year"].astype(int)))
            lines.append(f"- **{stage_names[stage]}:** Weather-model MAE was USD {weather:,.1f}m; "
                         f"the previous-Q2 baseline was USD {previous:,.1f}m and revenue-only ridge was USD {prior:,.1f}m. "
                         f"The weather model {result} ({', '.join(map(str, actual_years))}).")
        else:
            lines.append(f"- **{stage_names[stage]}:** Insufficient matching predictions to compare all three methods.")
    for year in (2022, 2023):
        rows = pred[(pred.year == year) & (pred.model == "ridge_weather")]
        if not rows.empty:
            row = rows.iloc[0]
            lines.append(f"- **{year} weather-model error:** USD {row.absolute_error_usd / 1e6:,.1f}m "
                         f"({row.absolute_pct_error:.1f}%); actual revenue was USD {row.target_usd / 1e6:,.1f}m "
                         f"and the estimate was USD {row.prediction_usd / 1e6:,.1f}m.")
    if sensitivity is not None and not sensitivity.empty:
        lines.extend(["", "## Sensitivity to weather-point selection", "",
                      "This post-hoc exploratory check repeats the same fixed model, leaving out each weather point in turn. "
                      "The holdouts were already inspected; these checks do not supply a new independent test or select a point.", "",
                      "| Omitted point | Period | Years | MAE (USD m) | MAPE |",
                      "|---|---|---:|---:|---:|"])
        for _, row in sensitivity.iterrows():
            pct = f"{row.mape_pct:.1f}%" if np.isfinite(row.mape_pct) else "n/a"
            lines.append(f"| {row.excluded_point} | {stage_names[row.stage]} | {int(row.n_years)} | {row.mae_usd / 1e6:,.1f} | {pct} |")
        lines.append("\nThe full sensitivity results are saved in `sensitivity_private.csv`.")
    else:
        lines.extend(["", "Point-selection sensitivity was not run because per-point aggregates were unavailable."])
    lines.extend([
        "", "## Data and limitations", "",
        f"- The joined dataset contains {len(dataset)} crop years ({int(dataset.year.min())}–{int(dataset.year.max())}). "
        f"Years lacking a usable current/prior revenue pair or weather are omitted; "
        f"full coverage and provenance are in `annual_features_private.csv` and `provenance_private.json`.",
        "- Weather is the equally weighted average of three NASA POWER grid cells near Ames, Sioux City and Cedar Rapids, "
        "not a statewide measurement. Historical weather may have been revised; its availability on each forecast date is unverified.",
        "- The target is company-wide CF revenue, which includes markets and uses outside Iowa corn. "
        "It is not a direct measure of local fertilizer sales or use.",
        "- Values are as originally reported. The 2005 lag comes from a pre-IPO predecessor period; "
        "2006 Q2 was later corrected, after the 2007 prediction date.",
        ("- Longer NASA daily intervals were retrieved temporarily in browser memory; only April 1–May 20 aggregates "
         "were retained in the transferred snapshot." if used_snapshot else
         "- The direct API path requests only April 1–May 20 daily weather for each year."),
        "- Expanding-window evaluation uses only previously filed labels, a fixed 25 mm heavy-rain threshold, "
        "and within-training-fold scaling. The 2026 result is a separate single-year holdout; all test samples are small.",
        "- This retrospective comparison does not establish that rainfall causes revenue changes and is not evidence of trading value.",
        "", "Sources: [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) and "
        "[SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). "
        "Full year-by-year comparisons are in `predictions_private.csv`.", "",
    ])
    return "\n".join(lines)


def run_analysis(snapshot: dict | str | Path | None = None) -> None:
    if snapshot is None:
        print("Fetching CF public quarterly facts (raw response stays in memory)", flush=True)
        revenue = extract_revenue(request_json(SEC_URL))
        weather = None
        input_meta = {"input_type": "live API responses"}
    else:
        revenue, weather, input_meta = read_snapshot(snapshot)
        print("Loaded aggregated snapshot; no internet requests will be made", flush=True)
    verify_original_filings(revenue)
    eligible = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y in revenue.index and y - 1 in revenue.index]
    missing = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in eligible]
    print(f"Revenue coverage: {len(eligible)} eligible years; {len(missing)} omitted", flush=True)
    if len(eligible) < MIN_TRAIN + 1:
        raise ValueError("Insufficient eligible revenue history to evaluate even one out-of-sample year")
    if weather is None:
        weather = fetch_weather(eligible)
    dataset = make_dataset(revenue, weather)
    pred = evaluate(dataset)
    if pred.empty:
        raise ValueError("No out-of-sample predictions; inspect revenue coverage")
    scores = pred.groupby(["stage", "model"], as_index=False).agg(
        n_years=("year", "size"), mae_usd=("absolute_error_usd", "mean"),
        mape_pct=("absolute_pct_error", "mean"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT / "annual_features_private.csv", index=False)
    pred.to_csv(OUTPUT / "predictions_private.csv", index=False)
    scores.to_csv(OUTPUT / "scores_private.csv", index=False)
    sensitivity = point_sensitivity(snapshot, dataset) if snapshot is not None else None
    if sensitivity is not None:
        sensitivity.to_csv(OUTPUT / "sensitivity_private.csv", index=False)
    metadata = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sec_url": SEC_URL,
        "nasa_url": POWER_URL, "points_lat_lon": POINTS, "weather_window": "April 1–May 20 local solar time",
        "weather_variables": ["PRECTOTCORR", "T2M"], "cutoff": "May 31",
        "omitted_years_no_q2_pair": missing, "eligible_years": eligible,
        "ridge_alpha": ALPHA, "minimum_training_years": MIN_TRAIN,
        "warning": "NASA history may be revised; availability as of historical forecast dates is not verified. Global company sales are not Iowa corn sales.",
        "manual_checks": ["https://www.sec.gov/Archives/edgar/data/1324404/000132440425000024/cf-20250630.htm", "https://www.sec.gov/Archives/edgar/data/1324404/000104746909007107/a2193837z10-q.htm"],
        "snapshot_metadata": input_meta,
    }
    (OUTPUT / "provenance_private.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = build_report(dataset, pred, scores, input_meta, snapshot is not None, sensitivity)
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete. Read {OUTPUT / 'REPORT.md'} for results.", flush=True)


def self_test() -> None:
    def fact(year, val, filed, start=None, form="10-Q"):
        return {"start": start or f"{year}-04-01", "end": f"{year}-06-30", "filed": filed,
                "form": form, "val": val, "accn": f"test-{year}-{filed}"}
    records = [fact(2010, 100, "2010-08-01"), fact(2010, 999, "2011-08-01"),
               fact(2011, 200, "2011-08-01"), fact(2011, 300, "2011-08-01", "2011-01-01"),
               fact(2012, 123, "2014-08-01")]
    doc = {"facts": {"us-gaap": {"SalesRevenueNet": {"units": {"USD": records}}}}}
    rev = extract_revenue(doc, 2010, 2012)
    assert list(rev.index) == [2010, 2011] and rev.loc[2010, "q2_usd"] == 100
    days = pd.date_range("2024-04-01", "2024-05-20")
    values = {x.strftime("%Y%m%d"): 1.0 for x in days}
    weather_payload = {"properties": {"parameter": {"PRECTOTCORR": dict(values), "T2M": dict(values)}}}
    aggregate = summarize_point(weather_payload, 2024)
    assert aggregate == {"rain_mm": 50.0, "heavy_rain_days": 0, "temp_c": 1.0}
    weather_payload["properties"]["parameter"]["PRECTOTCORR"].pop("20240401")
    try:
        summarize_point(weather_payload, 2024)
        raise AssertionError("Missing daily weather should fail")
    except ValueError:
        pass
    train = pd.DataFrame({"prior_q2_usd": np.arange(1, 9) * 1e6,
                          "rain_mm": np.arange(8), "heavy_rain_days": np.arange(8) % 3,
                          "target_q2_usd": np.arange(2, 10) * 1e6})
    assert np.isfinite(ridge_predict(train, train.iloc[-1], FEATURES))
    print("Synthetic checks passed: original revenue selection, missing coverage, and ridge prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="Path to compact pre-aggregated JSON input")
    args, _ = parser.parse_known_args()
    if args.self_test:
        self_test()
    else:
        embedded = globals().get("SNAPSHOT_JSON")
        default_snapshot = Path("/kaggle/working/input_snapshot.json")
        snapshot = args.snapshot or embedded or (default_snapshot if default_snapshot.exists() else None)
        run_analysis(snapshot)
