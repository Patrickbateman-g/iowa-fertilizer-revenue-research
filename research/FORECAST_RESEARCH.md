# Historical weather-forecast research

This is an exploratory comparison against the existing fertilizer-price-only
model. The prepared Q2 2027 prediction protocol remains unchanged. The years
through 2026 had already been examined, so this comparison is not an untouched
test and cannot establish that forecasts will improve future predictions.

## Forecast feature

Use the [Iowa Environmental Mesonet GFS MOS archive](https://mesonet.agron.iastate.edu/mos/)
for three fixed Iowa stations: KAMW, KCID and KSUX. For every date from April 1
through May 20, inclusive, select the 00:00 UTC model initialization and its
`p12` value for the forecast time 24 hours later. This is the probability of
measurable precipitation during the 12 hours ending at the forecast time. It is
not a rainfall amount. See the [NOAA MOS product reference](https://vlab.noaa.gov/web/mdl/mav-card).

Average valid days within each station, then average the three station means
equally. Each station-year must have at least 40 of the 50 expected days.
Repeated identical values are counted once; days with conflicting values are
excluded and counted. Missing or invalid data are not imputed. The station-year
coverage table documents the omissions and source URLs.

Model initialization time is not independently verified publication time, and
the archive could have been revised. The available data do not establish what
any particular farmer saw or whether a forecast led to an order. Input values
for this run are compact station-year summaries derived through the browser; the
notebook validates their structure and source parameters but does not
independently reproduce the raw-record aggregation.

## Matched-year comparison

The price-only baseline multiplies the previous year's Q2 net sales by the
current-to-previous April nitrogenates producer-price index ratio, using the
May 31 vintage. The experimental model applies a single weather adjustment to
that baseline. Its coefficient uses only earlier, already-reported years and a
training-only mean and standard deviation, with a fixed ridge penalty of one
and no intercept. Six prior complete years are required, so at most five years
can be evaluated in 2022–2026. Both candidates are compared on the same years.
Show the annual errors, coverage and training counts, not just average error.

The small number of years, earlier inspection of the data, forecast coverage,
and the single feature choice limit any inference. Results describe
association, not whether forecast rain caused farmer purchases or company sales.
Consider a promising result only as a separate research candidate for a future
test, not as a reason to replace the existing frozen 2027 model.

## Reproduction and storage

Run the existing Kaggle follow-up notebook cells to produce
`V2_RESULTS`, then run `analysis_forecast_research.py` as a notebook code cell
and load the compact `FORECAST_SNAPSHOT` from the notebook. The notebook
calls `snapshot_forecasts(FORECAST_SNAPSHOT)` followed by
`evaluate_forecasts(V2_RESULTS, annual, coverage)`. See the notebook for
the inputs, coverage, results and report.

Only small derived annual summaries, predictions, provenance and reports are
saved under `/kaggle/working/iowa_forecast` and in a **saved Kaggle
version**. Raw weather responses are handled one station-year at a time without
being saved locally. The repository contains analysis code and this method note,
not the data or results.
