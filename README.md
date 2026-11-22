# Can Iowa weather forecasts improve a fertilizer revenue nowcast?

[Open the live Iowa fertilizer research report](https://forecast-observatory-iowa-2026.humorouspixie2.chatgpt.site/)

This research compares simple estimates of CF Industries' second-quarter company-wide net sales using information available by May 31. It focuses on Iowa's corn-growing season, but CF sells across many regions and markets. Company revenue is not a direct measure of Iowa fertilizer purchases, crop use, profit or stock returns.

The results are **exploratory, not a trading strategy**. Every year through 2026 has already been examined while developing the analyses; none is an untouched final test. The project does not establish that rain causes fertilizer sales.

## Explore the project

- [Research code, notes and saved results](research/)
- [Standalone forecast runner](research/FORECAST_CORE_README.md)
- [Matched comparison and source audit](research/research_audit/README.md)
- [Delivery estimate code](research/iowa_delivery_estimates.py)
- [Live-site source](docs/index.html)

## What the comparison found

The matched comparison estimates the same 11 years with four methods: a price-only baseline, price plus an Iowa weather forecast feature, price plus a damped Q1 shipment-volume adjustment, and price plus both adjustments. The table shows mean absolute errors in millions of US dollars; smaller is better.

| Evaluation years | Price | Price + weather | Price + shipments | Price + both |
|---|---:|---:|---:|---:|
| 2016–2021 (6 years) | 107.3 | 109.0 | 163.4 | 166.0 |
| 2022–2026 (5 years) | 147.0 | 148.7 | 113.4 | 116.9 |

The weather addition did not reduce average error in either period. The shipment adjustment improved the later period but performed worse in the earlier one. It was selected after examining historical outcomes, so the apparent improvement is not independent evidence of future performance. Small samples, changing business conditions and prior model selection make the estimates uncertain. Other experiments use different cohorts and should not be compared directly to this table without checking their methods.

## Methods and limitations

- **Target and timing:** originally reported CF Q2 net sales, estimated on May 31. This is a nowcast: part of the quarter has already passed.
- **Price baseline:** previous Q2 revenue multiplied by the ratio of current to previous April's nitrogenates producer-price index. A historical vintage should be used where possible.
- **Weather:** three Iowa stations' GFS MOS forecasts initialized April 1–May 20. The selected P12 feature is the probability of measurable precipitation during lead hours 12–24; it is not rainfall amount. Coverage can vary across station-years.
- **Shipments:** the fixed adjustment is the square root of the current/prior Q1 company product-ton ratio. It is a research assumption and was chosen after inspecting historical results.
- **Evaluation:** models are compared on matched years, with earlier filed outcomes used for training. Historical forecast publication times, possible archive revisions, and some other point-in-time evidence remain unverified. A checksum verifies retained bytes, not historical availability. The replay audit therefore does not certify a complete historical reconstruction.
- **Interpretation:** a few annual observations cannot reliably establish a relationship, and changing prices, company operations, geography, buying and delivery timing can all affect revenue. The simplified Iowa weather feature cannot identify individual farmer decisions.

## Data and reproducibility

The underlying sources are public. The saved notebook and selected research outputs are included in this repository; see [PUBLICATION_CONTENTS.md](research/PUBLICATION_CONTENTS.md). **The notebook is a record of the saved run, not a fully standalone reproducible bundle:** some cells refer to separately retained Kaggle inputs. Those inputs have not automatically been made public with this repository.

Main sources:

- [SEC filings](https://www.sec.gov/search-filings) for CF sales and shipment disclosures.
- [ALFRED nitrogenates price index](https://alfred.stlouisfed.org/series?seid=WPU065201) for historical price vintages.
- [Iowa Environmental Mesonet MOS archive](https://mesonet.agron.iastate.edu/mos/) and [NOAA's MOS format guide](https://www.weather.gov/media/mdl/mdltpb05-03.pdf) for forecast data and definitions.

For the standalone forecast calculations, use Python 3.10 or newer and install the dependencies in `requirements-forecast-core.txt`. Run the synthetic offline tests from the research folder.

```sh
cd research
python -m pip install -r requirements-forecast-core.txt
python -m unittest -v test_forecast_core
```

See [FORECAST_CORE_README.md](research/FORECAST_CORE_README.md) for input fields and the standalone runner; [research_audit/README.md](research/research_audit/README.md) describes the matched comparison and replay; and [research_audit/PROVENANCE_FINDINGS.md](research/research_audit/PROVENANCE_FINDINGS.md) documents source checks and remaining gaps. Synthetic tests verify code behavior, not the historical truth or completeness of the input data.

## Prepared 2027 test

The pre-specified main test uses the price-only formula on May 31, 2027, with a fallback to prior Q2 revenue if the April price index is unavailable. A damped Q1 shipment adjustment is a separate sensitivity when the required price and shipment data are available. Neither formula uses weather. Record the sourced inputs and prediction before CF releases Q2 results, and score it afterwards. No 2027 prediction has been made. One future year will be informative but cannot establish reliable performance on its own. See [README_2027.md](research/README_2027.md) and [protocol_2027.py](research/protocol_2027.py).
