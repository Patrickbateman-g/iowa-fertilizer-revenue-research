# CF Q2 2027: prospective test

This is a manual checklist for the Kaggle notebook. No 2027 data or forecasts are included yet. Both formulas were fixed after seeing earlier outcomes, so only a genuinely new forecast captured before disclosure can be a new test. Q2 is April 1–June 30, and the fixed forecast date is **May 31, 2027, UTC**. All May 31 times below mean UTC: 00:00:00 to 23:59:59.999999 UTC.

The previous-year baseline is CF's originally reported consolidated Q2 2026 net sales. The main formula multiplies that figure by the April 2027 nitrogenates producer price index divided by the April 2026 index, **both from the same May 31, 2027 ALFRED WPU065201 vintage**. The experimental sensitivity multiplies that main forecast by the square root of the ratio of current and previous Q1 product sales volume, in thousand product tons. If the April index is missing, the main forecast falls back to prior Q2 revenue and the volume sensitivity is not made. If volume is missing, only the main forecast and prior-Q2 baseline are made. These rules must not be revised after seeing the 2027 result. The optional Q1 customer advances field is descriptive company-wide context, not total orders and not an input to either formula.

## Purchase, payment and revenue timing

CF sells to cooperatives, retailers, distributors and industrial buyers, among others. It generally recognizes revenue when control transfers, commonly upon loading or delivery; cash receipt is not necessarily the revenue date. Credit terms are generally 30 days or less, while forward sales can involve a deposit before delivery, recorded as a customer-advance liability until control transfers. Terms and timing vary. [CF 2025 annual report](https://www.sec.gov/Archives/edgar/data/1324404/000132440426000007/cf-20251231.htm).

Farmers may buy fertilizer through an intermediary months before spring planting and apply it at a different time. The USDA describes purchases before most U.S. corn planting; this is not an Iowa-specific measured delay. [USDA ERS](https://ers.usda.gov/data-products/charts-of-note/108828). Iowa State describes spring and, when conditions allow, fall application. [Iowa State](https://crops.extension.iastate.edu/cropnews/2015/10/fertilizer-nitrogen-application-fall). Consequently, rain, field use, farmer purchases, dealer purchases, CF deliveries, company revenue and payment need not fall in the same quarter. The length and direction of these gaps vary, and these sources do not justify an automatic fixed lag.

As a potentially useful future research clue, CF reported $132 million of company-wide customer advances at March 31, 2026 in its [Q1 2026 filing](https://www.sec.gov/Archives/edgar/data/1324404/000132440426000013/cf-20260331.htm). Such deposits are a partial pipeline indicator, not total contracted sales or guaranteed next-quarter revenue. If the Q1 2027 balance is publicly disclosed by the capture time, record it as optional context; do not change the pre-specified 2027 formulas or imply that the balance has been validated as a predictive feature. Investigate its historical usefulness separately after this prospective test.

## Before May 31

1. Run the protocol code cell so its functions are defined. No 2027 prediction is created at this stage.
2. Record CF's originally reported Q2 2026 net sales in USD and the SEC filing or original earnings announcement link and publication date. Record CF's originally reported Q1 2026 product sales volume and source; Q1 2027 must wait for its own release. Use product tons, not nitrogen nutrient tons. Use the original Q2 2027 earnings release only for later scoring, never for the prediction inputs.
3. On May 31, inspect the [ALFRED nitrogenates price series](https://alfred.stlouisfed.org/series?seid=WPU065201). Select its May 31, 2027 vintage **at the time of the live check**, record the exact values for April 2027 and April 2026, the first publication date of the 2027 April value, the source link and the UTC time you observed the values. A later download of that vintage does not prove what was visible at the earlier capture time; record contemporaneous evidence. The date-only vintage is not independent proof of the exact time. Do not use a future revision.
4. Record the Q1 2027 and Q1 2026 volumes, publication dates and source links from CF/SEC filings already public by the capture time. If available, record the 2027 Q1 customer advances balance and its filing date and source as optional context. Deposits may reflect purchase timing but do not represent the whole order book or prove causation.

## On May 31: capture

In a notebook, after running the protocol code cell, replace **every** `None` below with a real sourced value. Strings with fixed dates identify the observation periods and are not predicted data. Use source links to the original SEC filing or CF release. Do not put the 2027 Q2 actual in these inputs. Capture times must be ISO UTC strings such as `2027-05-31T16:45:00Z`, recorded when you actually observed the data, and not later than the forecast creation time.

```python
inputs = {
  'prior_q2': {'value': 2222000000, 'units': 'USD', 'period_start': '2026-04-01',
               'period_end': '2026-06-30', 'published': '2026-08-05',
               'source': 'https://www.sec.gov/Archives/edgar/data/1324404/000132440426000017/cf-08052026_ex991xearnings.htm'},
  'price': {'available': True, 'series': 'WPU065201', 'vintage': '2027-05-31',
            'current_observation': '2027-04-01', 'previous_observation': '2026-04-01',
            'current_value': None, 'previous_value': None,
            'current_first_published': None,
            'source': 'https://alfred.stlouisfed.org/series?seid=WPU065201',
            'captured_at': None},
  'q1_volume': {
    'available': True,
    'current': {'value': None, 'units': 'thousand product tons',
                'period_start': '2027-01-01', 'period_end': '2027-03-31',
                'published': None, 'source': None},
    'previous': {'value': 4683, 'units': 'thousand product tons',
                 'period_start': '2026-01-01', 'period_end': '2026-03-31',
                 'published': '2026-05-06',
                 'source': 'https://www.sec.gov/Archives/edgar/data/1324404/000132440426000011/cf-05062026_ex991xearnings.htm'}
  }
}

# Optional: include only if CF has disclosed it by the capture time.
# inputs['q1_customer_advances'] = {
#     'value': None, 'units': 'USD', 'balance_date': '2027-03-31',
#     'published': None, 'source': None
# }
# If not disclosed, optionally record the reason instead:
# inputs['q1_customer_advances'] = {'available': False, 'reason': 'Explain why this was unavailable'}

forecast = capture_2027(inputs)
print(forecast['prediction']['main_prediction_usd'])
print(forecast['prediction']['sensitivity_prediction_usd'])
```

If the April index is not available on May 31, **replace the whole `price` object** with:

```python
inputs['price'] = {'available': False, 'reason': 'Explain the missing April index',
  'checked_at': None, # Actual May 31 UTC check time, e.g....Z
  'source': 'https://alfred.stlouisfed.org/series?seid=WPU065201'}
```

If either required Q1 volume is unavailable by that time, **replace the whole `q1_volume` object** with `{'available': False, 'reason': 'Explain which Q1 figure is missing'}`. Missing price or volume must be recorded; do not silently substitute revised or later information.

`capture_2027` writes `/kaggle/working/cf_2027_test/prediction_2027.json` once; it refuses to overwrite that filename. The production call reads Kaggle's UTC system clock and refuses capture outside May 31. The generated hash detects accidental changes, but is not a digital signature, and local files/system clocks can be changed. Manual metadata checks cannot independently prove the inputs were public.

**Immediately save a new Kaggle version, including this output, on May 31 and before the Q2 release.** A file in `/kaggle/working` alone is temporary. A saved Kaggle version provides a useful external timestamp, but its history and access controls are not a cryptographic guarantee. Preserve the version number and source evidence. Ensure any Save & Run All finishes before the cutoff; the live capture cell intentionally refuses to run after the cutoff. Do not replace the frozen forecast with a later run.

## After CF publishes Q2 2027 net sales

Use a separate later notebook session; attach the output of the saved **pre-disclosure prediction version** as an input, rather than rerunning the capture cell. Find the actual attached path in Kaggle's Input panel. Fill in the originally disclosed result and run:

```python
actual = {'value': None, 'units': 'USD', 'period_start': '2027-04-01',
          'period_end': '2027-06-30', 'published': None, 'source': None}
# Replace with the path shown under Input for the archived prediction version.
archived_forecast_path = '/kaggle/input/REPLACE_WITH_SAVED_VERSION_PATH/cf_2027_test/prediction_2027.json'
score = score_2027(archived_forecast_path, actual)
print(score['results'])
```

The score is written once to `/kaggle/working/cf_2027_test/score_2027.json`; save a **new Kaggle version** for it. The code verifies the forecast hash, recomputes the frozen formulas, and rejects a Q2 release before July 1 or after the scoring date. Verify independently that the prediction version was saved before the actual disclosure. One Q2 observation cannot establish whether either forecast is reliably better.
