# Point-in-time replay

`replay.py` is an offline, fail-closed provenance ledger and decision-date
replay. Pass explicit observations, source versions and required keys. It
returns a value only when its supplied provenance supports use by the stated
decision date; the command returns exit status 2 and a report when an input is
missing, late, ambiguous, or unverified. Unknown weather publication times and
reconstructed historic prices are never silently promoted to verified inputs.

## Record format

Each JSON record contains `series`, `period`, a finite numeric `value`,
`observation_date`, HTTPS `source_url`, `source_version`, `available_at`,
`retrieved_at`, and `evidence_type`. Dates are ISO `YYYY-MM-DD`; timestamps
must include a UTC offset. Null availability or retrieval timestamps are
explicit unknowns. When supplied, `sha256` must be a lowercase SHA-256 string.
`captured` records additionally require that digest and a `capture_reference`.
Provide `--capture-root` with a directory containing the captured
files. References can be paths relative to that directory or absolute paths
inside it. The offline replay reads each referenced regular file beneath the
explicit root and compares its current SHA-256 with the recorded digest.
Missing files, checksum mismatches, symlinks, paths outside the root, files
larger than 16 MiB and systems without safe directory-relative open support
are rejected. An HTTPS-only reference is not fetched and is unverified; save
compact evidence first. If a verified-document or as-of-vintage record
supplies a capture reference or digest, that artifact is checked by the same
rules; incomplete or mismatched optional evidence cannot be ignored. A hash
matches current bytes only: it does
not prove the source was authentic, that the bytes existed on the claimed
capture date, or when the observation became public. Independently check the
source, capture time, content and completeness.

Evidence types are:

- `captured`: availability and actual retrieval occurred by cutoff, with a
  specified version and a reference to locally hash-verified captured bytes.
  These are caller assertions; a saved observation is not necessarily the
  latest available version.
- `verified_immutable_document`: availability date, document version,
  retrieval time, `source_verified: true`, and a `verification_note` are supplied. The note
  should identify how the original publication and value were checked.
- `verified_asof_vintage`: as above, plus `vintage_as_of` exactly matches the
  requested cutoff at the same precision. A current or later vintage is not
  substituted for a missing historical one.
- `unverified`: a transcription or reconstruction whose content or version
  has not been independently verified. It never passes strict replay.

Standalone replay permits independently asserted verified-document or
as-of-vintage evidence without an attached artifact by default. Pass
`--require-capture` (and `--capture-root`) to require a hash-checked local
artifact for every supported evidence type; the full workflow uses this
stronger gate for its `strict_replay_complete` status. Even then, the program
does not verify that the bytes contain the record's value, that the source was
genuine, or that the bytes existed before the cutoff. Independent source- and
time-specific verification is still required.

These checks validate provided assertions. They cannot independently prove
that a source was truly available, the supplied archive contains every
candidate version, or a recorded value was in the original source. Source
documents, release metadata and candidate completeness still need review. If
there are multiple candidate versions with unknown timing or unverified
provenance, selection is blocked, even when unverified metadata claims a
late availability date. That claim does not prove the candidate was
unavailable on the decision date. A sufficiently supported strictly later
version is excluded, and
known earlier versions are ordered only when their availability precision is
comparable and unambiguous.

For a derived record, `constituent_keys` lists each required `[series, period]`
in calculation order. A caller cannot bypass the component checks by requesting
only the aggregate: replay follows the dependency list automatically, rejects
cycles, blocks the aggregate if a component is unsupported, and checks a
declared `ratio` or `mean` against the selected component values. The existing
forecast average allows a small rounding tolerance; ratio checks use a tight
numerical tolerance.

The project-specific nitrogen price and Q1 shipment ratios must use
`derivation: "ratio"` with exactly two ordered constituent keys for the same
year: current observation first, previous observation second. Their required
series are `nitrogen_price_current_april` and
`nitrogen_price_previous_april`, or `q1_shipments_current` and
`q1_shipments_previous`, respectively. A verified label on a bare ratio does
not substitute for the components. These named price indices, shipment volumes
and their ratios must be strictly positive, matching the comparison model's
input requirements; invalid values fail validation even if two negative values
would yield a positive ratio. A derived record is rejected if its
claimed availability is provably earlier than any selected constituent's
availability. A date-only value paired with an exact timestamp on the same UTC
day has unknown ordering and is rejected as `derived_availability_order_unknown`,
even if both are before the decision cutoff. Two date-only values on the same
day retain calendar-day semantics; that check does not establish intraday order.

## Weather feature recipe

The project feature `forecast_probability` must be a `mean` of exactly the
same-year `forecast_station_KAMW`, `forecast_station_KCID` and
`forecast_station_KSUX` keys. Each station mean must in turn be a `mean` of
40–50 distinct daily keys named `forecast_daily_<STATION>`, where the `period`
is the initialization date in ISO form. Runs must fall from April 1 through
May 20 of that year. These limits match the original study and do not make
different missing dates harmless.

The annual mean, every station mean and every daily record must each declare
the same feature definition: `model: "GFS"`, `element: "P12"`,
`units: "percent"`, `lead_start_hours: 12`, `lead_end_hours: 24`,
`precipitation_threshold_inches: 0.01` and
`precipitation_basis: "liquid_equivalent"`. P12 is the probability of at
least 0.01 inch of liquid-equivalent precipitation during lead hours 12–24.
A record for another forecast model, element, unit, threshold or interval is
not the project's feature. These explicit fields are caller-supplied metadata;
they do not independently establish the source contents or publication time.

Each daily record needs the normal provenance fields above, plus `station`,
`model_run_utc` and `valid_at_utc`. The station must match its series, and
`observation_date` and `period` must both match the run date. The run must be
00:00 UTC and its valid time exactly 24 hours later. For example, a May 20
initialization has a May 21 valid time; the window constrains initialization,
not valid date. The numeric probability must be between 0 and 100. The
selected daily and aggregate records each still need acceptable independent
availability/capture or vintage evidence by the relevant decision cutoff;
the model initialization time is **not** assumed to be the publication time.

Missing declared dependencies, duplicate keys, fewer than 40 runs, incorrectly
dated runs or wrong-station dependencies block the feature; selected component
values must rebuild each mean. Candidate record
completeness and verification flags remain caller assertions. A later capture,
checksum, or an archived run timestamp alone does not establish past
availability or that no missing candidate or revision exists. The existing
project adapter deliberately leaves historical forecast summaries unverified;
its matching-row adapter annotates the feature definition only on reconstructed
groups and refuses conflicting existing metadata. Adding this recipe does not
automatically certify or replace those records.

A calendar-day cutoff treats the whole UTC calendar day as the decision period;
it does not certify what was available at a specific hour. For an exact-time
cutoff, a same-day date-only release is rejected as intraday-unknown. For
same-day financial data, establish a suitable decision time and source timezone
before making an intraday claim.

Replay also rejects a retrieval that is definitely earlier than its stated
availability. When date-only and timestamp metadata fall on the same UTC day,
the intraday ordering is unknown; passing that consistency check does not prove
the order. A forecast can describe a future period, so observation dates are
not categorically required to precede forecast issuance.

For the six known realized project series (company and phosphate Q2 sales,
current/prior Q1 shipments, and current/prior April price-index constituents),
the year key must agree with the recorded quarter-end or April reference-month
date. Their availability cannot provably precede that date. These checks do not
prove the source is genuine and do not apply the same rule to future-valid
weather forecasts. The April date identifies a month, not an exact measurement
time or a demonstrated release date.

## Run

Prepare JSON arrays of provenance records and required pairs, e.g.
`[["q2_sales", "2021"], ["nitrogen_price_ratio", "2022"]]`. Run:

```sh
python replay.py --ledger /private/path/provenance.json \
  --requirements /private/path/required.json --cutoff 2022-05-31 \
  --capture-root /private/path/captures \
  --output /private/path/replay_report.json
python -m unittest -v replay_tests
```

The first command saves the detailed report even when it exits with status 2.
It requires only the Python standard library; the project adapter below accepts
notebook pandas frames but `replay.py` does not import pandas.

## Existing notebook integration

Execute `replay.py` as definitions (do not invoke its command-line `main`),
then call:

```python
records = project_ledger(
    DECOUPLED_FORECAST_RESULTS['data'], SNAPSHOT_JSON, EXTRA_POINTS['q1_volume'],
    FORECAST_INPUT_AUDIT['manifest'], HISTORICAL_PRICES, PHOSPHATE_SEGMENT,
    browser_price_evidence=BROWSER_PRICE_RESULTS['observations'])
report = project_replay_report(records, range(2022, 2027))
fold = project_fold_replay(records, 2022, [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
save_private_ledger(records, report, '/kaggle/working/iowa_replay')
```

Pass the actual selected training years from each model's fold, not the example
list, to `project_fold_replay`. It checks training features against each
training year's own date and training outcomes against the current fold's date,
including old phosphate adjustment components. Save its report separately if
needed. It does not imply those records actually passed: the current snapshots
are deliberately marked `unverified`, and the captured forecast station
summaries lack original historical publication times. A strict historical
replay is incomplete until source/vintage evidence is supplied. The metadata
and missingness report are useful evidence of this limitation, not a way to
fill the gaps. The frozen 2027 protocol is untouched.

The ledger keeps separate records for current and previous April nitrogen
producer-price observations (including the period, available release dates,
source and recorded vintage), their derived ratio, each of the three forecast
station summaries and their mean, the current and previous Q1 shipment values
and their ratio, prior Q2 sales, and older phosphate components. Individual
price and station records are required alongside the aggregate in decision and
training-feature checks; an unavailable constituent is not inferred from a
ratio or mean. For newer price observations, any browser-recorded as-of graph
transcription retains its source URL, value, capture interval and transcription
checksum. That interval is the later audit time, not the historical release
time; the checksum covers the transcription, not the original source bytes.
The station records reference the existing compact daily-forecast audit output.
