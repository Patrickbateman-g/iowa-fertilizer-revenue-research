# Reproducible retrospective workflow

`workflow.py` runs the matched four-model comparison, sequential 80% nominal
error ranges, a fold-by-fold point-in-time provenance replay, reconstructed
experiment register, and concise research report without notebook globals or
network access. The test period is restricted to 2016–2026; it does not produce
a 2027 prediction or change the frozen 2027 test.

Install Python 3.10+, NumPy and pandas. The adjacent project dependency list
`requirements-forecast-core.txt` specifies compatible ranges. The following single command uses **example paths**;
change them to the actual locations:

```sh
python workflow.py \
  --annual /private/annual_adjusted.csv \
  --shipments /private/q1_shipments.json \
  --ledger /private/provenance_ledger.json \
  --artifacts /private/existing_study_outputs \
  --output /private/new_research_output
```

`--output` must be a new path or an empty directory and must be outside a Git
checkout or worktree. An output directory within the artifacts root is
supported; the inventory is assembled before new output is written. No raw
inputs are copied except that the provenance ledger itself is intentionally
saved in the replay output. The original annual and shipment files are
not copied.

Use `--first-year` and `--last-year` to request a consecutive subset of
2016–2026. The default covers all 11 years, with four matched model predictions
per year. The program fails if a requested annual or Q1 shipment row is missing,
duplicated or unusable, a recorded Q1 shipment release is late, fewer than six
previous filed training targets are available, or any variant lacks a matching
prediction. The exploratory run can still calculate a prediction when its
prior-Q2 baseline has missing, late or unverified source evidence; the replay
reports that status and strict replay fails unless the baseline is supported
at its own cutoff. A later 10-Q date alone does not rule out an earlier earnings
release; each source's timing must be checked independently. It
checks a supplied ledger against actual inputs, including eligible selected
versions where available. It checks *candidate values*, source hashes, and
stated eligibility; these are not independent confirmation of the underlying
sources, and an incomplete ledger cannot show that another version did not
exist.

For a subset beginning after 2016, the workflow also attempts to recreate
matched earlier folds starting in 2016 as range-calibration candidates. A fold
can be skipped because its annual or shipment input is missing, too few earlier
filed training years exist, or training weather has no variation. Preparation
alone is not provenance verification. The range uses a prior model-matched
error only when that outcome's recorded filing date is by the prediction
cutoff. It reports predictions and coverage only for the requested years; the
earlier folds are calibration context. The 2016 starting point is the boundary
of this study, not a claim that no earlier forecast can exist.

Before writing the report, the workflow recomputes each displayed range from
the supplied model-matched prior predictions and filed outcomes, and verifies
its calibration years, 80% rank, radius and bounds. For a subset, it supplies
the earlier prepared folds as well as the requested rows; missing earlier
folds are not invented. This is an arithmetic and chronology check on supplied
records, not proof that those predictions were genuinely made out of sample or
that their inputs and filing dates were historically available as claimed.

For every earlier fold that contributes to a requested range, the strict runner
also audits its prediction-time inputs, training cohort and scored outcome.
Missing or unverified evidence prevents strict certification rather than being
silently discarded. Range results remain exploratory unless those checks can
be independently supported.

The original Iowa weather notebook is a separate artifact source. When mounted
in Kaggle, include its outputs alongside the follow-up run using:

```sh
--artifacts /kaggle/working \
--additional-artifacts original=/kaggle/input/notebooks/nicolasrster/iowa-corn-fertilizer-revenue-research
```

Add that option to the command above. `--additional-artifacts NAME=DIRECTORY`
can be repeated for other sources. Names must be unique and contain
only letters, numbers, `-` and `_`; `primary` is reserved for `--artifacts`.
Artifact roots must be distinct, non-overlapping existing directories. Register
IDs and source paths are prefixed with `primary/` or the selected name, so the
same relative file in two notebooks is not silently merged. Each input root
must contain at least one recognized result artifact.

By default a missing, unknown, or unverified historical source does **not**
prevent an exploratory analysis: the manifest reports incomplete replay and
the report states its limitations. To refuse to write any output unless every
required feature, target and phosphate constituent has eligible replay
evidence and matching values, add `--strict-replay`. The real historical
archive currently contains unverified sources, so strict replay should be
expected to fail until actual provenance is independently supplied. Do not
convert a missing release time into an inferred one merely to pass the check.

For ledger records labeled `captured`, pass `--capture-root /private/captures`.
Their `capture_reference` must name a regular file within that selected root;
the offline replay recomputes its SHA-256 and rejects missing or altered files,
symlinks, outside-root paths and files larger than 16 MiB. External URLs alone
cannot pass the byte check. Without a root, captured records remain unverified.
If a verified-document or as-of-vintage record supplies a capture reference
or digest, that artifact is checked by the same rules; incomplete or mismatched
optional evidence cannot be ignored. The workflow requires such a matching
local artifact for **every** evidence
type, including verified documents and historical vintages, before its strict
replay status can be complete. Ordinary standalone replay can instead accept a
verified document or vintage without a saved artifact; that is a weaker,
explicit caller-assertion mode. Neither mode checks whether the saved bytes
actually establish a record's value or claimed time.
The digest checks current bytes, not source authenticity, historic publication,
capture time or completeness of versions. The other verified evidence types
remain caller assertions requiring independent source review.

The strict runner also audits every evaluated Q2 outcome and its claimed
`target_filed` day, including the last test year. The manifest contains
`evaluation_outcome_requirement_counts`; detailed evidence is saved as
`replay/evaluation_outcomes_private.json` and summarized in the report. The
current filing-date check requires the selected source's availability day to
match the supplied filing day. An earlier earnings release can establish that
a value was known by the later filing but cannot by itself corroborate that
filing day, so `filing_date_mismatch` is an evidence limitation, not proof of
an incorrect revenue figure. Model separate earlier releases and SEC filings
explicitly rather than equating their dates.

The runner also audits every earlier annual candidate, including those omitted
from a fold, against the declared target filing date and the decision cutoff.
It reports when an omitted outcome was already available, or when the reason
for exclusion cannot be verified. The detailed file is
`replay/training_cohorts_private.json`; status totals are in the manifest and
report. Later documents are used only to check a historical cohort and do not
become model inputs. A documented earnings release earlier than a claimed SEC
filing needs the two dates modeled separately; a mismatch is not by itself
proof of a wrong value. The audit also compares the value available at the
claimed filing date with the annual target, including the historical phosphate
adjustment where applicable, so a later matching revision cannot hide an
earlier value mismatch. All findings are conditional on the supplied source
verification and candidate set being complete.

## Input formats

* `--annual`: CSV with one row per year. Columns are `year`, `as_of` (the
  exact ISO May 31 date), `target_filed` (ISO, after Q2), `target_q2_usd`,
  `prior_q2_usd`, `price_ratio`, and `forecast_mean_p12_pct`. Values must
  already reflect the intended cohort and the historical phosphate adjustment
  for 2010–2015; the workflow does **not** fetch source data or compute that
  adjustment. Recent target years remain company-wide reported figures. The
  v20 notebook exports its matching adjusted table as
  `iowa_intercept_research/adjusted_annual_inputs_private.csv`.
* `--shipments`: JSON array, one object per test year. Required fields are
  `year`, `current_q1_tons`, `previous_q1_tons`, `units` (exactly
  `thousand product tons`), `current_filed`, `previous_filed`,
  `current_source`, and `previous_source`. Filing dates are ISO dates and
  sources must be HTTPS SEC filings for CF. The fixed multiplier is the square
  root of current/prior Q1 tons. Source data and Q1 releases are company-wide.
* `--ledger`: JSON array matching the schema and evidence rules in
  `REPLAY_README.md`. It includes the source series, period, value, observation
  date, source URL and version, availability and retrieval times, and evidence
  type. For each test fold replay asks for prior-Q2 sales, current/prior April
  nitrogen prices and their ratio, three station forecasts and their aggregate,
  and current/prior Q1 shipments and their ratio. For each training year it
  also audits inputs as of that year's cutoff, the label as of the test cutoff,
  and relevant historic phosphate components. The supplied ledger must cover
  all candidate versions; the workflow cannot prove its completeness. To pass
  strict weather replay, the overall mean must depend on the three named
  station means, and each station mean on 40–50 distinct daily forecast
  records with correctly named station, initialization dates from April 1
  through May 20, 00:00 UTC run time and a valid time exactly 24 hours later.
  The final May 20 run therefore has a May 21 valid time. Each record and
  derived mean must have eligible evidence at the decision cutoff. A run time
  is not an assumed publication time. The nitrogen-price and Q1-shipment
  ratios also require their two ordered same-year constituents. Derived values
  and their stated availability are checked against those components. If a
  derived record and an input mix a same-day date and exact timestamp, their
  ordering is unknown and replay rejects the derived record. Same-day dates
  retain calendar-day semantics, which do not prove intraday order. See
  `REPLAY_README.md` for exact field names and important evidence limits.
  Existing unverified project records are not automatically promoted.
* `--artifacts`: an existing directory containing retained prior
  studies (such as `iowa_v2/scores_private.csv`). At least one recognized score,
  prediction, diagnostic, sensitivity, leave-one-out, skipped, omitted, paired,
  win-count, or annual-error CSV must be present. Skipped and omitted records
  retain their saved reasons; they do not automatically mean that a model failed.
  Empty artifacts are identified as empty rather than as failures.
  Only retained files are indexed; unrecorded experiments and artifacts under
  other filenames may be absent. Historic motives and proposal-time knowledge
  are not reliably recoverable, even if a register has a known creation date.

## Outputs and validation

The output contains `comparison/`, `uncertainty/`, `replay/`, `register/`,
`report/RESEARCH_REPORT.md`, and `manifest.json`. The report gives each annual
prediction and signed error, matched-period comparisons, empirical range
coverage and replay status counts. The manifest includes source checksums and
eligibility counts; checksum agreement establishes byte identity, not
historical availability. Nominal 80% is a construction target, not a
coverage guarantee: with so few independent years and selected models, ranges
may be unstable or unavailable.

`uncertainty/calibration_inventory_private.csv` lists candidate earlier folds
and whether each was used, skipped or prepared but not used;
`uncertainty/calibration_predictions_private.csv` contains the earlier matched
predictions actually used by requested ranges. The manifest records
`calibration_years_used`, `calibration_skipped_folds`, and separate calibration
replay, outcome, cohort and filing status counts. The detailed checks are kept
in `replay/calibration_replay_private.json`,
`replay/calibration_outcomes_private.json`, and
`replay/calibration_cohorts_private.json`. Coverage tables count only requested
years with a computable range and known outcome; period labels show the years
actually present.

Run offline synthetic checks from the module directory:

```sh
python -m unittest -v test_comparison test_intervals test_register test_experiment_log test_notebook_log_runner test_workflow test_report replay_tests test_alfred_chart test_bls_capture test_sec_capture test_sec_index_capture test_shipment_capture test_source_link_state test_calibration test_calibration_workflow test_daily_ledger
```

They exercise matched years, no use of future outcomes, cutoff checks, duplicate
and missing records, range coverage, inventory hashes, a complete synthetic CLI
run, refusal to overwrite or write under Git, and a case where a matching older
candidate differs from the version actually selected by replay. They also
check the forecast recipe and constituent timing, subset calibration and
its provenance. These tests do not independently verify historical weather or
company disclosures.

`bls_capture.py` can parse compact saved BLS release excerpts and compare
their April nitrogenates values against a ledger copy. It requires the source
URL, retrieval time, release preamble, table caption, header and target row,
with a checksum of the selected browser DOM. It does not fetch evidence, prove
the server response bytes, establish historical FRED availability, or upgrade
a replay record to verified. The accompanying tests use synthetic excerpts.

When the original excerpts have actually been retained, call
`compare_captures_to_ledger(ledger, captures, expected_keys)` to parse them
again and attach each comparison to the matching ledger record. Each capture
item has `decision_year`, `role` (`current` or `previous`), and the saved DOM
fields (or a nested `capture` dictionary). Specify the exact `(decision_year, role)` pairs being
checked; missing, unexpected and duplicate pairs fail closed. The adapter
uses the project's May 31 cutoff, checks value and observation date, and
records the source and excerpt digest. Save the original excerpts and the
annotated ledger together in storage. A successful comparison is
evidence of consistency with those later-retained excerpts, not proof of the
FRED vintage or historical availability; provenance stays unverified.

`sec_capture.py` validates selected browser-DOM excerpts from CF quarterly
financial statements and compares current-quarter net sales with a copied
ledger. It checks the physical positions of quarterly, half-year and
year columns in the retained rows, conditional on retained inline styles;
external CSS, hidden ancestors and the selection context are not established by
these excerpts. It does not independently prove server bytes, historical
publication time or archive immutability, and it does not promote replay records.

`sec_index_capture.py` validates retained selected SEC index DOM excerpts,
including accession, report period, filing date, displayed acceptance timestamp
and primary document link. An acceptance timestamp does not by itself establish
when the filing was publicly disseminated. Earlier earnings announcements and
later 10-Q filings are separate events. The capture and checksum describe a
later browser observation, not original server bytes or prior availability.
Tests use synthetic records.

After the source-capture cells have run, `source_links_notebook_runner.py`
reparses the retained excerpts and attaches them to the replay ledger: twelve
BLS price comparisons for 2010–2015, eleven SEC Q2 statement comparisons for
2016–2026, nine matching SEC index excerpts, and 22 Q1 shipment comparisons
against twelve cited earnings disclosures. It requires exact expected
coverage and refuses duplicates, mismatched values or mismatched documents. It
asserts every pre-existing ledger field and replay eligibility decision is
unchanged, then saves the annotated ledger and a compact link-count summary.
The shipment adapter checks each current or previous record against
its own cited source and does not substitute a later comparative column. The
notebook records source selection and column-layout limitations. The
earlier full workflow and report run before these annotations; the unchanged
eligibility assertion ensures the replay conclusions remain the same. The
later source links do not prove historical public availability, original
FRED vintages or an earlier earnings-release date.

The source-link cell can be rerun in the same kernel. It retains a detached
copy of its unannotated input and checks both that input and the last linked
ledger before recomputing. An unexpected change fails closed; rerun the
earlier replay cell to start again in a fresh kernel. This in-memory check is
not a timestamped, independently retained integrity guarantee.

The historical register separates one-week and six-week fieldwork prediction
summaries. Where a linked investigation describes an already examined year, the
register records that as a minimum study-specific seen year, not a complete list
or proof of when an original model was proposed. Other proposal-time knowledge
remains unknown.

For the named candidates in the two 2025 sensitivity studies, the matched
four-model comparison, and the one-/six-week and longer-history fieldwork
studies, the register also records model-specific inputs recovered from saved
analysis code and links the relevant source files. In the one-/six-week
comparison, inspect the result source path or its window field to determine
which fieldwork window was used; the model label alone does not identify it.
The generic fieldwork study also accepts either window, so its model label alone
does not establish which window a particular artifact used. These studies use
matched cohorts requiring complete forecast and fieldwork data even for a model
without one or both as prediction features. Other candidate inputs remain
explicitly unknown at model level; the broader study description is not a
precise input list.

The links supporting those recovered descriptions point to GitHub source
snapshot `434c8808e4fe77cf8c59e9de0c6cca152737cb92`, rather than the moving
`main` branch. This fixes which source text is cited; it does not establish
when an experiment was originally proposed or which information was then known.

## Registering a new experiment prospectively

`experiment_log.py` is separate from the reconstructed historical inventory.
Use one JSONL writer, and log the proposal **before**
viewing the intended test outcome. Make a proposal JSON object such as:

```json
{"model":"example_model","why":"Explain the pre-test rationale","inputs":["price","weather"],"seen_years":[2022,2023,2024,2025,2026],"stage":"prospective","planned_tests":[{"year":2027,"cutoff_utc":"2027-05-31T00:00:00Z"}]}
```

Append it with `python experiment_log.py --log /private/experiments.jsonl --kind proposal --id example_model --payload /private/proposal.json`.
After the result is public, write a separate result JSON object with
`status` (one of `success`, `failure`, `mixed`, `inconclusive`, `error`),
`test_years`, a nonempty `metrics` object, and an `artifact` path.
Append it using the same command with `--kind result`, the same `--id`, and
the result JSON path. Verify later with
`python experiment_log.py --log /private/experiments.jsonl --kind verify`.
Each new prospective proposal must declare its test years and timezone-aware
cutoffs in `planned_tests`; every cutoff must be in that year and no later than
May 31 at 00:00 UTC. The recorded proposal time must be strictly earlier than
every declared cutoff. This conservative deadline precedes May 31 in Iowa;
it does not assert that the frozen protocol has an exact intraday cutoff. A
result's years must be a nonempty subset of the declared years. The prospective
result marked success, failure or mixed cannot be recorded before July 1 UTC
of any test year. This coarse UTC bound does not by itself prove that Q2 has
ended in the company's local time or verify the actual release date. An error
or inconclusive attempt can be recorded earlier. The prospective
result cannot include a year listed as seen in the proposal or one present in
a result recorded before the proposal. A prior error might not mean the outcome
was seen, so this deliberately conservative rule can block a genuinely unseen
test. Multiple proposals registered before any result may still share the same
future test. Recorded timestamps must be nondecreasing across the log, but
this check does not prove their real time or that the log is complete.

Old logs remain verifiable without rewriting their hash chains. Verification
reports `legacy_prospective_without_cutoffs`: those older proposals did not
declare test cutoffs and cannot substantiate a timely prospective claim. New
legacy-style proposals and new results attached to legacy prospective
proposals are rejected. An existing historical record is not retroactively
converted into proof of a prospective forecast; register a new proposal before
the applicable cutoff for a new test year. Recorded cutoffs and timestamps are
still user assertions, not independent proof of registration time or secrecy.
When verifying against a trusted anchor, a legacy prospective proposal or a
result attached to one is also rejected if it appears after that anchor. Legacy
events within the anchored prefix remain readable. The notebook carry-forward
checks a mounted previous log's complete prefix in the same way; without a
trusted earlier prefix, verification cannot tell old legacy events from newly
constructed ones and reports them as uncertified rather than proving timing.
An internally consistent hash chain alone does not detect someone editing old
events and recomputing all later hashes, or removing a suffix. To detect those
changes to a saved prefix, first verify a **nonempty** log and save its count
and head independently, for example in a separately retained notebook
version. The verify output can itself be used as an anchor:

```sh
python experiment_log.py --log /private/experiments.jsonl --kind verify > /private/anchor.json
python experiment_log.py --log /private/experiments.jsonl --kind verify --anchor /private/anchor.json
```

The second command checks that the trusted count and head match the same prefix
of the current log, allowing subsequent valid appends. Keep the trusted anchor
separately from the current log, protect it against replacement, and update it
in a new independently retained version after subsequent appends. A chain and
anchor cannot prove the timestamp, prevent omitted or backdated experiments,
independently prove a result was unseen, or protect an unanchored suffix from
recomputation. Failed and inconclusive outcomes should be recorded.

Kaggle's `/kaggle/working` is a run's temporary output area, not a persistent
append-only filesystem across notebook versions. To continue an existing log,
make the previous version's saved output available to the new run,
copy its JSONL to the current working directory, verify it, and only
then append. Save the updated output with the new notebook version.
If the previous log is unavailable, stop rather than silently starting a new
chain. Avoid parallel writers and verify the carried-forward head and contents.

`notebook_log_runner.prepare_prospective_log(destination, previous_log=None,
anchor=None)` assists with this carry-forward without recreating historic
events. Pass the actual path of the mounted prior notebook output as
`previous_log`. It verifies that file, copies its exact bytes with exclusive
creation, and optionally checks a separately trusted count/head. If a working
destination already exists, it verifies that the mounted previous log is a
prefix and does not overwrite the destination. If a specified mount is missing,
corrupt, conflicts with the working file, or fails the supplied anchor check,
it raises an error. If no destination or previous log is supplied, it reports
`no_prospective_log_carried` and creates nothing; this permits a clean
retrospective analysis but **does not establish that there was no earlier log**.
The historical artifact register is independent and may still be rebuilt.
Before recording any new experiment, locate and mount the previous log, then
carry it forward. Only intentionally create a new log after establishing that
there is no previous chain to preserve. Never run an old notebook cell that
automatically recreates the four retrospective entries alongside this helper.

The existing retrospective log is also preserved in the Kaggle dataset
`iowa-research-private-archive-version-23`, dataset version 1. The research
notebook can pin that dataset version and carry forward its log. This is a
baseline, not an automatically updated archive: whenever an experiment is
appended, save the updated log in a new dataset version and pin that
new version before the next run. Otherwise a fresh run would reuse the older
baseline, and the helper cannot discover newer logs stored elsewhere. Compare
the verified head/count with a separately retained trusted anchor when doing
the update. Never assume a successful chain check proves all attempts were
recorded or that a proposal was made before an outcome was seen.


The notebook additionally uses `daily_ledger.attach_daily_forecasts` to link
retained selected daily rows to a copy of the replay ledger. It checks
canonical source-row hashes, run and valid times, duplicate/conflicting days,
station counts and summary values, and records missing or unusable station-years
in `iowa_replay/forecast_daily_links_private.json`. These are later-retrieved
selected rows; their availability and source version remain unknown. Linking
them does not prove that the archive is complete or promote any evidence.
The project register also retains row-level results from the narrowly scoped
`iowa_fieldwork_2022/annual_private.csv`; files with the same name in other
studies are not automatically treated as results.

The source-link cell can also check a selected, later-retrieved ALFRED
chart DOM excerpt for the May 31, 2025 vintage against both April price records.
It verifies the visible series title, exact vintage header, row dates, values,
and source checksum before adding annotations, and refuses changed captures on
a rerun. This is an assessment of selected browser-rendered content; it does
not establish original server bytes, the historical time of availability, or
whether the archive was later changed. It does not upgrade replay eligibility.

For retained prediction CSVs, the register also indexes each parsed source row
alongside the existing model summaries. It preserves duplicate rows and links
each entry to its source checksum; the ordinal is the parsed-record position,
not a physical file line. A wide row stays a single source record. These rows
are evidence of retained outputs, not a count of unique experiments, and they
cannot reconstruct missing studies or proposal-time motives.

The matched four-model comparison records each candidate's specific inputs
from the fixed `research_audit/comparison.py` source snapshot. Its annual
incremental-error table is also indexed, scoped to the known comparison output
folder; a row contains multiple comparisons and is not itself a model trial.

The source-link runner can check retained, later-retrieved ALFRED
chart excerpts against current and previous April price records for a declared
set of decision years. It reports exact matches, mismatches and missing records
separately and attaches only exact matches without promoting replay eligibility.
It checks the series, vintage column and observation labels. A selected DOM
excerpt and its checksum do not prove original server bytes, historical
availability or an unchanged archive. Captures and diagnostics remain in
Kaggle output.


Strict weather replay also requires an explicit GFS MOS P12 definition at
each daily, station and annual level: percent probability of at least 0.01
inch liquid-equivalent precipitation during lead hours 12–24. The daily
adapter records this definition without certifying historical availability.
Original observed-weather and earlier shipment summaries in the report now
require matching source-specific scores and annual evidence; duplicates or
missing year evidence are disclosed rather than silently combined.
