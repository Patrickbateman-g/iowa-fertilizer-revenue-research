# Iowa fieldwork conditions: exploratory research

The Kaggle follow-up notebook compares price-only and forecast models with Iowa fieldwork measures. The original one-week investigation is in analysis_fieldwork_research.py. The follow-up in analysis_fieldwork_six_weeks.py compares one versus six consecutive weeks on identical eligible years and training folds. Compact inputs, coverage, scores and year-by-year results remain in the notebook. Do not promote a candidate or change the frozen 2027 protocol based on these retrospective comparisons.

The one-week feature is Iowa statewide days suitable for fieldwork in the latest Sunday-ending week on or before May 20. The six-week feature averages that week and the five preceding weeks. These measure fieldwork opportunity, not fertilizer application or farmer orders. USDA describes suitable days as those on which weather and field conditions allow work for a major portion of the day: https://www.nass.usda.gov/Publications/National_Crop_Progress/terms_definitions.php

All used documents are dated by the May 31 cutoff. Some values are taken from a later report's previous-week column and may incorporate revisions; the snapshot records the later document date. A few reports are hosted by a university or reproduced by contemporary news outlets; the recorded date is the date of the document used, not a verified original release date. Source values were transcribed and checked, but the offline notebook cannot independently reconstruct them or verify historical revisions.

The matched-year rolling comparisons require at least six earlier complete years with previously filed targets, use training-only scaling, and fit fixed-penalty ridge adjustments against the price baseline. All historical outcomes were already examined and the test sample is tiny, so this is exploratory research, not a new untouched test. Two features together are particularly vulnerable to overfitting.

Keep compact inputs and outputs on Kaggle; this repository contains source and documentation only.
