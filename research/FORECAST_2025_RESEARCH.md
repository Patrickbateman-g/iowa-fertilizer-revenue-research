# Investigating the 2025 forecast error

The Kaggle notebook compares the original price-only and forecast-adjusted predictions, identifies how far the forecast feature fell outside the previous training range, and tests two post-hoc sensitivities. Run the existing forecast research cells followed by analysis_forecast_2025_diagnostic.py. The small input data and generated results remain on Kaggle.

CF's [Q2 2025 filing](https://www.sec.gov/Archives/edgar/data/1324404/000132440425000024/cf-20250630.htm) attributes its increase in company-wide net sales primarily to higher selling prices, with a smaller contribution from sales volume. It also attributes higher ammonia sales volume partly to favorable spring application weather and higher starting inventory. These are post-quarter explanations, not information available for the May 31 prediction.

[USDA's Iowa report for the week ending May 11](https://www.nass.usda.gov/Statistics_by_State/Iowa/Publications/Crop_Progress_%26_Condition/2025/IA-Crop-Progress-05-12-25.pdf) described favorable fieldwork and planting conditions. Dry weather can help application; these observations do not establish what caused the forecast error or CF's company-wide revenue. CF's [May 7 Q1 release](https://www.sec.gov/Archives/edgar/data/1324404/000132440425000013/cf-05072025_ex991xearnings.htm) also described higher ammonia production and an expectation of strong North American spring demand.

The half-strength and training-range cap experiments were chosen after historical results were inspected. They are sensitivities, not validation or replacements for the frozen 2027 protocol. Forecast probability is not actual rainfall or farmer purchasing; archive release timing and potential revisions remain unverified.
