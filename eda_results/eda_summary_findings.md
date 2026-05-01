# EDA Summary Report: Revenue Forecasting Strategy

## Overview
This report summarizes the enhanced exploratory data analysis (EDA) performed to validate forecasting assumptions for the 2026 Datathon. The objective was to refine the forecasting pipeline by evaluating historical trends, anomalies, and operational data, while strictly adhering to the constraint that no external data (e.g., inventory, web traffic) is available for the 2023-2024 test period.

## Performed Analysis
1.  **Regime Shift Analysis:** Evaluated the impact of COVID-19 using 90-day rolling revenue means to determine if pre-2020 data remains relevant for long-term forecasting.
2.  **Promotion Reality Check:** Analyzed the distribution of promotion durations to determine if hardcoded schedules in existing notebooks are sufficient or if dynamic handling is required.
3.  **Tết Demand Mapping:** Overlaid revenue patterns for the 30 days before and 15 days after Tết (2013-2022) to identify the optimal mathematical representation for this critical anomaly.
4.  **Operational Data Correlation:** Examined web traffic sessions against historical revenue to identify leading indicators and potential validation metrics.

## Key Findings
*   **Regime Shift:** Revenue exhibits significant non-stationarity post-2020. **Recommendation:** Utilize era-weighted training (prioritizing post-2020 data) rather than discarding pre-2020 data, preserving valuable seasonality and long-term trend information.
*   **Tết Pattern:** Data consistently shows a sharp, volatile ramp-up in the final week leading to Tết, followed by an immediate drop. **Recommendation:** Use a categorical flag approach rather than a smooth Gaussian curve, as it better captures the abrupt demand changes.
*   **Operational Trends:** Web traffic is highly correlated with revenue. **Recommendation:** While traffic data cannot be used in the test set, use historical conversion ratios to sanity-check model outputs.
*   **Promotions:** Promotional events show varied durations. **Recommendation:** Dynamic feature engineering should be prioritized over hardcoded seasonal arrays to ensure robustness against shifting business strategies.

## Strategic Path Forward
The forecasting strategy will consolidate these insights into a unified pipeline:
*   **Model Input:** Historical Revenue/COGS and calendar features (including dynamic Tết flags).
*   **Training Strategy:** Era-weighted training to respect the COVID-19 regime shift while utilizing the full historical dataset.
*   **Model Selection:** Leverage the N-HiTS architecture for its strong performance on daily horizons and Prophet for its native handling of holiday calendar complexity.
