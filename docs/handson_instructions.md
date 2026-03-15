# Hands-on Instructions

## Context
This exercise adapts the style of the existing Data Science and Machine Learning slides into a Bank Indonesia–relevant use case. The focus is on macro-financial monitoring and early warning analysis, aligned with the 2026 ToR topics on Data Science, EDA, machine learning, digital economy, and responsible data governance.

## Dataset overview
Unit of analysis: **province-month**

Main variables:
- `cpi_inflation_yoy`: synthetic year-on-year CPI inflation
- `core_inflation_yoy`: synthetic year-on-year core inflation
- `food_price_index`: proxy for food price pressure
- `exchange_rate_change_pct`: monthly exchange-rate movement
- `qris_transactions_growth_pct`: proxy for digital payment activity
- `online_sentiment_score`: proxy for public sentiment
- `unemployment_rate`: proxy for labour market conditions
- `hotel_occupancy_rate`: proxy for tourism and mobility activity
- `rainfall_index`: proxy for weather-related supply shocks
- `commodity_price_index`: proxy for commodity cost pressure
- `high_inflation_pressure`: binary label, equal to 1 when inflation is at least 2.5

## Session 1
### Title
Exploratory Data Analysis for Regional Inflation Monitoring

### Learning objectives
Participants should be able to:
1. read a structured CSV dataset;
2. inspect data quality and summary statistics;
3. create visualizations for time trend and distribution;
4. identify relationships between inflation and its potential drivers;
5. interpret results in an economics and finance context.

### Suggested tasks
1. Load the CSV file.
2. Display the first five rows and inspect variable names.
3. Check shape, data type, and missing values.
4. Produce summary statistics.
5. Compute average inflation by province.
6. Plot monthly average inflation.
7. Plot inflation distribution.
8. Plot food price index versus inflation.
9. Plot QRIS growth versus inflation.
10. Generate the correlation matrix.
11. Discuss which variables look most relevant and why.

### Discussion questions
- Which provinces show relatively higher average inflation?
- Is inflation more closely related to food prices or exchange-rate change?
- Does stronger QRIS growth coincide with inflation pressure or resilience?
- Which variables may represent supply-side shock, and which may represent demand conditions?

## Session 2
### Title
Machine Learning Early Warning for High Inflation Pressure

### Learning objectives
Participants should be able to:
1. convert a policy dataset into a classification problem;
2. split data into training and testing sets;
3. train and compare multiple machine-learning models;
4. interpret Accuracy, Precision, Recall, F1, and ROC-AUC;
5. explain why governance and human oversight remain important.

### Suggested tasks
1. Load the same CSV file.
2. Create the binary target `high_inflation_pressure`.
3. Select features relevant to early warning.
4. Split train and test sets.
5. Train logistic regression, decision tree, random forest, and kNN.
6. Compare the metrics.
7. Plot confusion matrix and ROC curve for the best model.
8. Plot feature importance.
9. Discuss false positives, false negatives, and institutional implications.

### Discussion questions
- Which model performs best on F1 and ROC-AUC?
- Why is accuracy alone not enough?
- Which features contribute most to early-warning prediction?
- Would the model be suitable for direct policy action or only for decision support?
- What governance issues arise from institutional use of such a model?

## Responsible data governance points
- Synthetic data is useful for teaching but not for deployment.
- Correlation does not automatically imply causation.
- Predictive models should support analysts, not replace them.
- False positives and false negatives have different policy costs.
- Ongoing validation is necessary because relationships can drift over time.

## Suggested participant deliverables
- 1 slide or short summary on key EDA findings
- 1 model comparison table
- 1 short paragraph on responsible use of the model in a BI setting
