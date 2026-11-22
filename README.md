# Activity 1: Multiple Linear Regression – Hourly Traffic Volume Prediction

**Course:** Regression and Classification Algorithms | **CIE 2 – Activity 1** | Unit I – Regression Foundations & Linear Regression
**Student:** Kunal Arun Patil | **Roll No.:** AD3264
**Submission GitHub ID:** [https://github.com/complexiota](https://github.com/complexiota)

## Activity description
Predict a continuous target from several input features at once: work out the least-squares equation by hand on a small sample, verify it with a library, compare a single-feature model with a multiple-feature model, interpret every coefficient, check multicollinearity and residuals, evaluate on unseen data and make one new prediction.

**Problem:** a city transport analyst wants to estimate hourly traffic on a road segment and understand which weather/time factors drive it.

## Dataset
- **Name:** Metro Interstate Traffic Volume
- **Source:** UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- **File:** `Metro_Interstate_Traffic_Volume.csv.gz` (CSV, gzip) – 48,204 rows × 9 columns (40,562 rows after cleaning); one row = one hour on Interstate 94 (Oct 2012 – Sep 2018). Public, non-personal data.
- **Target (y):** `traffic_volume` (vehicles per hour)
- **Features (X):** `temp_C` (°C), `rain_1h` (mm), `clouds_all` (%), `hour` (0–23), `is_weekend` (0/1), `is_holiday` (0/1)

## Work completed
1. Problem framing, data dictionary, units and ranges, data-quality checks
2. Cleaning with a decision log: repeated timestamps (7,629), impossible values (0 K temperature, 9,831 mm rain, zero traffic), holiday flag fixed, near-constant `snow_1h` and redundant weather text columns dropped
3. Scatter plots (with observations), correlation heatmap, feature–target and feature–feature correlations, causation caution
4. **Manual calculation worksheet** (6 observations, features hour and temp): design matrix X, XᵀX, Xᵀy, determinant, inverse, solved b0, b1, b2, fitted equation, one prediction and residual
5. Chronological 80:20 split (32,449 train / 8,113 test)
6. Mean baseline, single-feature model, multiple linear regression; hand calculation **verified** with scikit-learn (identical)
7. Fitted equation, interpretation of each coefficient and the intercept, standardised coefficients
8. Test predictions (actual vs predicted), MAE, MSE, RMSE, R², adjusted R², comparison table
9. Residual plot, residual histogram, predicted-vs-actual plot, VIF, Durbin-Watson and an assumptions table
10. **One diagnosed improvement:** hour-of-day dummies (linear regression is not linear in `hour`)
11. One new prediction (inside the training range), conclusion, limitations, ethics note, decision log

## Key results (test set, later period 2017-10 to 2018-09)

| Model | MAE | RMSE | R² | Adj R² |
|---|---|---|---|---|
| Mean baseline | 1724 | 1970 | 0.000 | 0.000 |
| Single feature (`hour`) | 1637 | 1838 | 0.130 | 0.130 |
| Multiple linear regression (6 features) | 1551 | 1770 | 0.193 | 0.193 |
| Multiple + hour-of-day dummies | 588 | 793 | 0.838 | 0.837 |

**Fitted equation (6 features, training data):**
`traffic_volume = 2207.95 + 17.02·temp_C − 58.34·rain_1h + 3.11·clouds_all + 96.21·hour − 974.23·is_weekend − 1008.97·is_holiday`

- All VIF ≈ 1 → no multicollinearity.
- The extra features help, but a straight line in `hour` cannot model the two rush-hour peaks. Giving each hour its own column raises R² from 0.19 to 0.84.
- Residuals of hourly data are strongly autocorrelated (Durbin-Watson ≈ 0.24 → 0.35), so independence is violated.

## Limitations
- Hourly time-series data violates independence of errors; p-values and confidence intervals should not be trusted.
- Accidents, roadworks, events, seasons and long-term trends are not in the data; some night-time predictions are negative.
- One road segment in one city; do not use for other roads or for real-time or safety-critical decisions.
- Coefficients show association, not causation.

## Repository files
```
.
├── traffic_volume_multiple_linear_regression.ipynb   # full executed notebook with outputs
├── Metro_Interstate_Traffic_Volume.csv.gz            # dataset
├── requirements.txt                                   
└── README.md
```

## How to run
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook traffic_volume_multiple_linear_regression.ipynb
```
Run all cells top to bottom (seed fixed at 42). Keep the `.csv.gz` file in the same folder as the notebook.
