# NSW Property Price Prediction — Technical Report

**Dataset:** NSW Valuer General / Land Registry Services, 1.88M residential sales, 2010–2026  
**Models:** OLS, Ridge, Lasso, Random Forest, XGBoost  
**Best result:** Random Forest, R² = 0.59, RMSE = $843k on 375,514 test properties

---

## 1. Problem and Approach

The goal was to predict residential property sale prices across NSW using publicly available government transaction data. With 1.88 million sales across 4,021 suburbs spanning 15 years, the dataset is large enough that model choice and feature representation matter more than data volume. The central modelling challenge is location: suburb is by far the strongest price signal, but encoding 4,000+ unique suburbs without creating a sparse matrix or leaking target information into preprocessing requires care.

The analysis follows a standard supervised regression workflow — EDA, feature engineering, five-model comparison, SHAP interpretation — with an emphasis on understanding *why* the models perform as they do, not just what their metrics are.

---

## 2. Data

**Source:** NSW Valuer General Bulk Property Sales Information (data.nsw.gov.au, CC BY 4.0)  
**Raw size:** 2,202,140 rows  
**After cleaning:** 1,877,569 residential sales, 2010–2026

### Cleaning decisions

| Decision | Detail |
|---|---|
| Scope | Residential sales only (`nature_of_property = R`, 85.6% of raw data) |
| Price range | $50k – $30M (removes gifts, data-entry errors, non-arm's-length transfers) |
| Date range | 2010 onwards (pre-2010 records are sparse — <50k/year vs 170k+ post-2010) |
| Area units | Hectare rows converted to m² (multiply by 10,000); unified to single unit |
| Area cap | 500,000 m² maximum (removes corrupt source values up to 2.7 billion m²) |
| Missing area | Imputed with suburb-level median (lot sizes vary significantly by location) |

The price target is strongly right-skewed (raw skewness = 7.32). Log-transformation reduces this to 0.29, near-normal, and all models predict log(price). Predictions are exponentiated back to AUD for metric reporting.

---

## 3. Feature Engineering

Four features were used for all models:

| Feature | Construction | Rationale |
|---|---|---|
| `suburb_encoded` | Target encoding: train-set mean log-price per suburb | Location captures amenity, transport access, school zones; high cardinality (4,021 suburbs) makes one-hot infeasible |
| `log_area` | `log1p(area_m2)` after unit conversion and capping | Area is right-skewed; the price-area relationship is log-linear rather than linear |
| `year` | Extracted from contract date | Captures the 15-year market trend (2010–2026) |
| `quarter` | Extracted from contract date | Low signal but included for completeness |

**Target encoding details:** Suburb means are computed on the training set only. Unseen test suburbs fall back to the global training mean. Encoded values are clipped to ±3 standard deviations of the training encoding distribution to prevent extreme suburb means from producing large z-scores after StandardScaler, which caused numerical instability in linear models on the full raw dataset.

**Pipeline:** All transformations are implemented as a scikit-learn `Pipeline` + `ColumnTransformer`, fit on training data only. The train/test split is 80/20 (1,502,055 train / 375,514 test, `random_state=42`).

---

## 4. Models

Five models were trained and evaluated:

| Model | Configuration |
|---|---|
| OLS | No regularisation; baseline |
| Ridge | L2 regularisation; alpha selected by 5-fold CV (best alpha = 6.55) |
| Lasso | L1 regularisation; alpha selected by 5-fold CV (best alpha = 0.01, all 4 features retained) |
| Random Forest | 200 trees, `min_samples_leaf=5`, `n_jobs=-1` |
| XGBoost | 3,000 estimators, `learning_rate=0.3`, `max_depth=6`, early stopping at round 2,703 |

---

## 5. Results

### Model comparison — test set (375,514 properties)

| Model | RMSE (AUD) | MAE (AUD) | R² |
|---|---|---|---|
| **Random Forest** | **$843,386** | **$290,478** | **0.59** |
| XGBoost | $984,815 | $352,009 | 0.44 |
| OLS | $1,121,884 | $436,424 | 0.27 |
| Ridge | $1,121,884 | $436,424 | 0.27 |
| Lasso | $1,126,515 | $436,615 | 0.26 |

Random Forest achieves the best performance across all metrics. The large gap between tree models (R² 0.44–0.59) and linear models (R² 0.27) is the key result: it confirms that price is a non-linear function of the available features, primarily through interactions between suburb encoding, area, and year.

Regularisation (Ridge/Lasso) provides essentially no improvement over plain OLS. With only 4 features and 1.5M training samples, variance is not the problem — the linear functional form is the ceiling.

### Prediction accuracy at different thresholds (XGBoost)

| Threshold | % of test properties |
|---|---|
| Within 10% of actual | 25.6% |
| Within 20% of actual | 47.6% |

These figures are consistent with entry-level automated valuation model (AVM) accuracy on a feature set of this size. Commercial AVMs with property attributes (bedrooms, bathrooms, condition) typically achieve within-20% rates of 60–70%.

---

## 6. Feature Interpretation

### Linear model coefficients (Ridge, standardised features)

| Feature | Coefficient |
|---|---|
| suburb_encoded | 0.495 |
| year | 0.129 |
| log_area | 0.098 |
| quarter | 0.017 |

All coefficients are positive. No feature reduces predicted price — there are no negative signals in this feature set, only varying magnitudes of positive contribution. Lasso with `alpha=0.001` retains all four features, indicating that none can be safely zeroed out.

### Tree feature importances (RF impurity, % variance explained)

| Feature | Random Forest | XGBoost |
|---|---|---|
| suburb_encoded | 71.9% | 68.3% |
| log_area | 18.1% | 19.4% |
| year | 7.6% | 10.1% |
| quarter | 2.4% | 2.2% |

### SHAP analysis (XGBoost, 2,000 test samples)

SHAP values confirm the importance ranking from coefficients and impurity measures:

- `suburb_encoded` produces the widest SHAP spread — a high-value suburb alone can shift predicted log-price by over ±0.5 units (roughly ±60% in AUD)
- `log_area` shows a consistent positive monotonic relationship: more area, higher positive SHAP value
- `year` contributes a narrow but uniformly positive SHAP distribution, reflecting the secular trend
- `quarter` SHAP values cluster near zero for nearly all properties

---

## 7. Residual Analysis

The XGBoost residual scatter shows a fan shape: predictions for sub-$2M properties sit tightly along the 45° line, but residuals widen substantially above $3M. The error distribution is right-skewed — the model underestimates more often than it overestimates for high-value properties. This is consistent with mean reversion toward the suburb target encoding: when a $6M house is in a $2M-median suburb, the model lacks the feature signal to predict the premium.

This pattern is not fixable without additional property attributes (quality, views, renovation status). It is a data limitation, not a modelling one.

---

## 8. Limitations

**Feature set:** Four features is thin for property valuation. The model has no access to bedrooms, bathrooms, year built, renovation status, building type, or distance to amenity. The R² ceiling with this feature set is probably around 0.60–0.65 for tree models.

**Suburb encoding cold-start:** Target encoding falls back to the global mean for suburbs with fewer than 30–50 transactions. These are often fast-growing outer suburbs where valuation accuracy matters most for first-home buyers.

**Linear market trend:** Representing time as a single `year` feature captures a long-run trend but cannot distinguish the 2017 peak, the 2019–2020 correction, or the rate-driven pullback from 2022 onwards. A time-series approach or lagged median price features would improve this.

**No spatial structure:** Suburbs are encoded as independent units. In reality, adjacent suburbs are correlated — spatial smoothing or lat/lon features would reduce noise for suburbs with few observations.

---

## 9. Further Work

- Add property attributes: bedrooms, bathrooms, year built, parking — expected to raise R² to 0.75+
- Add spatial features: suburb centroid coordinates, distance to CBD, nearest train station, school zone ICSEA decile
- Replace target encoding with LightGBM native categorical handling — eliminates leakage risk and handles unseen suburbs naturally
- Time-series formulation: include lagged 12-month suburb median price as a feature, or use walk-forward validation to properly evaluate on out-of-sample future dates
- Calibration: evaluate prediction interval coverage (does the 80% CI contain the actual price 80% of the time?) for AVM deployment readiness
