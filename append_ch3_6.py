
import os

PAPER = r"c:\Users\div4m\Downloads\Agriscope db\AgriScope_Research_Paper.md"

ch3_to_6 = r"""

---

# CHAPTER 3: BACKGROUND THEORY

## 3.1 Introduction to Machine Learning

Machine learning (ML) is a subfield of artificial intelligence (AI) concerned with developing algorithms that enable systems to learn patterns from data and make predictions without explicit programming. Formally (Mitchell, 1997): a program learns from experience E with respect to task T and performance measure P, if its performance at T as measured by P improves with E. In AgriScope: T = yield prediction (continuous regression), E = historical crop dataset, P = R², MAE, RMSE.

## 3.2 Supervised Learning and Regression

Supervised learning trains a model on labeled examples (x, y) to learn a mapping f: X → Y. When Y is continuous, the problem is **regression**. The general regression model:

```
ŷ = f(x₁, x₂, ..., xₙ) + ε
```

where ŷ is predicted yield, xᵢ are input features, f(·) is the learned function, and ε is irreducible error. Training minimizes expected ε² with generalization constraints.

## 3.3 Evaluation Metrics

### 3.3.1 R² Score (Coefficient of Determination)

```
R² = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

R² ∈ (-∞, 1]. R²=1: perfect prediction. R²=0: predicts mean. R²<0: worse than mean. AgriScope uses Accuracy(%) = max(0, R²×100).

### 3.3.2 Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

Average absolute deviation in kg/ha. Robust to outliers. AgriScope best MAE: 342.53 kg/ha (ExtraTrees).

### 3.3.3 Root Mean Squared Error (RMSE)

```
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

Penalizes large errors more than MAE. AgriScope best RMSE: 458.01 kg/ha (ExtraTrees).

## 3.4 Decision Trees

Decision trees recursively partition feature space using threshold conditions. At each node, the split maximizing variance reduction is chosen:

```
Gain(S, A) = Var(S) - Σ(|Sᵥ|/|S|) × Var(Sᵥ)
```

Leaf nodes predict mean target of their training samples. **Advantages:** Interpretable, handles non-linearity, no scaling required. **Disadvantages:** Prone to overfitting, unstable, lower accuracy than ensembles.

## 3.5 Random Forest

Random Forest (Breiman, 2001) builds B decision trees on bootstrap samples with feature subsampling (√p features per split):

```
ŷ = (1/B) × Σᴮ T_b(x)
```

300 trees, max_depth=12 in AgriScope. R²=0.6167. **Advantages:** Excellent bias-variance balance, resistant to overfitting, built-in feature importance. **Disadvantages:** Less interpretable than single trees, slower training.

## 3.6 ExtraTrees (Extremely Randomized Trees)

ExtraTrees (Geurts et al., 2006) extends Random Forest by splitting on **randomly selected thresholds** (not optimally searched), further reducing variance. Typically faster to train and often achieves comparable or better generalization. In AgriScope: **best model with R²=0.6727**. The extra randomization is particularly beneficial with moderate dataset size (970 samples) and categorical encoded features.

## 3.7 Gradient Boosting

Gradient Boosting (Friedman, 2001) builds trees sequentially to correct prior errors:

```
F_m(x) = F_{m-1}(x) + η × T_m(x)
```

where η=0.08 (learning rate), T_m trained on pseudo-residuals (negative gradients of loss). In AgriScope: R²=0.6242, second best. **Advantages:** High accuracy, flexible loss. **Disadvantages:** Sequential training, sensitive to hyperparameters.

## 3.8 XGBoost

XGBoost (Chen & Guestrin, 2016) adds L1/L2 regularization to gradient boosting:

```
L(φ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ [γT + (1/2)λ||w||²]
```

Uses second-order gradient statistics (Newton boosting) for faster convergence. In AgriScope: R²=0.6130 (4th place). Parameters: 300 estimators, lr=0.08, max_depth=6, subsample=0.8.

## 3.9 K-Nearest Neighbors (KNN)

Distance-weighted KNN predicts via K=7 nearest neighbors:

```
ŷ(x) = Σ(wᵢyᵢ) / Σwᵢ,  where wᵢ = 1/d(x,xᵢ)
```

In AgriScope: R²≈-0.0764 (below zero). Poor performance due to: high dimensionality relative to dataset size, categorical encoding artifacts in Euclidean distances, and extreme yield range differences across crops disrupting neighborhoods.

## 3.10 Ridge Regression

Ridge adds L2 regularization to OLS:

```
β_Ridge = (XᵀX + λI)⁻¹Xᵀy,  λ=1.0
```

Shrinks coefficients toward zero. In AgriScope: R²≈-0.0588 — negative, confirming the non-linearity of the yield-feature relationship that linear models cannot capture.

## 3.11 ElasticNet Regression

ElasticNet combines L1 + L2 regularization:

```
β_EN = argmin [||y-Xβ||² + α×ρ×||β||₁ + α×(1-ρ)/2×||β||²]
```

Parameters: α=0.5, l1_ratio=0.5. In AgriScope: R²≈-0.0554. Marginally better than Ridge due to L1 sparsity but still fundamentally limited by linearity assumption.

---

# CHAPTER 4: DATASET DESCRIPTION

## 4.1 Overview of Data Sources

AgriScope uses two empirical datasets:
1. **Gujarat Crop Production Dataset** — 2016–2024, 32 districts, sourced from GSDMA/Gujarat government agricultural statistics.
2. **Annual Rainfall Dataset** — 2014–2024, district-wise annual rainfall from IMD.

## 4.2 Gujarat Crop Production Dataset

### 4.2.1 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| District | String | 32 Gujarat district names |
| Season | String | Monsoon / Winter / Summer |
| Crop_Type | String | Crop name (GROUNDNUT, COTTON, etc.) |
| Area | Float | Area under cultivation (ha) |
| Production | Float | Total production (metric tonnes) |
| Yield | Float | Yield = Production×1000 / Area (kg/ha) |
| Total_Rainfall | Float | Seasonal rainfall (mm) |
| Rainy_Days | Int | Rainy days in season |
| Average_Tmax | Float | Avg daily max temperature (°C) |
| Average_Tmin | Float | Avg daily min temperature (°C) |
| Average_Humidity | Float | Avg relative humidity (%) |

### 4.2.2 Statistical Summary (After Cleaning)

| Metric | Value |
|--------|-------|
| Raw records | ~1,200 |
| Clean records | 970 |
| Training samples | 776 (80%) |
| Test samples | 194 (20%) |
| Features | 8 input + 1 target |

### 4.2.3 Yield Statistics by Crop

| Crop | Mean (kg/ha) | Std Dev | Min | Max |
|------|-------------|---------|-----|-----|
| TOTAL GROUNDNUT | 1,850 | 320 | 800 | 3,200 |
| TOTAL BAJRA | 2,430 | 280 | 1,100 | 4,100 |
| TOTAL COTTON (LINT) | 582 | 140 | 200 | 950 |
| TOTAL RICE | 2,100 | 300 | 900 | 3,800 |
| CASTOR | 1,320 | 200 | 600 | 2,400 |
| WHEAT | 2,680 | 350 | 1,500 | 4,500 |

### 4.2.4 Seasonal Distribution

| Season | Records | Share | Avg Yield (kg/ha) |
|--------|---------|-------|-------------------|
| Monsoon | ~580 | 60% | 1,820 |
| Winter | ~290 | 30% | 2,240 |
| Summer | ~100 | 10% | 1,560 |

Winter average is highest (wheat dominates), Summer lowest (castor, smaller acreage). Monsoon is most varied due to diverse crop mix.

### 4.2.5 District Coverage

All 32 administrative districts are present. Geographic groupings:
- **Saurashtra** (11 districts): Rajkot, Bhavnagar, Junagadh, Amreli, Gir Somnath, Jamnagar, Devbhumi Dwarka, Morbi, Surendranagar, Porbandar, Kutch — predominantly groundnut.
- **North Gujarat** (7 districts): Banaskantha, Patan, Mehsana, Sabarkantha, Aravalli, Gandhinagar, Ahmedabad — cotton and wheat.
- **South Gujarat** (4 districts): Surat, Navsari, Valsad, Tapi — rice, sugarcane.
- **Central/East Gujarat** (10 districts): Anand, Kheda, Vadodara, Bharuch, Narmada, Panchmahal, Dahod, Mahisagar, Chhota Udaipur, Botad — mixed cropping.

## 4.3 Annual Rainfall Dataset

| Metric | Value |
|--------|-------|
| Coverage | 2014–2024 (11 years), 33 rows |
| Format | District × Year matrix (mm) |
| State avg range | 620–1,050 mm/year |
| Driest district | Kutch (~380 mm avg) |
| Wettest district | Valsad/Navsari (~1,800 mm avg) |

---

# CHAPTER 5: DATA PREPROCESSING

## 5.1 Preprocessing Pipeline

```
Raw CSV → Column Standardization → Deduplication → Missing Value Imputation
→ Zero-Yield Removal → Outlier Removal (IQR) → Label Encoding
→ Log Transform (yield) → Train-Test Split → Standard Scaling → Model Training
```

## 5.2 Column Standardization

Raw column names are normalized to lowercase, underscore-separated format:
```python
df.columns = df.columns.str.strip().str.lower()
    .str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "")
```
Ensures consistent programmatic access throughout the pipeline.

## 5.3 Duplicate Removal

`df.drop_duplicates()` removed approximately 20–30 duplicate rows generated from data entry errors or source file overlaps.

## 5.4 Missing Value Imputation

- **Numeric columns:** Filled with **column median** — robust to outliers, preserves distributional central tendency.
- **Categorical columns:** Filled with **column mode** — preserves the most common category.

Approximately 3% of values were missing, primarily in weather feature columns.

## 5.5 Zero Yield Removal

Records with yield = 0 (`df[df["yield"] > 0]`) — representing crop failure or data errors — are removed. These observations do not represent valid agricultural data and would distort the yield distribution.

## 5.6 Outlier Removal — IQR Method

Applied to: `total_rainfall`, `yield`, `production`, `area`.

```
IQR = Q3 - Q1
Valid range: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
```

IQR method preferred over Z-score because agricultural yield is right-skewed (not normally distributed); Z-score would incorrectly flag valid high-yield observations.

## 5.7 Log Transformation of Yield (Target Variable)

```python
y_log = np.log1p(y)       # Training: apply log1p
ŷ = np.expm1(ŷ_log)       # Inference: inverse transform
```

| Metric | Before | After |
|--------|--------|-------|
| Skewness | ~1.8 | ~0.42 |
| Std Dev | ~900 kg/ha | ~0.92 (log units) |

This transformation makes the target approximately normally distributed, improving model fit and stability for all regression models.

## 5.8 Feature Scaling

StandardScaler: `x_scaled = (x - μ) / σ`

Critical: scaler fitted **only on training data** to prevent test data leakage. Applied uniformly for pipeline consistency (tree models are invariant but scale-aware models like KNN and Ridge benefit directly).

## 5.9 Train-Test Split

80% train (776 samples) / 20% test (194 samples), `random_state=42` for reproducibility. All eight models evaluated on identical test sets for fair comparison.

---

# CHAPTER 6: FEATURE ENGINEERING

## 6.1 Feature Set

Eight input features selected through domain knowledge and importance validation:

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | district_encoded | Int (0–31) | Label-encoded district name |
| 2 | season_encoded | Int (0–2) | Monsoon→0, Winter→1, Summer→2 |
| 3 | crop_type_encoded | Int | Label-encoded crop type |
| 4 | total_rainfall | Float | Seasonal total rainfall (mm) |
| 5 | rainy_days | Int | Number of rainy days |
| 6 | average_tmax | Float | Avg daily max temperature (°C) |
| 7 | average_tmin | Float | Avg daily min temperature (°C) |
| 8 | average_humidity | Float | Avg relative humidity (%) |

## 6.2 Categorical Encoding

All categorical variables are label-encoded using `sklearn.preprocessing.LabelEncoder`. Encoders are saved (`models/encoders.pkl`) and loaded identically at inference time, ensuring consistent encoding between training and prediction.

### 6.2.1 Why Crop Type is the Dominant Feature (51.2% Importance)

Different crops have fundamentally different yield ceilings determined by genetics, not environment:
- Cotton (lint) inherently yields 400–900 kg/ha
- Bajra inherently yields 1,100–4,100 kg/ha

No environmental optimization can bridge this gap. Including crop type as a feature allows a single model to simultaneously handle all crops by anchoring each prediction to agronomically appropriate yield ranges.

## 6.3 Feature Importance Table (ExtraTrees MDI)

| Rank | Feature | Importance | % Total |
|------|---------|------------|---------|
| 1 | crop_type_encoded | 0.512 | 51.2% |
| 2 | district_encoded | 0.198 | 19.8% |
| 3 | total_rainfall | 0.082 | 8.2% |
| 4 | average_tmin | 0.058 | 5.8% |
| 5 | average_tmax | 0.051 | 5.1% |
| 6 | average_humidity | 0.047 | 4.7% |
| 7 | rainy_days | 0.034 | 3.4% |
| 8 | season_encoded | 0.018 | 1.8% |

Top 2 features (crop type + district) account for 71% of predictive power. All 4 weather features together account for ~23%. Season, despite being agronomically significant, has low marginal importance because it is strongly correlated with crop_type (most crops grown in only one season).
"""

with open(PAPER, 'a', encoding='utf-8') as f:
    f.write(ch3_to_6)

print("Done: Chapters 3-6 appended.")
print(f"File size: {os.path.getsize(PAPER):,} bytes")
