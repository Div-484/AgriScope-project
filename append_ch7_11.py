
import os

PAPER = r"c:\Users\div4m\Downloads\Agriscope db\AgriScope_Research_Paper.md"

ch7_to_11 = r"""

---

# CHAPTER 7: MACHINE LEARNING MODELS

## 7.1 Random Forest Regressor

### 7.1.1 Theory
Random Forest builds an ensemble of decision trees on bootstrapped training subsets. At each split, only a random subset of features is considered, decorrelating trees and reducing ensemble variance. The final prediction is the mean of all tree predictions.

### 7.1.2 AgriScope Parameters
- n_estimators = 300 (300 trees in the forest)
- max_depth = 12 (maximum tree depth for controlled complexity)
- min_samples_split = 4 (minimum samples required to split an internal node)
- n_jobs = -1 (parallel training using all CPU cores)
- random_state = 42

### 7.1.3 Results in AgriScope
- R² = 0.6167 | Accuracy = 61.67% | MAE = 369.58 kg/ha | RMSE = 495.60 kg/ha

### 7.1.4 Why Used
Random Forest is the gold standard baseline for tabular ML tasks. It provides excellent generalization with minimal tuning, making it the ideal benchmark against which all other models are compared.

### 7.1.5 Advantages
- Excellent out-of-the-box performance
- Built-in feature importance via Mean Decrease Impurity
- Robust to noise and missing values
- Parallelizable (fast with n_jobs=-1)
- Resistant to overfitting due to bagging

### 7.1.6 Disadvantages
- Moderate training time with 300 trees
- Less accurate than boosting methods on many datasets
- Memory-intensive for large forests
- Less interpretable than single trees

---

## 7.2 ExtraTrees Regressor (Best Model)

### 7.2.1 Theory
ExtraTrees adds a second layer of randomization to Random Forest: split thresholds are selected randomly rather than optimally searched. For each candidate feature at each node, a random threshold is drawn uniformly from the feature's range in the training subset. The split with the best criterion score among all random thresholds is selected. This extreme randomization dramatically reduces variance, at the cost of a slight increase in bias — a tradeoff that benefits performance when the dataset is moderately sized.

### 7.2.2 AgriScope Parameters
- n_estimators = 300
- max_depth = 12
- random_state = 42
- n_jobs = -1

### 7.2.3 Results in AgriScope (BEST MODEL)
- **R² = 0.6727 | Accuracy = 67.27% | MAE = 342.53 kg/ha | RMSE = 458.01 kg/ha**

### 7.2.4 Why It is the Best Model
ExtraTrees outperforms all other models because:
1. The 970-sample dataset benefits from maximum variance reduction, which ExtraTrees achieves through double randomization (bootstrap + random thresholds).
2. Categorical encoded features (district, crop_type) have many valid split thresholds — random threshold selection avoids overfitting to specific threshold values.
3. The non-linear, hierarchical nature of yield determinants (crop type → district → weather) maps perfectly onto the tree ensemble structure.

### 7.2.5 Advantages
- Lowest variance among all tree methods
- Fastest training of all ensemble methods (no optimal split search)
- Excellent generalization on moderate-sized tabular datasets
- Built-in feature importance

### 7.2.6 Disadvantages
- Slightly higher bias than Random Forest
- Less interpretable than individual trees
- Very large model file size (21.8 MB) for 300 deep trees

---

## 7.3 Gradient Boosting Regressor

### 7.3.1 Theory
Gradient Boosting fits trees sequentially, where each tree is trained to predict the negative gradient (pseudo-residuals) of the loss function with respect to the current ensemble prediction. The ensemble grows incrementally, each tree correcting the errors of its predecessors. The loss function for regression is typically Mean Squared Error.

```
r_im = -[∂L(yᵢ, F_{m-1}(xᵢ)) / ∂F_{m-1}(xᵢ)]
```

### 7.3.2 AgriScope Parameters
- n_estimators = 300
- learning_rate = 0.08 (controls contribution of each tree; smaller = more trees needed but less overfitting)
- max_depth = 5
- random_state = 42

### 7.3.3 Results
- R² = 0.6242 | Accuracy = 62.42% | MAE = 368.01 kg/ha | RMSE = 490.73 kg/ha

### 7.3.4 Advantages
- Powerful bias reduction through sequential correction
- Often achieves best results with sufficient data and tuning
- Flexible, supports custom loss functions

### 7.3.5 Disadvantages
- Sequential training (cannot parallelize tree building)
- Sensitive to hyperparameters (learning rate, depth, n_estimators)
- Can overfit without early stopping or regularization

---

## 7.4 XGBoost Regressor

### 7.4.1 Theory
XGBoost extends gradient boosting with L1/L2 regularization on leaf weights and uses second-order Taylor expansion of the loss function for more accurate gradient estimates (Newton boosting). Built-in support for missing values, column sampling, and row subsampling provides implicit regularization.

### 7.4.2 AgriScope Parameters
- n_estimators = 300
- learning_rate = 0.08
- max_depth = 6
- subsample = 0.8 (row sampling)
- colsample_bytree = 0.8 (feature sampling per tree)
- eval_metric = "rmse"
- verbosity = 0

### 7.4.3 Results
- R² = 0.6130 | Accuracy = 61.30% | MAE = 368.12 kg/ha | RMSE = 498.02 kg/ha

### 7.4.4 Why Slightly Below Gradient Boosting
XGBoost's additional regularization, while beneficial on larger datasets, introduces slight underfitting on the 970-sample AgriScope dataset. The added regularization prevents it from fitting some patterns that Gradient Boosting captures.

### 7.4.5 Advantages
- Highly optimized C++ implementation (fast)
- Built-in L1/L2 regularization prevents overfitting
- Handles missing values natively
- Widely validated across Kaggle competitions

### 7.4.6 Disadvantages
- Many hyperparameters to tune
- Slightly outperformed by ExtraTrees on this dataset
- Requires careful early stopping for optimal performance

---

## 7.5 Decision Tree Regressor

### 7.5.1 Theory
A single unpruned regression tree that recursively partitions the feature space to minimize Mean Squared Error at each split. Without ensemble averaging, a single tree is high-variance — small changes in training data produce very different trees.

### 7.5.2 AgriScope Parameters
- max_depth = 12
- min_samples_split = 4
- random_state = 42

### 7.5.3 Results
- R² = 0.3156 | Accuracy = 31.56% | MAE = 473.92 kg/ha | RMSE = 662.23 kg/ha

### 7.5.4 Why Lower Performance
Without ensemble averaging, the single tree overfits training data and generalizes poorly to unseen districts/seasons. Its R² of 0.3156 is dramatically below the ensemble methods, starkly demonstrating the value of ensemble averaging.

### 7.5.5 Advantages
- Fully interpretable: entire decision logic can be visualized
- Fast training and inference
- No feature scaling required

### 7.5.6 Disadvantages
- High variance (overfitting)
- Unstable: sensitive to training data perturbations
- Much lower accuracy than ensemble methods

---

## 7.6 KNN Regressor

### 7.6.1 Theory
KNN makes predictions by finding the K most similar training samples (by Euclidean distance in scaled feature space) and averaging their yield values with distance-based weights.

### 7.6.2 AgriScope Parameters
- n_neighbors = 7
- weights = "distance"

### 7.6.3 Results
- R² = -0.0764 | Accuracy = 0.0% | MAE ≫ ensemble models

### 7.6.4 Why KNN Fails in AgriScope
1. Euclidean distance in label-encoded categorical space is semantically meaningless — district 0 (Ahmedabad) and district 1 (Amreli) are far apart geographically but adjacent in encoded space.
2. Crop type is the dominant predictor; without semantic distance for crop types (categorical), KNN's neighborhoods are miscalibrated.

---

## 7.7 Ridge Regression

### 7.7.1 Theory
Ridge adds L2 penalty to Ordinary Least Squares, shrinking coefficients toward zero. Closed form: β = (XᵀX + λI)⁻¹Xᵀy, λ=1.0.

### 7.7.2 Results
- R² = -0.0588 | Accuracy = 0.0%

### 7.7.3 Why Ridge Fails
Agricultural yield is fundamentally non-linear. The relationship between crop_type_encoded (integer label) and yield is a step function (not linear) — cotton=582, wheat=2680, bajra=2430 have no linear relationship with their integer labels.

---

## 7.8 ElasticNet Regression

### 7.8.1 Theory
ElasticNet combines L1 (Lasso) sparsity and L2 (Ridge) shrinkage: objective = ||y-Xβ||² + α×l1_ratio×||β||₁ + α×(1-l1_ratio)/2×||β||². Parameters: α=0.5, l1_ratio=0.5.

### 7.8.2 Results
- R² = -0.0554 | Accuracy = 0.0%

### 7.8.3 Why ElasticNet Fails
Same fundamental issue as Ridge — linearity assumption is violated. The L1 component sparsifies coefficients but cannot recover non-linear patterns.

---

# CHAPTER 8: EXPERIMENT DESIGN

## 8.1 Experimental Setup

All experiments were conducted in the following standardized environment:
- Python 3.10 on Windows 10/11
- scikit-learn 1.3.x, XGBoost 2.0.x
- Hardware: Intel Core i5/i7, 8–16 GB RAM
- No GPU required (tree models are CPU-parallelized)

## 8.2 Experimental Methodology (Fair Comparison Protocol)

To ensure scientifically valid model comparisons, the following protocol was strictly followed:

1. **Single Train-Test Split:** All models trained on identical training data (`random_state=42`, 80/20 split) and evaluated on identical test sets.
2. **Identical Preprocessing:** The same scaler (fitted on training data only) was applied to all models.
3. **Log Transform:** All models trained on `log1p(yield)`; all metrics computed after inverse-transforming predictions to original kg/ha scale.
4. **Same Feature Set:** All models used the identical 8-feature input vector.
5. **Standardized Metrics:** R², MAE, RMSE computed uniformly for all models.

## 8.3 Hyperparameter Configuration

| Model | Key Parameters | Selection Rationale |
|-------|---------------|--------------------:|
| Random Forest | n=300, depth=12, min_split=4 | Literature defaults; balanced bias-variance |
| ExtraTrees | n=300, depth=12 | Matched RF for direct comparison |
| Gradient Boosting | n=300, lr=0.08, depth=5 | Reduced depth vs RF to prevent overfitting in sequential setting |
| XGBoost | n=300, lr=0.08, depth=6, sub=0.8 | Standard Kaggle starting parameters |
| Decision Tree | depth=12, min_split=4 | Matched RF leaf settings |
| KNN | k=7, weights=distance | Literature recommendation for moderate datasets |
| Ridge | α=1.0 | Standard default |
| ElasticNet | α=0.5, l1=0.5 | Balanced L1/L2 blend |

## 8.4 Model Evaluation Metrics

All models are evaluated using three complementary metrics:
- **R²:** Overall proportion of variance explained (higher is better; scale: 0–1 for useful models)
- **MAE:** Average absolute error in kg/ha (lower is better; interpretable in yield units)
- **RMSE:** Penalizes large errors; more informative for models that occasionally make large mistakes

## 8.5 Error Analysis Methodology

Beyond aggregate metrics, error analysis examines:
1. **Error distribution by crop type:** Do models systematically underpredict/overpredict specific crops?
2. **Error distribution by season:** Are errors higher in monsoon season due to greater rainfall variability?
3. **Error distribution by district:** Do certain districts (e.g., Kutch with unique agro-climatic conditions) show higher errors?
4. **Residual plot behavior:** Are residuals randomly distributed (good) or show systematic patterns (indicates un-modeled structure)?

---

# CHAPTER 9: RESULTS AND ANALYSIS

## 9.1 Comprehensive Model Comparison Table

| Model | R² Score | Accuracy (%) | MAE (kg/ha) | RMSE (kg/ha) | Rank |
|-------|----------|-------------|------------|-------------|------|
| **ExtraTrees** | **0.6727** | **67.27%** | **342.53** | **458.01** | **🏆 1** |
| Gradient Boosting | 0.6242 | 62.42% | 368.01 | 490.73 | 2 |
| Random Forest | 0.6167 | 61.67% | 369.58 | 495.60 | 3 |
| XGBoost | 0.6130 | 61.30% | 368.12 | 498.02 | 4 |
| Decision Tree | 0.3156 | 31.56% | 473.92 | 662.23 | 5 |
| KNN | -0.0764 | 0.0% | — | — | 6 |
| ElasticNet | -0.0554 | 0.0% | — | — | 7 |
| Ridge | -0.0588 | 0.0% | — | — | 8 |

## 9.2 Tier Analysis

### Top Tier: Ensemble Tree Methods (R² = 0.61–0.67)
ExtraTrees, Gradient Boosting, Random Forest, and XGBoost form a compact performance cluster. All achieve R² > 0.61, with ExtraTrees as the clear leader (+4.85 percentage points above Gradient Boosting). The tight clustering suggests that the variance explainable from available features is approximately 67%, with the remaining 33% attributable to unobserved factors (soil quality, seed variety, irrigation, pest pressure, farming practice).

### Middle Tier: Decision Tree (R² = 0.31)
Decision Tree achieves only 31.56% — less than half the ExtraTrees R². This dramatic gap quantifies the value of ensemble averaging: 300 trees averaging their predictions reduces variance to roughly half of what a single tree achieves.

### Bottom Tier: Linear Models and KNN (R² < 0)
KNN (R²=-0.076), ElasticNet (R²=-0.055), and Ridge (R²=-0.059) all produce negative R² scores — they perform worse than simply predicting the dataset mean for every sample. This result is unambiguous: linear models are fundamentally incompatible with agricultural yield prediction from tabular data.

## 9.3 Why ExtraTrees is the Best Model

The ExtraTrees superiority over Random Forest can be explained through the bias-variance tradeoff lens:
- Both models use identical parameters (n=300, depth=12)
- ExtraTrees' random threshold selection further reduces tree-to-tree correlation
- On the 970-sample dataset, the additional variance reduction outweighs the marginal bias increase
- The categorical encoded features (crop_type, district) have many valid split points across their integer range; random threshold selection avoids overfitting to specific boundary values

## 9.4 Why Linear Models Failed

The fundamental reason for linear model failure is misspecification. Agricultural yield cannot be expressed as a linear combination of District + Season + Crop_Type + Rainfall + Temperature + Humidity. Consider:
- Cotton with 800mm rainfall: ~600 kg/ha
- Wheat with 800mm rainfall: ~2,800 kg/ha

The crop_type feature has an integer label (e.g., Cotton=0, Wheat=5), but yield difference ≈ 4.7× cannot be captured by a linear weight on the integer label. Tree models handle this naturally through branching: "if crop_type=0 (Cotton), predict in [400-900] range; if crop_type=5 (Wheat), predict in [1500-4500] range."

## 9.5 Error Interpretation

**ExtraTrees MAE of 342.53 kg/ha** means the average prediction error is 342.53 kg/ha. In context:
- As percentage of mean yield (~1,800 kg/ha): ~19% error
- Agronomically: sufficient for strategic crop planning (district-year-crop selection) but too coarse for individual field-level management
- In economic terms: ~342 kg/ha error on groundnut at ₹50/kg ≈ ₹17,100/ha potential planning uncertainty

**RMSE (458.01 kg/ha) > MAE (342.53 kg/ha):** The RMSE/MAE ratio ≈ 1.34 indicates some large prediction errors exist (RMSE penalizes outliers more). These large errors likely occur for extreme yield observations (exceptional drought or bumper crop years) where limited training examples make generalization harder.

## 9.6 Impact of Dataset Size

At 970 samples across 32 districts × 3 seasons × 6 crops (theoretical 576 district-season-crop combinations), many cells have only 1–3 observations. This data sparsity is the primary constraint on model accuracy. A dataset 5–10× larger would likely push ExtraTrees R² to 0.80+. This limitation is acknowledged and directly informs the future work chapter.

---

# CHAPTER 10: SYSTEM ARCHITECTURE

## 10.1 Architectural Overview

AgriScope follows a **modular, layered architecture** comprising four functional layers:

```
┌──────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                     │
│         Streamlit Dashboard (app/app.py)                  │
│   Dashboard Page │ Crop Prediction │ Analytics │ History  │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    APPLICATION LAYER                      │
│   utils/prediction.py   │   utils/weather_api.py         │
│   utils/data_cleaning.py                                 │
└──────┬───────────────────┬───────────────────────────────┘
       │                   │
┌──────▼──────┐     ┌──────▼───────────────────────────────┐
│  MODEL LAYER │     │            DATA LAYER                │
│  models/     │     │  data/ (CSVs)  │ database/ (SQLite) │
│  *.pkl files │     │  cleaned_data/ │ agriscope.db        │
└─────────────┘     └─────────────────────────────────────┘
```

## 10.2 Data Flow — Prediction Pipeline

```
User selects district + season
        ↓
Weather API (Open-Meteo) called
        ↓
Real-time weather returned: Temp, Humidity, Rainfall, Wind
        ↓
Weather processed → Seasonal rainfall estimated (rainfall × 90 + 200)
Rainy days estimated (max(30, int(total_rainfall / 8)))
        ↓
Crop heuristic determines recommended crop (based on district + season + rainfall)
        ↓
Feature vector assembled: [district_enc, season_enc, crop_enc, total_rainfall,
                           rainy_days, avg_tmax, avg_tmin, avg_humidity]
        ↓
StandardScaler applied
        ↓
ExtraTrees model (model.pkl) predicts log(yield)
        ↓
Inverse log transform → predicted yield in kg/ha
        ↓
Results displayed in dashboard + saved to SQLite DB
```

## 10.3 Module Descriptions

### 10.3.1 `app/app.py` — Streamlit Dashboard
The main entry point for the web application. Contains the complete UI logic for both pages (Dashboard and Project Details) across 5 tabs. Imports from all utility and database modules. Implements the full CSS theme, chart rendering using matplotlib, and data loading with Streamlit caching.

### 10.3.2 `utils/prediction.py` — Prediction Engine
Encapsulates all ML inference logic: loading model/scaler/encoder artifacts, encoding user inputs, assembling feature vectors, applying scaling, running model inference, and inverse-transforming predictions. Also contains the crop recommendation heuristic based on district geography and season.

### 10.3.3 `utils/weather_api.py` — Weather Integration
Queries the Open-Meteo REST API with coordinates for any of 32 Gujarat districts. Returns current temperature, humidity, precipitation, and wind speed. Falls back to historical averages if the API is unavailable (offline resilience).

### 10.3.4 `utils/data_cleaning.py` — Data Pipeline
Implements the complete preprocessing pipeline: column standardization, deduplication, missing value imputation, dtype correction, feature engineering, outlier removal, and label encoding.

### 10.3.5 `database/database.py` — Persistence Layer
SQLite database wrapper providing: connection management, auto-table creation, `save_prediction()` for inserting new predictions, `fetch_predictions()` for history retrieval, and `clear_predictions()` for testing.

### 10.3.6 `train_model.py` — Training Script
Standalone script (run once) that executes the full ML training pipeline: data cleaning, feature engineering, model training (all 8 models), evaluation, ranking, and artifact saving.

### 10.3.7 `models/` — Artifact Store
Saves all trained model `.pkl` files, StandardScaler, LabelEncoders, log-transform info, and `metrics.json` containing performance results and metadata for all models.

## 10.4 External Integrations

### 10.4.1 Open-Meteo API
- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Parameters:** latitude, longitude, current weather variables
- **Response:** JSON with current temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m
- **Timeout:** 10 seconds (fallback activated on timeout)
- **Cost:** Free, no API key required, open-source

### 10.4.2 SQLite Database
- **File:** `database/agriscope.db`
- **Table:** `prediction_logs` with columns: id, district, season, temperature, humidity, rainfall, predicted_crop, predicted_yield, timestamp
- **Access pattern:** Write on each prediction; read on History tab load; supports up to 200 record display

---

# CHAPTER 11: IMPLEMENTATION

## 11.1 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.10 | Core development |
| Web Framework | Streamlit | 1.28+ | Interactive dashboard |
| ML Library | scikit-learn | 1.3.x | All models except XGBoost |
| Boosting | XGBoost | 2.0.x | XGBoost model |
| Data Manipulation | Pandas | 2.0.x | DataFrame operations |
| Numerical Computing | NumPy | 1.24.x | Array operations, log transforms |
| Visualization | Matplotlib | 3.7.x | All charts and figures |
| Model Serialization | joblib | 1.3.x | Saving/loading .pkl files |
| Database | SQLite3 | Built-in | Prediction history storage |
| Weather API | requests | 2.31.x | HTTP calls to Open-Meteo |
| Notebook Training | Jupyter | 7.x | Model training notebook |

## 11.2 Running the Application

### 11.2.1 Setup
```bash
cd "AgriScope"
pip install -r requirements.txt
```

### 11.2.2 Train Models (First Time)
```bash
python train_model.py
```

### 11.2.3 Launch Dashboard
```bash
streamlit run app/app.py
```

Dashboard opens at http://localhost:8501

## 11.3 Key Implementation Details

### 11.3.1 Model Persistence Strategy
All trained artifacts are saved using `joblib.dump()`:
- `model.pkl` — best model (ExtraTrees)
- `scaler.pkl` — fitted StandardScaler
- `encoders.pkl` — LabelEncoders for district, season, crop_type
- `transform_info.pkl` — log-transform flag + crop LabelEncoder
- `metrics.json` — all model performance metrics + feature list
- Individual model files: `extratrees_model.pkl`, `randomforest_model.pkl`, etc.

### 11.3.2 Real-Time Weather to ML Feature Conversion
Since Open-Meteo returns only **current** weather (not seasonal totals), a conversion heuristic is applied:
```python
total_rainfall = weather["rainfall"] * 90 + 200   # current × 90 days + 200mm base
rainy_days = max(30, int(total_rainfall / 8))       # estimate rainy days
avg_tmax = weather["temperature"] + 4.0             # estimate daily max
avg_tmin = weather["temperature"] - 5.0             # estimate daily min
avg_humidity = weather["humidity"]                  # direct from API
```

This heuristic provides reasonable seasonal estimates from point-in-time weather readings.

### 11.3.3 Streamlit Caching
All data loaders use `@st.cache_data` to prevent repeated file I/O on every user interaction:
```python
@st.cache_data
def load_main_data():
    return pd.read_csv(CLEANED_PATH)
```

### 11.3.4 Feature Count Adaptation
The prediction module auto-detects whether the trained model expects 8 or 7 features (with or without crop_type_encoded) by checking `model.n_features_in_`, ensuring backward compatibility.

### 11.3.5 Crop Recommendation Heuristic
Geographic regions are hardcoded based on agricultural domain knowledge:
```python
SAURASHTRA = ["Rajkot","Bhavnagar","Junagadh","Amreli",...]
NORTH = ["Banaskantha","Mehsana","Patan","Sabarkantha",...]
SOUTH = ["Surat","Navsari","Valsad","Tapi"]
```

For Monsoon season + Saurashtra district → "TOTAL GROUNDNUT"
For Monsoon season + North Gujarat + low rainfall → "TOTAL COTTON (LINT)"
For Winter season + North Gujarat → "WHEAT"
"""

with open(PAPER, 'a', encoding='utf-8') as f:
    f.write(ch7_to_11)

print(f"Done: Chapters 7-11 appended. File size: {os.path.getsize(PAPER):,} bytes")
