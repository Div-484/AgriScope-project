"""
AgriScope - ML Prediction Module
Loads the trained model and runs predictions given user inputs.
"""

import os
import numpy as np
import joblib


MODEL_PATH       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model.pkl")
SCALER_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scaler.pkl")
ENCODER_PATH     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "encoders.pkl")
TRANSFORM_PATH   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "transform_info.pkl")

# Season mapping
SEASON_MAP = {"Monsoon": 0, "Winter": 1, "Summer": 2}

# District list (the ones used during training)
GUJARAT_DISTRICTS = [
    "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha",
    "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod",
    "Devbhumi Dwarka", "Gandhinagar", "Gir Somnath", "Jamnagar",
    "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana", "Morbi",
    "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar",
    "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi",
    "Vadodara", "Valsad",
]

# Crop heuristic map for encoding
CROP_HEURISTIC_MAP = {
    "Monsoon": {
        "saurashtra": "TOTAL GROUNDNUT",
        "north":      "TOTAL COTTON (LINT)",
        "south":      "TOTAL RICE",
        "central":    "TOTAL BAJRA",
        "east":       "TOTAL BAJRA",
    },
    "Winter": {"north": "WHEAT", "default": "TOTAL RICE"},
    "Summer": {"default": "CASTOR"},
}

SAURASHTRA = ["Rajkot","Bhavnagar","Junagadh","Amreli","Gir Somnath",
              "Porbandar","Jamnagar","Devbhumi Dwarka","Morbi","Surendranagar","Kutch"]
NORTH      = ["Banaskantha","Patan","Mehsana","Sabarkantha","Aravalli","Gandhinagar","Ahmedabad"]
SOUTH      = ["Surat","Navsari","Valsad","Tapi"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)


def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def load_encoders():
    if os.path.exists(ENCODER_PATH):
        return joblib.load(ENCODER_PATH)
    return None


def load_transform_info():
    if os.path.exists(TRANSFORM_PATH):
        return joblib.load(TRANSFORM_PATH)
    return {"log_transform": False, "le_crop": None}


def encode_district(district: str, encoders: dict = None) -> int:
    if encoders and "district" in encoders:
        le = encoders["district"]
        try:
            return int(le.transform([district])[0])
        except ValueError:
            pass
    districts_sorted = sorted(GUJARAT_DISTRICTS)
    if district in districts_sorted:
        return districts_sorted.index(district)
    return 0


def encode_season(season: str, encoders: dict = None) -> int:
    if encoders and "season" in encoders:
        le = encoders["season"]
        try:
            return int(le.transform([season])[0])
        except ValueError:
            pass
    return SEASON_MAP.get(season, 0)


def infer_crop(district: str, season: str, rainfall: float) -> str:
    """Heuristic crop suggestion for display."""
    if season == "Monsoon":
        if district in SAURASHTRA:
            return "TOTAL GROUNDNUT"
        elif district in NORTH:
            return "TOTAL COTTON (LINT)" if rainfall < 700 else "CASTOR"
        elif district in SOUTH:
            return "TOTAL RICE"
        else:
            return "TOTAL BAJRA"
    elif season == "Winter":
        return "WHEAT" if district in NORTH else "TOTAL RICE"
    else:
        return "CASTOR"


def encode_crop_type(crop_type: str, transform_info: dict, encoders: dict = None) -> int:
    """Encode crop_type for model feature."""
    # Try from transform_info le_crop
    le_crop = transform_info.get("le_crop")
    if le_crop is not None:
        try:
            return int(le_crop.transform([crop_type])[0])
        except (ValueError, AttributeError):
            pass
    # Try from encoders
    if encoders and "crop_type" in encoders:
        le = encoders["crop_type"]
        try:
            return int(le.transform([crop_type])[0])
        except ValueError:
            pass
    return 0


def prepare_features(
    district: str,
    season: str,
    total_rainfall: float,
    rainy_days: float,
    avg_tmax: float,
    avg_tmin: float,
    avg_humidity: float,
    encoders: dict = None,
    transform_info: dict = None,
    crop_type: str = None,
    use_crop_feature: bool = False,
) -> np.ndarray:
    """Build the feature vector for the ML model."""
    district_enc = encode_district(district, encoders)
    season_enc   = encode_season(season, encoders)

    if use_crop_feature and crop_type:
        crop_enc = encode_crop_type(crop_type, transform_info or {}, encoders)
        features = np.array([[
            district_enc, season_enc, crop_enc,
            total_rainfall, rainy_days, avg_tmax, avg_tmin, avg_humidity,
        ]])
    else:
        features = np.array([[
            district_enc, season_enc,
            total_rainfall, rainy_days, avg_tmax, avg_tmin, avg_humidity,
        ]])
    return features


def predict(
    district: str,
    season: str,
    total_rainfall: float,
    rainy_days: float,
    avg_tmax: float,
    avg_tmin: float,
    avg_humidity: float,
) -> dict:
    """
    Run yield prediction and crop type prediction.
    Returns dict with 'predicted_yield' and 'predicted_crop'.
    """
    try:
        model          = load_model()
        scaler         = load_scaler()
        encoders       = load_encoders()
        transform_info = load_transform_info()
        log_transform  = transform_info.get("log_transform", False)

        # Determine likely crop (used as both display + feature)
        predicted_crop = infer_crop(district, season, total_rainfall)

        # Check if model expects crop_type_encoded feature (9 features vs 8)
        use_crop_feature = False
        try:
            n_features = model.n_features_in_
            use_crop_feature = (n_features >= 8)   # 8 = with crop_type, 7 = without
        except AttributeError:
            use_crop_feature = False

        features = prepare_features(
            district, season, total_rainfall, rainy_days,
            avg_tmax, avg_tmin, avg_humidity,
            encoders=encoders,
            transform_info=transform_info,
            crop_type=predicted_crop,
            use_crop_feature=use_crop_feature,
        )

        if scaler is not None:
            features = scaler.transform(features)

        raw_pred = float(model.predict(features)[0])

        # Inverse log-transform if used during training
        if log_transform:
            yield_pred = float(np.expm1(raw_pred))
        else:
            yield_pred = raw_pred

        yield_pred = max(0.0, round(yield_pred, 2))

        return {
            "predicted_yield": yield_pred,
            "predicted_crop":  predicted_crop,
        }

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return {"predicted_yield": 0.0, "predicted_crop": "Model not trained yet"}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return {"predicted_yield": 0.0, "predicted_crop": "Prediction error"}


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict(
        district="Rajkot",
        season="Monsoon",
        total_rainfall=700.0,
        rainy_days=80,
        avg_tmax=32.0,
        avg_tmin=23.0,
        avg_humidity=68.0,
    )
    print(result)
