from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
BASE = Path(__file__).resolve().parent

FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "transaction_type",
    "location",
    "transactions_last_24h",
    "avg_transaction_amount",
    "international",
    "distance_from_home_km",
]

NUMERIC = [
    "amount",
    "transaction_hour",
    "transactions_last_24h",
    "avg_transaction_amount",
    "distance_from_home_km",
]

CATEGORICAL = [
    "merchant_category",
    "transaction_type",
    "location",
    "international",
]

DEFAULT_OPTIONS = {
    "merchant_category": [
        "Education", "Electronics", "Entertainment", "Fashion", "Food",
        "Fuel", "Grocery", "Healthcare", "Online Marketplace", "Restaurants", "Travel", "Utilities"
    ],
    "transaction_type": [
        "ATM", "Contactless", "Online", "POS"
    ],
    "location": [
        "Ahmedabad", "Bengaluru", "Coimbatore", "Delhi", "Hyderabad",
        "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune", "Vijayawada", "Visakhapatnam"
    ],
    "international": ["No", "Yes"]
}


def load_model():
    model_path = BASE / "models" / "best_model.pkl"
    meta_path = BASE / "models" / "model_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError("best_model.pkl not found. Please run train_model.py first.")
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, meta


def dataset_options():
    data_dir = BASE / "data"
    path = data_dir / "indian_credit_card_fraud_multiclass_100k.csv"
    
    if not path.exists():
        csv_files = list(data_dir.glob("*.csv"))
        path = csv_files[0] if csv_files else None

    if not path or not path.exists():
        return DEFAULT_OPTIONS

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        options = {}
        for column in CATEGORICAL:
            if column == "international":
                options[column] = ["No", "Yes"]
            elif column in df.columns:
                cleaned = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .unique()
                )
                vals = sorted([x for x in cleaned if x not in ["Nan", "None", "", "N/A"]])
                options[column] = vals or DEFAULT_OPTIONS[column]
            else:
                options[column] = DEFAULT_OPTIONS[column]

        return options
    except Exception:
        return DEFAULT_OPTIONS


def risk(score):
    if score <= 25:
        return "LOW", "GENUINE TRANSACTION", "APPROVE TRANSACTION"
    if score <= 65:
        return "MEDIUM", "SUSPICIOUS TRANSACTION", "CROSS-VERIFY WITH CARD HOLDER"
    return "HIGH", "FRAUDULENT TRANSACTION", "BLOCK CARD / TRANSACTION"


def causes(row, score):
    reasons = []

    try:
        amount = float(row.get("amount", 0))
        average = float(row.get("avg_transaction_amount", 0))
        if average > 0 and amount > 3 * average:
            reasons.append({
                "title": "Unusually high transaction amount",
                "detail": "The transaction amount is more than 3x the customer's average transaction amount.",
                "type": "amount",
            })
    except (TypeError, ValueError):
        pass

    try:
        hour = int(float(row.get("transaction_hour", -1)))
        if 0 <= hour <= 4:
            reasons.append({
                "title": "Late-night transaction",
                "detail": "The transaction occurred during a late-night or early-morning hour (12 AM - 4 AM).",
                "type": "time",
            })
    except (TypeError, ValueError):
        pass

    try:
        if int(float(row.get("transactions_last_24h", 0))) >= 8:
            reasons.append({
                "title": "High transaction frequency",
                "detail": "A high number of transactions were detected in the last 24 hours.",
                "type": "velocity",
            })
    except (TypeError, ValueError):
        pass

    intl_val = str(row.get("international", "")).strip().title()
    if intl_val in ["Yes", "1", "True", "Intl", "International"]:
        reasons.append({
            "title": "Foreign / international transaction",
            "detail": "The transaction is marked as international and requires extra verification.",
            "type": "international",
        })

    try:
        if float(row.get("distance_from_home_km", 0)) >= 60:
            reasons.append({
                "title": "Transaction far from home",
                "detail": "The transaction occurred far from the customer's home location.",
                "type": "distance",
            })
    except (TypeError, ValueError):
        pass

    if score >= 66:
        reasons.append({
            "title": "High-risk ML pattern",
            "detail": "The trained ML model detected a high-risk fraud transaction pattern.",
            "type": "model",
        })

    return reasons[:4] or [{
        "title": "No major risk factors detected",
        "detail": "The transaction did not trigger major risk threshold flags.",
        "type": "safe",
    }]


def prepare_input(df, manual=False):
    df.columns = df.columns.str.strip()

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = df[FEATURES].copy()

    if manual:
        empty = []
        for column in FEATURES:
            if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
                empty.append(column)
        if empty:
            raise ValueError("Please provide all fields: " + ", ".join(empty))

    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ["merchant_category", "transaction_type", "location"]:
        data[column] = data[column].astype(str).str.strip().str.title()

    def normalize_international(val):
        if pd.isna(val):
            return "No"
        val_str = str(val).strip().lower()
        if val_str in ["yes", "y", "true", "1", "international", "intl"]:
            return "Yes"
        if val_str in ["no", "n", "false", "0", "domestic"]:
            return "No"
        return "No"

    data["international"] = data["international"].map(normalize_international)

    # Compute engineered features required by trained model pipeline
    data["amount_ratio"] = data["amount"] / (data["avg_transaction_amount"] + 1.0)
    data["amount_diff"] = data["amount"] - data["avg_transaction_amount"]
    data["is_late_night"] = data["transaction_hour"].between(0, 4).astype(float)
    data["exceeds_std_limit"] = (data["amount"] > 200000).astype(float)
    data["high_value_tx"] = (data["amount"] >= 50000).astype(float)
    data["limit_utilization_ratio"] = data["amount"] / 200000.0

    return data


def predict(df, manual=False):
    model, meta = load_model()
    data = prepare_input(df, manual=manual)

    probabilities = model.predict_proba(data)

    if probabilities.ndim == 2 and probabilities.shape[1] >= 3:
        p_high = probabilities[:, 2]
        p_med = probabilities[:, 1]
        fraud_prob_pct = (p_high + 0.5 * p_med) * 100.0
    else:
        fraud_prob_pct = probabilities[:, 1] * 100.0

    scores = np.rint(fraud_prob_pct).astype(int)

    output = df.copy()
    output["fraud_probability"] = np.round(fraud_prob_pct, 2).astype(float)
    output["fraud_score"] = scores
    output["prediction"] = [risk(int(score))[1] for score in scores]
    output["risk_level"] = [risk(int(score))[0] for score in scores]
    output["possible_cause"] = [
        " | ".join(
            item["title"]
            for item in causes(data.iloc[index].to_dict(), int(score))
        )
        for index, score in enumerate(scores)
    ]
    output["recommended_action"] = [risk(int(score))[2] for score in scores]
    return output, meta


@app.route("/")
def home():
    try:
        _, meta = load_model()
    except Exception:
        meta = {}
    return render_template("index.html", meta=meta)


@app.route("/manual")
def manual():
    try:
        _, meta = load_model()
    except Exception:
        meta = {}
    return render_template("manual.html", meta=meta, options=dataset_options())


@app.post("/manual/predict")
def manual_predict():
    try:
        data = request.get_json(silent=True) or {}
        row = {column: data.get(column) for column in FEATURES}
        df = pd.DataFrame([row])
        result, meta = predict(df, manual=True)
        output_row = result.iloc[0]
        score = int(output_row["fraud_score"])
        prepared = prepare_input(df, manual=True)
        cause_items = causes(prepared.iloc[0].to_dict(), score)

        return jsonify({
            "fraud_probability": round(float(output_row["fraud_probability"]), 2),
            "fraud_score": score,
            "prediction": str(output_row["prediction"]),
            "risk_level": str(output_row["risk_level"]),
            "possible_cause": str(output_row["possible_cause"]),
            "possible_causes": cause_items,
            "recommended_action": str(output_row["recommended_action"]),
            "best_model": str(meta.get("best_model", "Unknown")),
        })
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route("/upload")
def upload():
    try:
        _, meta = load_model()
    except Exception:
        meta = {}
    return render_template("upload.html", meta=meta)


@app.post("/upload/predict")
def upload_predict():
    try:
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a valid CSV file.")

        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        result, meta = predict(df, manual=False)

        total = len(result)
        low = int((result.fraud_score <= 25).sum())
        medium = int(((result.fraud_score >= 26) & (result.fraud_score <= 65)).sum())
        high = int((result.fraud_score >= 66).sum())

        bins = [
            int((result.fraud_score <= 25).sum()),
            int(((result.fraud_score >= 26) & (result.fraud_score <= 45)).sum()),
            int(((result.fraud_score >= 46) & (result.fraud_score <= 65)).sum()),
            int(((result.fraud_score >= 66) & (result.fraud_score <= 85)).sum()),
            int((result.fraud_score >= 86).sum()),
        ]

        merchant_categories = sorted(
            result["merchant_category"].dropna().astype(str).str.strip().str.title().unique().tolist()
        )
        merchant_high_risk = [
            int(
                (
                    (result["merchant_category"].astype(str).str.strip().str.title() == value)
                    & (result["fraud_score"] >= 66)
                ).sum()
            )
            for value in merchant_categories
        ]

        result.to_csv(BASE / "prediction_results.csv", index=False)

        return jsonify(
            summary={
                "total": total,
                "legitimate": low,
                "medium": medium,
                "fraud": high,
                "fraud_percentage": round(high / total * 100, 2) if total else 0,
                "average_score": round(float(result.fraud_score.mean()), 2) if total else 0,
            },
            charts={
                "risk": [low, medium, high],
                "score": bins,
                "merchant_category_labels": merchant_categories,
                "merchant_category_high_risk": merchant_high_risk,
            },
            preview=result.head(100).fillna("").to_dict("records"),
            columns=list(result.columns),
            best_model=str(meta.get("best_model", "Unknown")),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route("/download")
def download():
    path = BASE / "prediction_results.csv"
    if not path.exists():
        return "No prediction results yet.", 404
    return send_file(path, as_attachment=True, download_name="fraud_prediction_results.csv")


if __name__ == "__main__":
    app.run(debug=True)
