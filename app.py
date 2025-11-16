# app.py
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
import json
import os

MODEL_PATH = "model.joblib"
FEATURES_PATH = "features.json"

app = Flask(__name__)

# Load model and features metadata at startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("model.joblib or features.json not found. Run save_model.py first to create these files.")

clf = load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    features = json.load(f)  # list of {"name":..., "dtype":...}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    # Build a single-row DataFrame in the same column order as training
    data = {}
    for feat in features:
        name = feat["name"]
        dtype = feat.get("dtype","object")
        raw = request.form.get(name)
        if raw is None:
            val = None
        else:
            # convert to numeric where appropriate
            if "int" in dtype or "float" in dtype or name.lower().find("charge")!=-1 or name.lower().find("revenue")!=-1 or name.lower().find("total")!=-1 or name.lower().find("tenure")!=-1:
                try:
                    # convert empty to NaN
                    val = float(raw) if raw != "" else None
                except:
                    val = None
            else:
                val = raw if raw != "" else None
        data[name] = [val]

    X_input = pd.DataFrame(data)

    # predict
    pred = clf.predict(X_input)[0]
    # get probability for positive class if available
    prob = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_input)[0]
        # assume positive class is 1 (churn)
        if len(proba) == 2:
            prob = float(proba[1])
        else:
            prob = float(proba[0])

    result_label = "Churn" if int(pred) == 1 else "Stayed"
    return render_template("result.html", result=result_label, probability=prob, inputs=X_input.to_dict(orient="records")[0])

if __name__ == "__main__":
    # For Anaconda/Dev use: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
