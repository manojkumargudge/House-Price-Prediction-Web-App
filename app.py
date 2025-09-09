from flask import Flask, request, render_template
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained model
model_path = os.path.join("..", "house_price_pipeline.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        try:
            # Extract features from form
            feature_names = ["OverallQual","GrLivArea","GarageCars",
                             "GarageArea","TotalBsmtSF","FullBath","YearBuilt"]
            features = [float(request.form[f]) for f in feature_names]

            # Convert input to DataFrame with correct column names
            features_df = pd.DataFrame([features], columns=feature_names)

            # Make prediction
            pred = model.predict(features_df)
            prediction = f"Predicted House Price: ${pred[0]:,.2f}"

        except ValueError:
            prediction = "Please enter valid numeric values for all fields."
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
