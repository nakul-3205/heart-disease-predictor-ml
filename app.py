from flask import Flask, render_template, request
import numpy as np
import pandas as pd # Although not directly used for prediction, often useful for data handling
import joblib
import os


app = Flask(__name__)


model_path = "models.pkl"
scaler_path = "scalerss.pkl"


try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Please ensure '{model_path}' and '{scaler_path}' are in the same directory as app.py.")
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    # Initialize input_data with default empty values for form persistence
    input_data = {
        "age": "", "sex": "", "cp": "", "trestbps": "", "chol": "", 
        "fbs": "", "restecg": "", "thalach": "", "exang": "", 
        "oldpeak": "", "slope": "", "ca": "", "thal": ""
    }
    error_message = None

    if request.method == "POST":
        # Check if model and scaler were successfully loaded
        if model is None or scaler is None:
            error_message = "Prediction system not fully loaded. Please ensure 'model.pkl' and 'scalerss.pkl' exist and are valid."
        else:
            try:
                # Get data from form and store in input_data for persistence
                input_data["age"] = float(request.form["age"])
                input_data["sex"] = int(request.form["sex"])
                input_data["cp"] = int(request.form["cp"])
                input_data["trestbps"] = float(request.form["trestbps"])
                input_data["chol"] = float(request.form["chol"])
                input_data["fbs"] = int(request.form["fbs"])
                input_data["restecg"] = int(request.form["restecg"])
                input_data["thalach"] = float(request.form["thalach"])
                input_data["exang"] = int(request.form["exang"])
                input_data["oldpeak"] = float(request.form["oldpeak"])
                input_data["slope"] = int(request.form["slope"])
                input_data["ca"] = int(request.form["ca"])
                input_data["thal"] = int(request.form["thal"])

                # Prepare data in the correct order for your model
                raw_features = [
                    input_data["age"],
                    input_data["sex"],
                    input_data["cp"],
                    input_data["trestbps"],
                    input_data["chol"],
                    input_data["fbs"],
                    input_data["restecg"],
                    input_data["thalach"],
                    input_data["exang"],
                    input_data["oldpeak"],
                    input_data["slope"],
                    input_data["ca"],
                    input_data["thal"]
                ]

               
                features_to_scale_indices = [0, 3, 4, 7, 9] 
                
                features_for_scaler = np.array([[raw_features[i] for i in features_to_scale_indices]])
                
                scaled_values = scaler.transform(features_for_scaler)[0]

                final_input = raw_features.copy()
                for i, scaled_val in zip(features_to_scale_indices, scaled_values):
                    final_input[i] = scaled_val

                # Predict
                result = model.predict([final_input])[0]
                
                # Interpret prediction
                if result == 1:
                    prediction = "No Disease ❤️ (Safe)"
                else:
                    prediction = "Possible Heart Disease ⚠️"

            except ValueError as e:
                error_message = f"Invalid input: Please ensure all fields are filled correctly with numbers. ({e})"
            except KeyError as e:
                error_message = f"Missing data: A required field was not submitted. ({e})"
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                print(f"Prediction processing error: {e}") # Log for debugging

    # Pass prediction, error, and input_data back to the template
    return render_template("index.html", prediction=prediction, form_data=input_data, error=error_message)

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000)) 
     app.run(host="0.0.0.0", port=port, debug=True)
