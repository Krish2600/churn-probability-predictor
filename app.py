from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages' # Needed for flash messages

# Define paths to model and scaler
MODEL_PATH = "churn_model.h5"
SCALER_PATH = "scaler.pkl"

# Load model and scaler only once when the app starts
try:
    model = load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    # Exit or handle the error appropriately if the model/scaler can't be loaded
    # For a production app, you might want to serve a maintenance page.
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        flash("Prediction service is currently unavailable. Please try again later.", "error")
        return redirect(url_for('home'))

    try:
        # Basic server-side validation
        errors = []
        
        credit_score_str = request.form['credit_score']
        if not credit_score_str.isdigit():
            errors.append("Credit Score must be a number.")
        else:
            credit_score = int(credit_score_str)
            if not (300 <= credit_score <= 850): # Assuming typical credit score range
                errors.append("Credit Score must be between 300 and 850.")

        geography = request.form['geography']
        if geography not in ["France", "Spain", "Germany"]:
            errors.append("Invalid Geography selected.")

        gender = request.form['gender']
        if gender not in ["Male", "Female"]:
            errors.append("Invalid Gender selected.")
        
        age_str = request.form['age']
        if not age_str.isdigit():
            errors.append("Age must be a number.")
        else:
            age = int(age_str)
            if not (18 <= age <= 100): # Assuming reasonable age range
                errors.append("Age must be between 18 and 100.")

        tenure_str = request.form['tenure']
        if not tenure_str.isdigit():
            errors.append("Tenure must be a number.")
        else:
            tenure = int(tenure_str)
            if not (0 <= tenure <= 10): # Assuming reasonable tenure range
                errors.append("Tenure must be between 0 and 10 years.")

        balance_str = request.form['balance']
        try:
            balance = float(balance_str)
            if balance < 0:
                errors.append("Balance cannot be negative.")
        except ValueError:
            errors.append("Balance must be a number.")

        num_products_str = request.form['num_products']
        if not num_products_str.isdigit():
            errors.append("Number of Products must be a number.")
        else:
            num_products = int(num_products_str)
            if not (1 <= num_products <= 4): # Assuming reasonable product range
                errors.append("Number of Products must be between 1 and 4.")

        has_card_str = request.form['has_card']
        if not has_card_str.isdigit() or int(has_card_str) not in [0, 1]:
            errors.append("Has Credit Card? must be 0 or 1.")
        else:
            has_card = int(has_card_str)

        is_active_str = request.form['is_active']
        if not is_active_str.isdigit() or int(is_active_str) not in [0, 1]:
            errors.append("Is Active Member? must be 0 or 1.")
        else:
            is_active = int(is_active_str)

        salary_str = request.form['salary']
        try:
            salary = float(salary_str)
            if salary < 0:
                errors.append("Estimated Salary cannot be negative.")
        except ValueError:
            errors.append("Estimated Salary must be a number.")

        if errors:
            for error in errors:
                flash(error, "error")
            return render_template('index.html', form_data=request.form) # Pass form data to re-populate

        # Geography encoding
        geo = [0, 0, 0] # Default to avoid index errors
        if geography == "France":
            geo = [1, 0, 0]
        elif geography == "Spain":
            geo = [0, 1, 0]
        else: # Germany
            geo = [0, 0, 1]

        # Gender encoding
        gender_val = 1 if gender == "Male" else 0

        features = geo + [
            credit_score,
            gender_val,
            age,
            tenure,
            balance,
            num_products,
            has_card,
            is_active,
            salary
        ]

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        pred = model.predict(features)[0][0]

        result = "Customer will EXIT ❌" if pred > 0.5 else "Customer will STAY ✅"
        prob = round(pred * 100, 2)

        return render_template('index.html', prediction_text=result, probability=prob)
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)