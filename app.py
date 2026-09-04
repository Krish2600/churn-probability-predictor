from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_key_for_churn_predictor')

# Define paths to model weights and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_H5_PATH = os.path.join(BASE_DIR, "churn_model.h5")

# Global variables for weights, scaler, and optional Keras model
weights = None
scaler = None
keras_model = None

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def numpy_nn_predict(scaled_features_array):
    """Fast, 100% mathematically exact NumPy forward pass of 3-layer ANN."""
    w = weights
    h1 = relu(np.dot(scaled_features_array, w['W1']) + w['b1'])
    h2 = relu(np.dot(h1, w['W2']) + w['b2'])
    out = sigmoid(np.dot(h2, w['W3']) + w['b3'])
    return float(out[0][0])

def get_model_and_scaler():
    """Lazily loads model weights and StandardScaler object into memory."""
    global weights, scaler, keras_model
    if (weights is not None or keras_model is not None) and scaler is not None:
        return True

    try:
        # Load Scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)

        # Load Weights for NumPy forward pass
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "rb") as f:
                weights = pickle.load(f)
            print("[OK] Neural network weights loaded for ultra-fast NumPy inference.")
            return True
        
        # Fallback to Keras if weights file is not found
        if os.path.exists(MODEL_H5_PATH):
            try:
                from tensorflow.keras.models import load_model, Sequential
                from tensorflow.keras.layers import Dense, Input
                try:
                    keras_model = load_model(MODEL_H5_PATH, compile=False)
                except Exception:
                    m = Sequential([
                        Input(shape=(12,)),
                        Dense(6, activation='relu'),
                        Dense(6, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    m.load_weights(MODEL_H5_PATH)
                    keras_model = m
                print("[OK] Keras ANN model loaded into memory.")
                return True
            except Exception as e_keras:
                print(f"[WARNING] Keras load attempt failed: {e_keras}")

        print(f"[ERROR] Model files missing. SCALER: {os.path.exists(SCALER_PATH)}, WEIGHTS: {os.path.exists(WEIGHTS_PATH)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model or scaler: {e}")

    return False

# Initial load on start
get_model_and_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for cloud deployments & load balancers."""
    is_ready = get_model_and_scaler()
    return jsonify({
        "status": "ok" if is_ready else "unhealthy",
        "model_loaded": weights is not None or keras_model is not None,
        "scaler_loaded": scaler is not None
    }), 200 if is_ready else 503

def process_features(form_data, scaler_obj):
    """Sanitizes form input and extracts scaled features for model prediction."""
    errors = []

    credit_score_str = form_data.get('credit_score', '')
    if not credit_score_str.isdigit():
        errors.append("Credit Score must be a valid number.")
    else:
        credit_score = int(credit_score_str)
        if not (300 <= credit_score <= 850):
            errors.append("Credit Score must be between 300 and 850.")

    geography = form_data.get('geography', '')
    if geography not in ["France", "Spain", "Germany"]:
        errors.append("Invalid Geography selected.")

    gender = form_data.get('gender', '')
    if gender not in ["Male", "Female"]:
        errors.append("Invalid Gender selected.")

    age_str = form_data.get('age', '')
    if not age_str.isdigit():
        errors.append("Age must be a valid number.")
    else:
        age = int(age_str)
        if not (18 <= age <= 100):
            errors.append("Age must be between 18 and 100.")

    tenure_str = form_data.get('tenure', '')
    if not tenure_str.isdigit():
        errors.append("Tenure must be a valid number.")
    else:
        tenure = int(tenure_str)
        if not (0 <= tenure <= 10):
            errors.append("Tenure must be between 0 and 10 years.")

    balance_str = form_data.get('balance', '')
    try:
        balance = float(balance_str)
        if balance < 0:
            errors.append("Balance cannot be negative.")
    except ValueError:
        errors.append("Balance must be a valid number.")

    num_products_str = form_data.get('num_products', '')
    if not num_products_str.isdigit():
        errors.append("Number of Products must be a valid number.")
    else:
        num_products = int(num_products_str)
        if not (1 <= num_products <= 4):
            errors.append("Number of Products must be between 1 and 4.")

    has_card_str = form_data.get('has_card', '0')
    if not has_card_str.isdigit() or int(has_card_str) not in [0, 1]:
        errors.append("Has Credit Card must be 0 or 1.")
    else:
        has_card = int(has_card_str)

    is_active_str = form_data.get('is_active', '0')
    if not is_active_str.isdigit() or int(is_active_str) not in [0, 1]:
        errors.append("Is Active Member must be 0 or 1.")
    else:
        is_active = int(is_active_str)

    salary_str = form_data.get('salary', '')
    try:
        salary = float(salary_str)
        if salary < 0:
            errors.append("Estimated Salary cannot be negative.")
    except ValueError:
        errors.append("Estimated Salary must be a valid number.")

    if errors:
        return None, errors

    # Geography encoding: France [1, 0, 0], Spain [0, 1, 0], Germany [0, 0, 1]
    if geography == "France":
        geo = [1, 0, 0]
    elif geography == "Spain":
        geo = [0, 1, 0]
    else:
        geo = [0, 0, 1]

    # Gender encoding: Male -> 1, Female -> 0
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

    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler_obj.transform(features_array)

    return scaled_features, []

def run_inference(scaled_features_array):
    """Executes inference using NumPy forward pass or Keras model fallback."""
    if weights is not None:
        return numpy_nn_predict(scaled_features_array)
    elif keras_model is not None:
        return float(keras_model.predict(scaled_features_array)[0][0])
    else:
        raise RuntimeError("No prediction engine available.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint returning JSON predictions for modern AJAX apps."""
    if not get_model_and_scaler():
        return jsonify({"success": False, "error": "Prediction model is currently unavailable."}), 503

    try:
        scaled_features, errors = process_features(request.form, scaler)
        if errors:
            return jsonify({"success": False, "error": " ".join(errors)}), 400

        pred = run_inference(scaled_features)
        prob = round(float(pred) * 100, 2)
        result = "Customer will EXIT" if pred > 0.5 else "Customer will STAY"
        
        return jsonify({
            "success": True,
            "probability": prob,
            "prediction_text": result,
            "will_exit": bool(pred > 0.5)
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal error during inference: {str(e)}"}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Form submission fallback route for standard HTML form post."""
    if request.method == 'GET':
        return redirect(url_for('home'))

    if not get_model_and_scaler():
        flash("Prediction service is currently unavailable. Please try again later.", "error")
        return redirect(url_for('home'))

    try:
        scaled_features, errors = process_features(request.form, scaler)
        if errors:
            for error in errors:
                flash(error, "error")
            return render_template('index.html', form_data=request.form)

        pred = run_inference(scaled_features)
        result = "Customer will EXIT" if pred > 0.5 else "Customer will STAY"
        prob = round(float(pred) * 100, 2)

        return render_template('index.html', prediction_text=result, probability=prob, form_data=request.form)
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('home'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)