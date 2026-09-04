from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_key_for_churn_predictor')

# Define paths to weights and scaler parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.pkl")
SCALER_PARAMS_PATH = os.path.join(BASE_DIR, "scaler_params.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Global variables for weights and scaler parameters
weights = None
scaler_params = None
scaler_obj = None

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

def load_engine():
    """Lazily loads weights and scaler parameters into memory."""
    global weights, scaler_params, scaler_obj
    if (weights is not None or scaler_params is not None or scaler_obj is not None):
        return True

    try:
        # 1. Load Scaler parameters
        if os.path.exists(SCALER_PARAMS_PATH):
            with open(SCALER_PARAMS_PATH, "rb") as f:
                scaler_params = pickle.load(f)
            print("[OK] Scaler parameters loaded for pure NumPy scaling.")
        elif os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                scaler_obj = pickle.load(f)
            print("[OK] StandardScaler object unpickled.")

        # 2. Load Model Weights
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "rb") as f:
                weights = pickle.load(f)
            print("[OK] Neural network weights loaded for ultra-fast NumPy inference.")
            return True
    except Exception as e:
        print(f"[ERROR] Engine loading error: {e}")

    return False

# Load engine on startup
load_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for cloud deployments & load balancers."""
    is_ready = load_engine()
    return jsonify({
        "status": "ok" if is_ready else "unhealthy",
        "weights_loaded": weights is not None,
        "scaler_loaded": scaler_params is not None or scaler_obj is not None
    }), 200 if is_ready else 503

def parse_input_data():
    """Extracts parameters whether submitted as form-data or JSON."""
    if request.is_json and request.get_json():
        return request.get_json()
    return request.form

def process_and_scale_features(data):
    """Sanitizes inputs and returns scaled feature array."""
    errors = []

    # 1. Credit Score
    try:
        credit_score = int(float(data.get('credit_score', 0)))
        if not (300 <= credit_score <= 850):
            errors.append("Credit Score must be between 300 and 850.")
    except (ValueError, TypeError):
        errors.append("Credit Score must be a valid number.")

    # 2. Geography
    geography = str(data.get('geography', '')).strip()
    if geography not in ["France", "Spain", "Germany"]:
        errors.append("Invalid Geography selected.")

    # 3. Gender
    gender = str(data.get('gender', '')).strip()
    if gender not in ["Male", "Female"]:
        errors.append("Invalid Gender selected.")

    # 4. Age
    try:
        age = int(float(data.get('age', 0)))
        if not (18 <= age <= 100):
            errors.append("Age must be between 18 and 100.")
    except (ValueError, TypeError):
        errors.append("Age must be a valid number.")

    # 5. Tenure
    try:
        tenure = int(float(data.get('tenure', 0)))
        if not (0 <= tenure <= 10):
            errors.append("Tenure must be between 0 and 10 years.")
    except (ValueError, TypeError):
        errors.append("Tenure must be a valid number.")

    # 6. Balance
    try:
        balance = float(data.get('balance', 0))
        if balance < 0:
            errors.append("Balance cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Balance must be a valid number.")

    # 7. Number of Products
    try:
        num_products = int(float(data.get('num_products', 1)))
        if not (1 <= num_products <= 4):
            errors.append("Number of Products must be between 1 and 4.")
    except (ValueError, TypeError):
        errors.append("Number of Products must be between 1 and 4.")

    # 8. Has Credit Card
    try:
        has_card = int(float(data.get('has_card', 0)))
        if has_card not in [0, 1]:
            has_card = 1 if has_card > 0 else 0
    except (ValueError, TypeError):
        has_card = 0

    # 9. Is Active Member
    try:
        is_active = int(float(data.get('is_active', 0)))
        if is_active not in [0, 1]:
            is_active = 1 if is_active > 0 else 0
    except (ValueError, TypeError):
        is_active = 0

    # 10. Estimated Salary
    try:
        salary = float(data.get('salary', 0))
        if salary < 0:
            errors.append("Estimated Salary cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Estimated Salary must be a valid number.")

    if errors:
        return None, errors

    # Geography encoding: France [1, 0, 0], Spain [0, 1, 0], Germany [0, 0, 1]
    if geography == "France":
        geo = [1.0, 0.0, 0.0]
    elif geography == "Spain":
        geo = [0.0, 1.0, 0.0]
    else:
        geo = [0.0, 0.0, 1.0]

    # Gender encoding: Male -> 1, Female -> 0
    gender_val = 1.0 if gender == "Male" else 0.0

    features = geo + [
        float(credit_score),
        gender_val,
        float(age),
        float(tenure),
        float(balance),
        float(num_products),
        float(has_card),
        float(is_active),
        float(salary)
    ]

    features_array = np.array(features, dtype=np.float64).reshape(1, -1)

    # Scaling via NumPy parameters or StandardScaler object
    if scaler_params is not None:
        scaled_features = (features_array - scaler_params['mean']) / scaler_params['scale']
    elif scaler_obj is not None:
        scaled_features = scaler_obj.transform(features_array)
    else:
        scaled_features = features_array

    return scaled_features, []

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint returning JSON predictions for modern AJAX apps."""
    if not load_engine():
        return jsonify({"success": False, "error": "Prediction model is currently unavailable."}), 503

    try:
        data = parse_input_data()
        scaled_features, errors = process_and_scale_features(data)
        if errors:
            return jsonify({"success": False, "error": " ".join(errors)}), 400

        pred = numpy_nn_predict(scaled_features)
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

    if not load_engine():
        flash("Prediction service is currently unavailable. Please try again later.", "error")
        return redirect(url_for('home'))

    try:
        data = parse_input_data()
        scaled_features, errors = process_and_scale_features(data)
        if errors:
            for error in errors:
                flash(error, "error")
            return render_template('index.html', form_data=request.form)

        pred = numpy_nn_predict(scaled_features)
        result = "Customer will EXIT" if pred > 0.5 else "Customer will STAY"
        prob = round(float(pred) * 100, 2)

        return render_template('index.html', prediction_text=result, probability=prob, form_data=request.form)
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('home'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)