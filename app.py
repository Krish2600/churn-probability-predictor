from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import numpy as np
import json
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_key_for_churn_predictor')

# Paths to JSON weights and scaler parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_JSON_PATH = os.path.join(BASE_DIR, "model_weights.json")
SCALER_JSON_PATH = os.path.join(BASE_DIR, "scaler_params.json")

# Global neural network matrices
W1, b1 = None, None
W2, b2 = None, None
W3, b3 = None, None
mean_vec, scale_vec = None, None

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_engine():
    """Loads weights and scaler parameters from JSON files into NumPy matrices."""
    global W1, b1, W2, b2, W3, b3, mean_vec, scale_vec
    if W1 is not None and mean_vec is not None:
        return True

    try:
        if os.path.exists(WEIGHTS_JSON_PATH) and os.path.exists(SCALER_JSON_PATH):
            # Load Weights
            with open(WEIGHTS_JSON_PATH, "r") as f:
                w_data = json.load(f)
                W1 = np.array(w_data['W1'], dtype=np.float64)
                b1 = np.array(w_data['b1'], dtype=np.float64)
                W2 = np.array(w_data['W2'], dtype=np.float64)
                b2 = np.array(w_data['b2'], dtype=np.float64)
                W3 = np.array(w_data['W3'], dtype=np.float64)
                b3 = np.array(w_data['b3'], dtype=np.float64)

            # Load Scaler
            with open(SCALER_JSON_PATH, "r") as f:
                s_data = json.load(f)
                mean_vec = np.array(s_data['mean'], dtype=np.float64)
                scale_vec = np.array(s_data['scale'], dtype=np.float64)

            print("[OK] Neural network matrices & scaler loaded from JSON successfully.")
            return True
        else:
            print(f"[ERROR] JSON files missing: weights={os.path.exists(WEIGHTS_JSON_PATH)}, scaler={os.path.exists(SCALER_JSON_PATH)}")
    except Exception as e:
        print(f"[ERROR] JSON Engine loading error: {e}")

    return False

# Initial load on start
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
        "engine_ready": is_ready
    }), 200 if is_ready else 503

@app.route('/api/test_post', methods=['POST'])
def test_post():
    """Diagnostic POST endpoint to test Gunicorn request handling."""
    data = parse_input_data()
    return jsonify({"success": True, "received_data": data}), 200

def parse_input_data():
    """Robustly extracts parameters whether submitted as Form-Data, Form-Encoded, or JSON."""
    data_dict = {}
    
    # 1. Try parsing JSON
    if request.is_json:
        try:
            j = request.get_json(silent=True)
            if j and isinstance(j, dict):
                data_dict.update(j)
        except Exception:
            pass

    # 2. Try parsing Form Data
    if request.form:
        try:
            data_dict.update(request.form.to_dict())
        except Exception:
            pass

    # 3. Fallback to args
    if request.args:
        try:
            data_dict.update(request.args.to_dict())
        except Exception:
            pass

    return data_dict

def process_and_scale_features(data):
    """Sanitizes inputs and returns scaled feature array using pure NumPy."""
    global mean_vec, scale_vec
    if mean_vec is None or scale_vec is None:
        if not load_engine():
            return None, ["Engine parameters could not be loaded."]

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
    scaled_features = (features_array - mean_vec) / scale_vec

    return scaled_features, []

def run_numpy_inference(scaled_features_array):
    """Executes ultra-fast 3-layer neural network forward pass using NumPy."""
    global W1, b1, W2, b2, W3, b3
    if W1 is None or b1 is None:
        load_engine()
    h1 = relu(np.dot(scaled_features_array, W1) + b1)
    h2 = relu(np.dot(h1, W2) + b2)
    out = sigmoid(np.dot(h2, W3) + b3)
    return float(out[0][0])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint returning JSON predictions for modern AJAX apps."""
    if not load_engine():
        return jsonify({"success": False, "error": "Prediction engine is currently unavailable."}), 503

    try:
        data = parse_input_data()
        scaled_features, errors = process_and_scale_features(data)
        if errors:
            return jsonify({"success": False, "error": " ".join(errors)}), 400

        pred = run_numpy_inference(scaled_features)
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

        pred = run_numpy_inference(scaled_features)
        result = "Customer will EXIT" if pred > 0.5 else "Customer will STAY"
        prob = round(float(pred) * 100, 2)

        return render_template('index.html', prediction_text=result, probability=prob, form_data=request.form)
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('home'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)