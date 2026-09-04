import pandas as pd
import numpy as np
import pickle

# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("Churn_Modelling.csv")

print("Dataset Loaded Successfully\n")
print(data.head())

print("\nDataset Shape:", data.shape)

# ==============================
# FEATURES & TARGET
# ==============================

X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

# ==============================
# ENCODING
# ==============================

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Geography Encoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Gender Encoding
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])

# ==============================
# SPLIT DATA
# ==============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==============================
# FEATURE SCALING
# ==============================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==============================
# BUILD ANN
# ==============================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining Started...\n")

model.fit(X_train, y_train, epochs=20, batch_size=32)

# ==============================
# EVALUATE MODEL
# ==============================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n[OK] Final Accuracy:", accuracy)

# ==============================
# SAMPLE PREDICTION
# ==============================

print("\n🔍 Testing on a sample input...\n")

sample = X_test[0].reshape(1, -1)

pred = model.predict(sample)

print("Raw Prediction Value:", pred)

if pred > 0.5:
    print("Customer will EXIT ❌")
else:
    print("Customer will STAY ✅")

print(f"Churn Probability: {pred[0][0]*100:.2f}%")

# ==============================
# USER INPUT PREDICTION
# ==============================

print("\n🎯 Try your own input:")

try:
    credit_score = int(input("Enter Credit Score: "))
    geography = input("Enter Geography (France/Spain/Germany): ")
    gender = input("Enter Gender (Male/Female): ")
    age = int(input("Enter Age: "))
    tenure = int(input("Enter Tenure: "))
    balance = float(input("Enter Balance: "))
    num_products = int(input("Enter Number of Products: "))
    has_card = int(input("Has Credit Card? (1/0): "))
    is_active = int(input("Is Active Member? (1/0): "))
    salary = float(input("Enter Estimated Salary: "))

    # Encode Geography
    geo_map = {
        'France': [1, 0, 0],
        'Spain': [0, 1, 0],
        'Germany': [0, 0, 1]
    }
    geo_encoded = geo_map.get(geography, [1, 0, 0])

    # Encode Gender
    gender_encoded = 1 if gender == "Male" else 0

    # Combine input
    user_input = geo_encoded + [
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        is_active,
        salary
    ]

    user_input = np.array(user_input).reshape(1, -1)

    # Scale input
    user_input = sc.transform(user_input)

    # Predict
    pred = model.predict(user_input)

    print("\n🔮 Prediction Result:", pred)

    if pred > 0.5:
        print("👉 Customer will EXIT ❌")
    else:
        print("👉 Customer will STAY ✅")

    print(f"Churn Probability: {pred[0][0]*100:.2f}%")

except:
    print("Invalid input! Please try again.")

    import pickle

# Save model
model.save("churn_model.h5")

# Save scaler
pickle.dump(sc, open("scaler.pkl", "wb"))

# Save encoder (if used)
pickle.dump(ct, open("encoder.pkl", "wb"))

print("✅ Model & preprocessing saved successfully!")