import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("Employee Attrition Prediction using Logistic Regression")

st.write("This application predicts whether an employee may leave the company or stay.")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ---------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Ensure all features are numeric
X = X.apply(pd.to_numeric)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# ---------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------
st.subheader("Model Evaluation")

st.write(f"Accuracy : {accuracy:.2f}")

st.write(f"MSE : {mse:.2f}")

st.write(f"RMSE : {rmse:.2f}")

st.write(f"MAE : {mae:.2f}")

st.write(f"R2 Score : {r2:.2f}")

st.write("Confusion Matrix")

st.write(cm)

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------
st.subheader("Enter Employee Details")

age = st.number_input("Age", 18, 60, 30)

daily_rate = st.number_input("Daily Rate", 100, 2000, 800)

distance = st.number_input("Distance From Home", 1, 50, 5)

monthly_income = st.number_input("Monthly Income", 1000, 50000, 10000)

years_company = st.number_input("Years At Company", 0, 40, 5)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
if st.button("Predict Attrition"):

    input_data = pd.DataFrame({
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_company]
    })

    # Add all missing columns from training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange columns in same order
    input_data = input_data[X.columns]

    # Convert to numeric
    input_data = input_data.apply(pd.to_numeric)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Prediction probability
    probability = model.predict_proba(input_scaled)

    # ---------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Employee is likely to leave the company.")
    else:
        st.success("Employee is likely to stay in the company.")

    st.write("Prediction Probability")

    st.write(probability)
