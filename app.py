import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="British Airways Customer Booking Analysis",
    layout="wide"
)

st.title("✈️ British Airways Customer Booking Analysis")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/customer_booking.csv",
        encoding="latin1"
    )
    return df

df = load_data()

# -----------------------------
# Dataset Overview
# -----------------------------
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Columns")
st.write(df.columns.tolist())

# -----------------------------
# Missing Values
# -----------------------------
st.header("🔍 Data Quality Check")

missing_values = df.isnull().sum()

st.write("Missing Values Per Column")
st.dataframe(missing_values[missing_values > 0])

# -----------------------------
# Data Preprocessing
# -----------------------------
df = df.dropna()

X = df.drop("booking_complete", axis=1)
y = df["booking_complete"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Booking Completion Analysis
# -----------------------------
st.header("📈 Booking Completion Analysis")

booking_counts = df["booking_complete"].value_counts()

booking_counts.index = [
    "Not Completed",
    "Completed"
]


st.bar_chart(booking_counts)

# -----------------------------
# Model Performance
# -----------------------------
st.header("🤖 Random Forest Model Results")

st.metric(
    label="Model Accuracy",
    value=f"{accuracy:.2%}"
)

st.subheader("Classification Report")

st.text(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------
# Feature Importance
# -----------------------------
st.header("⭐ Top 10 Important Features")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

st.dataframe(importance)

st.bar_chart(
    importance.set_index("Feature")
)

# -----------------------------
# Business Insights
# -----------------------------
st.header("💡 Business Insights")

st.write("""
- Random Forest was used to identify factors affecting booking completion.
- Feature importance highlights the variables that most influence customer booking behavior.
- These insights can help airlines improve marketing, pricing, and customer engagement strategies.
""")