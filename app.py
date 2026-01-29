import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

st.title("Automated Software flaw Estimation and Detection using Machine Learning")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.write(df.head())

    df = df.drop_duplicates()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['defects'] = LabelEncoder().fit_transform(df['defects'])

    X = df.iloc[:, :-1]
    y = df['defects']
    X = StandardScaler().fit_transform(X)

    st.write("Target Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette='coolwarm', ax=ax)
    st.pyplot(fig)

    sampler = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    st.write("Training Model...")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model, 'rf_oversampling_model.pkl')
    st.success("Model trained & saved successfully!")

    rf_model = joblib.load('rf_oversampling_model.pkl')

    y_pred = rf_model.predict(X_test)

    st.write("Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.sidebar.header("Make a Prediction")
    input_data = st.sidebar.text_input("Enter feature values (comma-separated)")
    
    if st.sidebar.button("Predict"):
        try:
            new_sample = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)
            new_sample = StandardScaler().fit_transform(new_sample)
            prediction = rf_model.predict(new_sample)
            result = "Defective" if prediction[0] == 1 else "Not Defective"
            st.sidebar.success(f"Prediction: {result}")
        except:
            st.sidebar.error("Invalid input. Please enter valid numbers.")

    st.write("Feature Importance")
    feature_importances = pd.Series(rf_model.feature_importances_, index=df.columns[:-1])
    fig, ax = plt.subplots()
    feature_importances.nlargest(10).plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)

st.sidebar.write("Upload your dataset and start predicting!")
