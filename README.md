# Automated-Software-Flaw-Estimation-and-Detection-Using-Machine-Learning

An intelligent system that predicts software defects using machine learning techniques with an interactive web interface for real-time analysis and prediction.

---

## Project Overview
This project focuses on identifying defective and non-defective software modules by analyzing dataset features using machine learning models. It automates data preprocessing, model training, evaluation, and prediction through a user-friendly web application built with Streamlit.

---

## Key Features
* **Automated Defect Prediction** – Classifies software modules as defective or non-defective  
* **Data Preprocessing Pipeline** – Handles missing values, normalization, and encoding automatically  
* **Imbalance Handling** – Uses oversampling techniques to manage class imbalance  
* **Model Training & Evaluation** – Trains models and evaluates them using accuracy, confusion matrix, and classification reports  
* **Interactive Web App** – Upload datasets, train models, visualize results, and make real-time predictions using Streamlit  
* **Visualization Support** – Displays class distribution, feature importance, and performance metrics  

---

## Technologies Used
* Python  
* Machine Learning  
* Random Forest Classifier  
* Streamlit  
* Pandas, NumPy  
* Scikit-learn  
* Matplotlib, Seaborn  
* Imbalanced-learn (Oversampling)

---

## Prerequisites
Before you begin, ensure you have the following installed:
* Python (3.7 or higher recommended)  
* pip package manager  

---

## Installation

### 1. Clone the repository
```bash
git clone [Your Repository URL]
cd [Your Repository Folder]
```


### 2. Install required libraries
```bash
pip install streamlit numpy pandas scikit-learn matplotlib seaborn imbalanced-learn joblib
```

### 3. Usage
Run the Streamlit application using:
```bash
streamlit run app.py
```

## How It Works
1. User uploads a CSV dataset through the web interface  
2. The system performs data cleaning, preprocessing, and feature scaling  
3. Class imbalance is handled using oversampling techniques  
4. The machine learning model is trained and saved automatically  
5. Model performance is evaluated using accuracy score, confusion matrix, and classification report  
6. Users can input custom feature values to get real-time defect predictions  
7. Feature importance and performance visualizations are displayed for better understanding  

---

## Future Enhancements
* Integration of deep learning models (CNN, LSTM)  
* Cloud deployment  
* Real-time CI/CD defect prediction  
* API integration for software pipelines  
* Multi-model comparison dashboard  

---

## Contributing
Contributions are welcome!  
If you find a bug or have suggestions for improvement, feel free to open an issue or submit a pull request.

