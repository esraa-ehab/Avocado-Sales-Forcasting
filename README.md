# 🥑 Avocado Sales Volume Prediction

This project predicts the next total sales volume of avocados based on the past 30 sales values provided by the user. It involves a complete workflow of data cleaning, training with machine learning (LSTM model), forecasting, evaluation, and deployment as a [web application](https://web-production-cb31.up.railway.app/).

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data Preparation](#data-preparation)  
- [Model Training](#model-training)  
- [Forecasting & Evaluation](#forecasting--evaluation)  
- [MLflow Integration](#mlflow-integration)  
- [Web App Deployment](#web-app-deployment)  
- [Usage](#usage)  
- [Requirements](#requirements)  
- [Installation](#installation)
  
---

## Project Overview

The goal is to predict avocado sales volume using historical sales data. I trained an LSTM neural network to forecast the next sales value given the last 30 sales volumes. The model is wrapped inside a FastAPI web application where users input their past 30 sales values and get a predicted next sales volume with visualizations.

---

## Data Preparation

- Collected raw avocado sales data.
- Cleaned and preprocessed the data by handling missing values and scaling features.
- Extracted sequences of 30 past sales volumes as input samples.
- Split data into training and validation sets.

---

## Model Training

- Used an LSTM neural network architecture for sequential forecasting.
- Scaled input features using a MinMaxScaler.
- Trained the model on prepared sequences.
- Saved the trained model (`LSTM_model.h5`) and scaler (`scaler.pkl`) for deployment.

---

## Forecasting & Evaluation

- The trained model predicts the next sales volume based on input sequences.
- Evaluated model performance using standard regression metrics.
- Visualized results comparing actual vs predicted values.
- Added log-scale visualization to better handle large ranges of sales volumes.

---

## MLflow Integration

- Used MLflow to track experiments, parameters, and metrics.
- Logged the model and artifacts for easy reproducibility.
- Facilitated model versioning and deployment.

---

## Web App Deployment

- Developed a FastAPI application to expose the prediction model via a REST API.
- Created a user-friendly frontend with HTML, CSS, and Chart.js for input and visualizing predictions.
- Deployed the app on Railway for easy online access.
- To test the app click [here](https://web-production-cb31.up.railway.app/)
---

## Usage

1. Run the FastAPI app locally or access the deployed version.
2. Input exactly 30 past sales values separated by commas.
3. Click the **Predict** button.
4. View the predicted next sales volume.
5. See interactive charts showing both linear and logarithmic scale visualizations.

---

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- TensorFlow
- Scikit-learn
- Joblib
- Pydantic
- Chart.js (frontend via CDN)

---

## Installation

```bash
git clone https://github.com/esraa-ehab/Avocado-Sales-Forcasting
cd avocado-sales-prediction
pip install -r requirements.txt
uvicorn main:app --reload
