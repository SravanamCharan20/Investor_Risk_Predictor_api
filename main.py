from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="IT Risk Prediction API",
    description="API to predict risk labels (Low, Medium, High) for IT companies using a trained XGBoost model.",
    version="1.0.0"
)

# Load the trained model and encoders
try:
    model = joblib.load("xgboost_risk_model.pkl")
    label_encoder_tier = joblib.load("label_encoder_tier.pkl")
    label_encoder_risk = joblib.load("label_encoder_risk.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model or encoder files not found")

# Define input data model using Pydantic
class RiskInput(BaseModel):
    tier: Literal["Tier 1", "Tier 2", "Tier 3"]
    revenue_growth: float
    profit_margin: float
    debt_to_equity: float
    free_cash_flow: float
    layoff_frequency: float
    employee_attrition: float
    client_concentration: float
    geographic_diversification: float
    rnd_spending: float
    stock_volatility: float
    pe_ratio: float
    beta: float
    currency_risk: float
    global_it_spending: float
    digital_exposure: float

# Define response model
class RiskOutput(BaseModel):
    risk_label: str
    probability: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the IT Risk Prediction API. Use /predict to make predictions."}

# Prediction endpoint
@app.post("/predict", response_model=RiskOutput)
async def predict_risk(input_data: RiskInput):
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        # Preprocess input
        # Encode 'tier'
        try:
            df['tier'] = label_encoder_tier.transform(df['tier'])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'tier' value. Must be 'Tier 1', 'Tier 2', or 'Tier 3'.")

        # Ensure numerical columns are float
        numerical_cols = [
            'revenue_growth', 'profit_margin', 'debt_to_equity', 'free_cash_flow',
            'layoff_frequency', 'employee_attrition', 'client_concentration',
            'geographic_diversification', 'rnd_spending', 'stock_volatility',
            'pe_ratio', 'beta', 'currency_risk', 'global_it_spending', 'digital_exposure'
        ]
        df[numerical_cols] = df[numerical_cols].astype(float)

        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        predicted_label = label_encoder_risk.inverse_transform([prediction])[0]
        max_probability = float(np.max(probabilities))

        return {"risk_label": predicted_label, "probability": max_probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the app (for local testing, use: uvicorn fastapi_risk_prediction:app --host 0.0.0.0 --port 8000)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)