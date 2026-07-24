"""
PROJECT 6 - PART 2: SERVE THE MODEL WITH A FASTAPI WEB API


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# STEP 1: LOAD THE TRAINED MODEL 

model = joblib.load("purchase_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# STEP 2: CREATE THE FASTAPI APP

app = FastAPI(title="Purchase Prediction API")


# STEP 3: DEFINE THE EXPECTED INPUT SHAPE (using Pydantic)

class CustomerData(BaseModel):
    age: int
    time_on_site_minutes: float
    pages_viewed: int
    past_purchases: int
    cart_value: float


# STEP 4: A SIMPLE "IS THE SERVER ALIVE" ENDPOINT

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Purchase Prediction API is running"}


# STEP 5: THE PREDICTION ENDPOINT

@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([{
        "age": customer.age,
        "time_on_site_minutes": customer.time_on_site_minutes,
        "pages_viewed": customer.pages_viewed,
        "past_purchases": customer.past_purchases,
        "cart_value": customer.cart_value,
    }])[feature_columns]

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "will_purchase": bool(prediction),
        "purchase_probability": round(probability, 3)
    }
