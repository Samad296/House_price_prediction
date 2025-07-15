from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List

import uvicorn.server

# initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API For Prediction using Trained Linear Regression Model",
    version="1.0.0"

)

try:
    with open('linear_regression.pkl','rb') as file:
        model = pickle.load(file)
    with open ('scaler.pkl','rb') as file:
        scaler = pickle.load(file)
    print("Model and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading in model or scaler : {e}")
    print(f"Please ensure your trainning notebook and scaler file")
# Define input model
class Housefeatures(BaseModel):
    
    MedInc : float
    HouseAge : float
    AveRooms : float
    AveBedrms : float   # <- Your code expects 'AveBedrms'
    Population: float
    AveOccup : float
    Latitude : float
    Longitude : float
#Define output model
class Predictionresponse(BaseModel):
    prediction_price : float
    input_feature : dict

@app.get("/", response_class=HTMLResponse)
async def root():
    return"""""
         <html>
        <head>
            <title>House Price Prediction API</title>
        </head>
        <body>
            <h1>Welcome to the House Price Prediction API</h1>
            <p>Use <a href='/docs'>/docs</a> to test the API.</p>
        </body>
    </html>
    """

@app.post("/predict", response_model=Predictionresponse)
async def predict_house_price(feature : Housefeatures):
    try:
        input_data = np.array([[
            feature.MedInc,
            feature.HouseAge,
            feature.AveRooms,
            feature.AveBedrms,
            feature.Population,
            feature.AveOccup,
            feature.Latitude,
            feature.Longitude
        ]])
        # scale the input using saved scale
        input_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        # convert to actual price
        predicted_price = float(prediction*1000)
        return Predictionresponse(
            prediction_price = predicted_price,
            input_feature = feature.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
# Example Using and Test
if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)
