from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler
# import model
from model import HeartDiseaseModel

class XORData(BaseModel):
    data: List[List[float]]

app = FastAPI()
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/root")
async def root():
    try:
        return { "message": "hello, world" }
    except Exception as e:
        return { "error": str(e) }, 501


@app.get("/api/v1/docs")
async def get_doc():
    try:
        return { "docs": "These are the docs!" }
    except Exception as e:
        return { "error": str(e) }, 501


@app.post("/api/v1/heart-disease")
async def prediction_route(data: XORData):
    try:
        model = HeartDiseaseModel() # Init model
        scaler = StandardScaler() # Init normalizer
        input_data = data.data # Extract data from post request
        input_array = np.array(input_data) # Convert data to numpy array
        scaler.fit(input_array) # Fit data for model input
        scaled_input = scaler.transform(input_array) # Transform data
        predictions = model.predict(scaled_input) # Make prediction
        predictions = predictions.round().flatten()
        predictions = (predictions.numpy() > 0.5).astype(int)
        
        # predictions_list = predictions.tolist() # Round to the nearst 1's place
        # rounded_predictions = [[round(pred) for pred in sublist] for sublist in predictions]
        print(predictions)
        return { "prediction": predictions.tolist() } # Return a prediction to the end user
    except Exception as e:
        return { "error": str(e) }, 501