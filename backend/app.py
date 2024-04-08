from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
import model

class XORData(BaseModel):
    data: List[List[float]]

app = FastAPI()
model.load_XOR_model()

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

@app.post("/api/v1/predict")
async def prediction_route(data: XORData):
    try:
        data = np.array(data.data, dtype=float)
        prediction = model.predict(data).round()
        prediction_list = prediction.tolist()
        return { "prediction": prediction_list }
    except Exception as e:
        return {"error": str(e)}, 501