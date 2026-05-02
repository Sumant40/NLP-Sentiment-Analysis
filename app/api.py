from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class Input(BaseModel):
    text: str

@app.post("/predict")
def get_prediction(data: Input):
    return predict(data.text)