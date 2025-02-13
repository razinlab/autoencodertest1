from typing import List, Dict
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import __version__
from app.model.model import predict

app = FastAPI()
class DataIn(BaseModel):
    data: List[Dict]
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

class DataOut(BaseModel):
    predictions: List[int]

@app.get('/')
def health():
    return {'status': 'UP', 'version': __version__}

@app.post('/predict', response_model=DataOut)
def predict(payload: DataIn):
    df = pd.DataFrame(payload.data)
    predictions = predict(df)
    return DataOut(predictions=predictions)