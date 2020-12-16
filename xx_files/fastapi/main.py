from typing import Optional
from fastapi import FastAPI, File, UploadFile
import torch
from pathlib import Path

from fastai.vision.all import load_learner, PILImage

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict_image(img):
    # Receives a PIL image and returns prediction and probability
    learn = load_learner('7emotion_080121.pkl')
    '''
      learn.predict expects an input of type in 
      - <class 'pandas.core.series.Series'>
      - <class 'pathlib.Path'>
      - <class 'str'>
      - <class 'torch.Tensor'>
      - <class 'numpy.ndarray'>
      - <class 'bytes'>
      - <class 'fastai.vision.core.PILImage'>
    '''
    pred, _, probs = learn.predict(img)
    return {"pred": pred, "probs": torch.max(probs).item()}
