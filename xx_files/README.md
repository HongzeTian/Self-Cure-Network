# Notes

### Data Location

- The training (and validation) data is from "combined_Tsinghua" folder
- The test data (manually labelled from NTU) can be found at https://intellikblob.blob.core.windows.net/emotionntu/manual-relabel/

### Preprocessing Scripts

- Training_Data_Exploration.ipynb - Analysis of number of training data per emotion
- DataCleaning.ipynb - How I took manually relabelled images and deleted duplicate images in the NTU dataset

### Training Scripts

- FastAI Training.ipynb - Training script using FastAI library.

### Inference Scripts

- inference.ipynb - Plot confusion matrix and print classification report from onnx file and pth files.

### Current Model Statistics (Trained with FastAI)

- Filename: 7emotion_080121.pkl (https://intellikblob.blob.core.windows.net/emotionntu/7emotion_080121.pkl) / 7emotion_080121.pth (https://intellikblob.blob.core.windows.net/emotionntu/emotion7_080121.pth)
- Baseline model: emotion_recognition.onnx (https://intellikblob.blob.core.windows.net/emotionntu/emotion_recognition.onnx)
- Classification Report on Validation dataset (20% of total training data)

  ```
                precision    recall  f1-score   support

         angry       0.97      0.96      0.96       224
       disgust       0.95      0.89      0.92       195
          fear       0.98      0.98      0.98        89
         happy       0.98      0.98      0.98      1217
       neutral       0.94      0.96      0.95       694
           sad       0.95      0.96      0.96       538
      surprise       0.95      0.96      0.96       390

      accuracy                           0.96      3347
     macro avg       0.96      0.96      0.96      3347
  weighted avg       0.96      0.96      0.96      3347
  ```

- Classification Report on Test dataset (Manually labelled NTU dataset)

  ```
                precision    recall  f1-score   support

      surprise       0.86      0.78      0.82        98
          fear       0.15      0.02      0.04        97
       disgust       0.44      0.40      0.42       128
         happy       0.86      0.94      0.89       358
           sad       0.49      0.49      0.49       110
         angry       0.50      0.46      0.48       126
       neutral       0.54      0.83      0.66       157

      accuracy                           0.66      1074
     macro avg       0.55      0.56      0.54      1074
  weighted avg       0.62      0.66      0.63      1074
  ```

### Running Web API Server for Prediction

- Dependenices
  - FastAPI (server)
  - FastAI (DL library)
- Run fastapi server
