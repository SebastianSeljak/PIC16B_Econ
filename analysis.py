import sqlite3 as sql
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
import tqdm
import random
import matplotlib.pyplot as plt
from models import *
from utils import *
import json
import sys



import pickle
# Load the model from the file
with open('models/model_combined.pkl', 'rb') as f:
    model, loss_data= pickle.load(f)


def generate_state_predictions(model, unemployment_rate, industry_code):
    """
    Generate predictions for each state based on the given model and encoders.

    Parameters:
    model (torch.nn.Module): The trained model for prediction.
    state_encoder (OneHotEncoder): Encoder for state names.
    industry_encoder (OneHotEncoder): Encoder for industry codes.
    state_dict (dict): Dictionary mapping state indices to state abbreviations.
    naics_codes (list): List of NAICS industry codes.
    industry_code (str): The NAICS code for the industry of interest.

    Returns:
    pd.DataFrame: DataFrame containing state abbreviations and their corresponding predictions.
    """
    # Given our parameters, we can now generate a prediction for each state
    all_inputs = [torch.tensor(np.concatenate([
                    np.array([unemployment_rate]).reshape(-1, 1),
                    state_encoder.transform([[state]]).reshape(-1, 1),
                    industry_encoder.transform([[industry_code]]).reshape(-1, 1)
                ], axis=0), dtype=torch.float32).T for state in state_encoder.categories_[0]]

    # Generate predictions using the model
    predictions = np.array([model.predict(input) for input in all_inputs]).squeeze()

    # Create a dataframe with the predictions and state abbreviations
    df_predictions = pd.DataFrame({
        'State': [state_dict[state] for state in state_encoder.categories_[0]],
        'Prediction': predictions
    })

    return df_predictions





# unemployment_rate = float(sys.argv[1])
# industry_code = sys.argv[2]
# predictions = generate_state_predictions(model, unemployment_rate, industry_code)
# predictions_json = predictions.to_dict(orient="records")
#print(json.dumps(predictions_json))
