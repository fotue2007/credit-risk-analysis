import streamlit 
import joblib
import pandas as pd 
import numpy as np
def charger_model():
    model = joblib.load("model_credit.joblib")
    return model
def charger_scaler():
    scaler = joblib.load("scaler_credit.joblib")
    return model 
    

