
import joblib
import pandas as pd 
import numpy as np
import streamlit as st
@st.cache_resource
def charger_modeles():
 
  model = joblib.load("model_credit.joblib")
  scaler = joblib.load("scaler_credit.joblib")
  return model, scaler

def predire_defaut(df_client: pd.DataFrame):
  model, scaler = charger_modeles()
  donnees_scaled = scaler.transform(df_client)
  proba = model.predict_proba(donnees_scaled)[0][1]
  return proba
