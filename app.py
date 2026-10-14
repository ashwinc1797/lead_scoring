import streamlit as st
import pandas as pd
import numpy as np
import joblib
import struct
import xgboost as xgb
from pathlib import Path

# Configurations 
st.set_page_config(page_title="AI Lead Scoring System",page_icon="📊🤖", layout="wide")

st.title("AI Lead Scoring System")

st.write( "Upload your lead data to get the lead scores using our AI model")
st.write("Please ensure your data is in CSV format and contains the necessary features for scoring.")

# load trained model

model_path=Path(__file__).resolve().parent/"xgb_pipeline.pkl"
if not model_path.exists():
    st.error("Trained model not found. please copy the model file to this folder and try again.")
    st.stop()

# A pickled XGBoost model stores {"Config": ..., "Model": ...}. The Config holds the training
# random-number state, whose text format differs between Linux and Windows, so a model pickled
# in Colab fails here with "input stream corrupted". That state only matters for training, so
# if the normal load fails, load just the "Model" section with XGBoost's portable loader.
_xgb_setstate = xgb.Booster.__setstate__
_MODEL_KEY = b"L" + struct.pack(">q", 5) + b"Model"  # the "Model" key in XGBoost's binary JSON

def _portable_booster_setstate(self, state):
    try:
        _xgb_setstate(self, dict(state))
    except xgb.core.XGBoostError:
        buf = bytes(state["handle"])
        booster = xgb.Booster()
        booster.load_model(bytearray(buf[buf.index(_MODEL_KEY) + len(_MODEL_KEY):-1]))
        handle, booster.handle = booster.handle, None  # take ownership of the loaded model
        self.__dict__.update(state)
        self.handle = handle

@st.cache_resource
def load_model():
    xgb.Booster.__setstate__ = _portable_booster_setstate
    try:
        return joblib.load(model_path)
    finally:
        xgb.Booster.__setstate__ = _xgb_setstate

model = load_model()

# Upload lead data

upload_file=st.file_uploader("Upload your lead data (CSV format)", type=["csv"])

if upload_file is not None:
    #read csv
    leads=pd.read_csv(upload_file)
    st.subheader("Uploaded Lead Data")
    st.dataframe(leads.head())

    # predict
    if st.button("Predict Lead Scores"):
        prediction = model.predict(leads)
        probabilities = model.predict_proba(leads)[:, 1]

         # prediction results
        results=leads.copy()
        results["Conversion_probality"]=(probabilities*100).round(2)
        results["Prediction"]=prediction 

        # assign lead priority based on probability
        def assign_priority(prob):
            if prob >= 80:
                return "High"
            elif prob >= 50:
                return "Medium"
            else:
                return "Low"       
        results["Lead_Priority"]=results["Conversion_probality"].apply(assign_priority)
        st.subheader("Prediction Results")
        st.dataframe(results)

        # Download results
        csv_data=results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Results as CSV",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv"
        )