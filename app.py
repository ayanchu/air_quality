"""Streamlit interface for Project 25 Air Quality Estimator.
Run locally after extracting air_quality_artifacts.zip:
  pip install streamlit pandas numpy scikit-learn tensorflow joblib
  streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).parent / 'air_quality_artifacts'
if not ARTIFACTS.exists():
    # Also allow extracted ZIP contents placed directly alongside app.py.
    ARTIFACTS = Path(__file__).parent

st.set_page_config(page_title='Air Quality Estimator', page_icon='🌫️')
st.title('🌫️ Air Quality Estimator')
st.caption('Educational same-hour NO₂ estimate (µg/m³) — not a medical or regulatory measurement')

@st.cache_resource
def load_artifacts():
    config = json.loads((ARTIFACTS/'config.json').read_text(encoding='utf-8'))
    models = {'Linear Regression':joblib.load(ARTIFACTS/'linear.joblib'),
              'Polynomial Regression':joblib.load(ARTIFACTS/'polynomial.joblib')}
    preprocessing = joblib.load(ARTIFACTS/'nn_preprocessor.joblib')
    return config, models, preprocessing

try:
    config, models, preprocessing = load_artifacts()
except FileNotFoundError:
    st.error('Model files missing. Extract air_quality_artifacts.zip beside app.py, or to an air_quality_artifacts/ folder.')
    st.stop()

model_name = st.selectbox('Model', ['Linear Regression','Polynomial Regression','Neural Network'],
                          index=['Linear Regression','Polynomial Regression','Neural Network'].index(config['default_model']))
st.subheader('Weather and measurements')
col1,col2,col3 = st.columns(3)
with col1:
    temp=st.number_input('Temperature (°C)',value=20.0,min_value=-30.0,max_value=60.0)
with col2:
    rh=st.number_input('Relative humidity (%)',value=50.0,min_value=0.0,max_value=100.0)
with col3:
    ah=st.number_input('Absolute humidity',value=1.0,min_value=0.0)
st.caption('AH uses the dataset’s measurement scale; values should come from a matching sensor.')

sensor_names=['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']
initial=[1000.,900.,800.,1100.,1000.]
values={}
with st.expander('Chemical sensor readings (all five required)',expanded=True):
    for name,default in zip(sensor_names,initial):
        values[name]=st.number_input(name,min_value=0.0,value=default,step=10.0)
measurement_date=st.date_input('Measurement date')
measurement_time=st.time_input('Measurement time')

if st.button('Estimate NO₂ concentration',type='primary'):
    dt=pd.Timestamp.combine(measurement_date,measurement_time)
    hour,weekday,month=dt.hour,dt.dayofweek,dt.month
    x={**values,'T':temp,'RH':rh,'AH':ah,
       'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
       'weekday_sin':np.sin(2*np.pi*weekday/7),'weekday_cos':np.cos(2*np.pi*weekday/7),
       'month_sin':np.sin(2*np.pi*(month-1)/12),'month_cos':np.cos(2*np.pi*(month-1)/12)}
    frame=pd.DataFrame([x],columns=config['features'])
    if model_name == 'Neural Network':
        from tensorflow import keras
        @st.cache_resource
        def get_nn():
            return keras.models.load_model(ARTIFACTS/'neural_network.keras',compile=False)
        estimated=float(get_nn().predict(preprocessing.transform(frame).astype('float32'),verbose=0).ravel()[0])
    else:
        estimated=float(models[model_name].predict(frame)[0])
    st.metric('Estimated NO₂',f'{max(0,estimated):.1f} µg/m³')
    st.info('Do not use this result for environmental compliance or health decisions. Historical calibration data may not generalize to modern sensors or locations.')

if (ARTIFACTS/'cv_metrics.csv').exists():
    st.subheader('Cross-validation comparison')
    st.dataframe(pd.read_csv(ARTIFACTS/'cv_metrics.csv').round(3),hide_index=True)
if (ARTIFACTS/'test_metrics.csv').exists():
    st.subheader('Holdout model comparison')
    st.dataframe(pd.read_csv(ARTIFACTS/'test_metrics.csv').round(3),hide_index=True)

for filename,title in [('predictions_residuals.png','Predicted vs. actual / residuals'),('error_by_hour_season.png','Error by time of day and season')]:
    if (ARTIFACTS/filename).exists():
        st.subheader(title)
        st.image(str(ARTIFACTS/filename),use_container_width=True)
