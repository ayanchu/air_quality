# Air Quality Estimator — Project 25

## Project Overview

This project estimates NO₂ concentration using meteorological measurements and chemical sensor readings.

It compares three machine learning approaches:

- Linear Regression
- Polynomial Regression
- Fully Connected Neural Network

## Dataset

UCI Air Quality Dataset, downloaded using KaggleHub.

## Technologies

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Streamlit.

## Evaluation

The models are evaluated using MAE, RMSE, R² and K-fold cross-validation.

The analysis includes predicted vs. actual plots, residual plots, error heatmaps and error analysis by time of day and season.

## Running the Notebook

Upload `Air_Quality_Estimator_Colab.ipynb` to Google Colab and execute all cells in order.

## Running the Interactive Application

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Streamlit:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Application

Users can select a regression model, enter weather conditions, chemical sensor readings, date and time, and obtain an estimated NO₂ concentration in µg/m³.

## Limitations

The dataset contains historical sensor measurements. Predictions are intended for educational demonstration and should not be used as official environmental or health measurements.
