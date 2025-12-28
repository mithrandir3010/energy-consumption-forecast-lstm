# LSTM-Based Energy Consumption Forecasting

This project focuses on forecasting hourly electricity consumption using a Long Short-Term Memory (LSTM) based deep learning model. The study is conducted on Turkey’s hourly electricity consumption data covering the period from 2018 to 2023.

The objective of the project is to model the temporal dependencies in energy consumption data and to predict the next-hour electricity demand based on the previous 24 hours of consumption.

---

## Project Motivation

Electricity consumption is a time-dependent process influenced by daily, weekly, and seasonal patterns. Accurate short-term forecasting is crucial for energy planning, load balancing, and efficient resource management.

Traditional methods often rely on simple assumptions, such as using the previous time step as the prediction. In this project, a deep learning–based approach is adopted to capture complex temporal patterns and long-term dependencies in the data.

---

## Dataset

- **Source:** Hourly electricity consumption data for Turkey  
- **Time Range:** 2018 – 2023  
- **Frequency:** Hourly  
- **Target Variable:** Electricity consumption (MWh)

The dataset is treated as a univariate time series, focusing solely on electricity consumption values. Time information is used for ordering the data chronologically.

---

## Methodology

1. **Data Preprocessing**
   - Time values are parsed and sorted chronologically.
   - Missing or invalid entries are removed.
   - The target variable is converted to numerical format.
   - Min–Max normalization is applied based on training data.

2. **Sequence Construction**
   - A sliding window approach is used.
   - Input: Previous 24 hourly consumption values  
   - Output: Next-hour consumption value

3. **Model Architecture**
   - LSTM (Long Short-Term Memory) network
   - Input size: 1  
   - Hidden units: 64  
   - Output: Single-step forecast

4. **Training Strategy**
   - Optimizer: Adam  
   - Loss function: Mean Squared Error (MSE)  
   - Early stopping based on validation loss

---

## Baseline Comparison

To evaluate the effectiveness of the LSTM model, a **Naive baseline** is implemented:

- **Naive approach:** The next value is predicted as the last observed value.

### Performance Comparison (Test Set)

| Model | MAE | RMSE | MAPE (%) |
|------|-----|------|----------|
| Naive Baseline | 1152.55 | 1527.32 | 3.14 |
| LSTM (24 → 1) | 406.22 | 560.22 | 1.12 |

The results clearly demonstrate that the LSTM model significantly outperforms the naive baseline across all evaluation metrics.

---

## Project Structure

energy-consumption-forecast/
│
├── notebook/
│ └── energy_forecast.ipynb
│
├── src/
│ ├── model.py # LSTM model definition
│ ├── train.py # Training and evaluation script
│ └── serve.py # Gradio-based demo application
│
├── artifacts/
│ ├── lstm_model.pth
│ ├── training_history.csv
│ ├── results_table.csv
│ └── scaler.json
│
├── README.md
├── requirements.txt
└── .gitignore


---

## Demo Application

An interactive demo is provided using **Gradio**, allowing users to input the last 24 hourly consumption values and obtain a prediction for the next hour.

The demo demonstrates how the trained model can be used in a real-world scenario.

---

## Technologies Used

- Python
- PyTorch
- NumPy & Pandas
- Matplotlib
- Gradio
- Google Colab / VS Code

---

## Conclusion

This project demonstrates that LSTM-based deep learning models are effective for short-term electricity consumption forecasting. By learning temporal dependencies from historical data, the proposed model achieves significantly lower prediction errors compared to a simple baseline approach.

Future work may include incorporating additional features such as weather conditions, calendar effects, or extending the model to multi-step forecasting.

---
