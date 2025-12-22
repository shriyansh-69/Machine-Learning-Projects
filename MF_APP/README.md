# Mutual Fund Price Prediction Web App

A Streamlit web app that predicts the future price/NAV of mutual funds and allows users to check historical prices.

🌐 **Live Demo:** [Mutual Fund Price Prediction Model](https://mutualfundpricepredictionmodel.streamlit.app/)

---

## Features

- Get the **current NAV** of any mutual fund with currency info.
- Get the **price/NAV on a specific date**, with nearest-date handling.
- Analyze **Recent Day's**  And Also **Selected Date** To Do  Operation's On The Selected Column's. 
- Calculates the buy and sell prices, **determines profit or loss**, **computes ROI**, and returns a summary for a selected date range and price type.
- **Visualizations** of predicted prices.
- Allows users to **plot selected price types with optional moving averages**** for a ticker over a date range and download the chart and data.
- Provides an **LSTM-based future price prediction** for a ticker, displays signals (Buy/Hold/Sell), plots predictions, and allows analysis and download of predicted data.


---

## Screenshots

![Home Page](screenshots/home.png)
![Prediction Result](screenshots/prediction.png)
![Historical Price Checker](screenshots/historical.png)



---

## Technologies Used

- streamlit
- pandas
- numpy
- yfinance
- tensorflow
- scikit-learn
- mysql-connector-python
- matplotlib  
