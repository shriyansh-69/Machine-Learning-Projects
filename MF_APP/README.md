# Mutual Fund Price Prediction Web App

A Streamlit web app that predicts the future price/NAV of mutual funds and allows users to check historical prices.

🌐 **Live Demo:** [Mutual Fund Price Prediction Model](https://mutualfundpricepredictionmodel.streamlit.app/)

---

## Features

- Fetches current NAV of any mutual fund with currency details.
- Retrieves historical NAV for a specific date with nearest-date handling.
- Allows custom analysis on recent or selected date ranges. 
- Calculates :
   - Buy & Sell price
   - Profit / Loss
   - ROI for a selected period
- Interactive visualizations of historical and predicted prices.
- Optional moving average overlays and data.
-  **LSTM-based future price prediction**  With:
     - Trend plots
     - Buy / Hold / Sell signals
- Downloadable:
   - Charts
   - Historical & predicted data



---

--- 

## Model And Approach
- Historical NAV data is collected using yfinance
- Data is preprocessed and scaled
- An LSTM neural network is used to capture temporal patterns
- Model performance is evaluated using train-test split
- Predictions are generated for future time steps and visualized



--- 

--- 

## Screenshots

![Home Page](screenshots/home.png)
![Prediction Result](screenshots/prediction.png)
![Historical Price Checker](screenshots/historical.png)



---

## Technologies Used

- Frontend & Deployment: Streamlit
- Data Processing: Pandas, NumPy
- Data Source: yfinance
- Machine Learning: Scikit-learn
- Deep Learning: TensorFlow (LSTM)
- Database: MySQL
- Visualization: Matplotlib

---

---

## Result

- The application successfully provides end-to-end mutual fund analysis, combining data retrieval, ML-based prediction, visualization, and deployment in a user-friendly interface.

---
