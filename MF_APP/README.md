# Mutual Fund Price Prediction Web App

A **Streamlit-based** interactive web application that **predicts** future mutual fund NAVs using **historical data** and **deep learning**, and allows users to **analyze** past performance, ROI, and trading signals.

🌐 **Live Demo:** [Mutual Fund Price Prediction Model](https://mutualfundpricepredictionmodel.streamlit.app/)

---

---

## Features

- Fetches **current NAV** of any **mutual fund** with currency details.
- Retrieves **historical NAV** for a specific date with nearest-date handling.
- Allows **custom analysis** on recent or selected date ranges. 
- **Calculates** :
   - Buy & Sell price
   - Profit / Loss
   - ROI for a selected period
- **visualizations** of historical and predicted prices.
- Optional **moving average** overlays and data.
-  **LSTM-based future price prediction**  With:
     - Trend plots
     - Buy / Hold / Sell signals
- **Downloadable**:
   - Charts
   - Historical & predicted data

---

---

## Problem Statement
**Investors** often struggle to **analyze** mutual fund performance and **predict** future NAV trends.
This project aims to provide a **simple**, **interactive platform** to:
- **Analyze** historical NAV data
- **Visualize** trends
- Predict **future prices** using LSTM
- Assist **decision-making** with Buy/Hold/Sell signals


--- 

---

## Solution Overview
This project **analyzes** historical mutual fund NAV data and **predicts** future prices using an LSTM-based **deep learning model**. It provides interactive **visualizations**,  **ROI calculations**, and **buy/sell** signals through a user-friendly **Streamlit web application**.



---

--- 

## Model And Approach
- Historical NAV data is collected using **yfinance**
- **Data** is **preprocessed** and **scaled**
- An **LSTM** neural network is used to capture temporal **patterns**
- Model **performance** is **evaluated** using **train-test split**
- **Predictions** are **generated** for future time **steps** and **visualized**



--- 

--- 

## Screenshots

![Home Page](screenshots/home.png)
![Prediction Result](screenshots/prediction.png)
![Historical Price Checker](screenshots/historical.png)



---

---
## Technologies Used

- Frontend & Deployment: **Streamlit**
- Data Processing: **Pandas**, **NumPy**
- Data Source: **yfinance**
- Machine Learning: **Scikit-learn**
- Deep Learning: **TensorFlow (LSTM)**
- Database: **MySQL**
- Visualization: **Matplotlib**



---

---

## Result

- The application **successfully** provides end-to-end **mutual fund** analysis, combining data retrieval, ML-based prediction, visualization, and **deployment** in a user-friendly interface.

---
