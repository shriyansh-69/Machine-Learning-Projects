import streamlit as st
import io

# Data handling
import pandas as pd
import numpy as np

# Word Handling
import re

# Date and time handling
from datetime import datetime

# Optional: Fetch financial data
import yfinance as yf


# Database Connector
import mysql.connector


# Machine Learning Library
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Deep Learning Library
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense,Input, Reshape # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping  # pyright: ignore[reportMissingImports]S
import tensorflow as tf

# Data Visulization
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Mutual_Fund_Prediction_Model",
    page_icon= "📚",
    layout = "wide" 

)


# ---------------------------------------------------------------Block-1------------------------------------------------------------------

def get_current_price(fund_ticker):
    try:
        fund = yf.Ticker(fund_ticker)
        data = fund.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            return "No Data Is Available For This Ticker"
    except Exception as e:
        return f"Error: {e}"

# -------- EXPANDER ----------
with st.expander("📌 Mutual Fund / Current Price Checker", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    ticker = st.text_input("Enter The Mutual Fund Name:")

    if st.button("Get Price"):
        if ticker:
            price = get_current_price(ticker.upper())
            st.write(
                f"The Current Price/NAV of **{ticker.upper()}** is: **{price}**"
            )
        else:
            st.warning("That's an invalid name")


# ---------------------------------------------------------------Block-2------------------------------------------------------------------

with st.expander("📅 Get Price on Specific Date", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing
    
    def get_price_on_date(fund_name, date_str):
        try:
            fund = yf.Ticker(fund_name)        
            data = fund.history(period="max")

            if data.empty:
                return "No Data Available"

            data.index = data.index.strftime("%Y-%m-%d")

            if date_str in data.index:
                return data.loc[date_str]['Close']

            previous_dates = [d for d in data.index if d <= date_str]
            if previous_dates:
                nearest = previous_dates[-1]
                return f"{data.loc[nearest]['Close']} (Nearest available date: {nearest})"

            return "No Data Available For This Date"

        except Exception as e:
            return f"Error: {e}"

    st.write("### Get Price on Specific Date")

    fund_name = st.text_input(
        "Which Fund To Want To TakeOut Price Of :- "
    ).upper()

    date_input = st.date_input("Select the Date:")

    if st.button("Get Price on Date"):
        if not fund_name:
            st.error("Please enter a valid fund name")
        else:
            date_str = date_input.strftime("%Y-%m-%d")
            price = get_price_on_date(fund_name, date_str)
            st.write(
                f"Price/NAV of **{fund_name}** on **{date_str}** is: **{price}**"
            )

#------------------------------------Block-3------------------------------------------------------------------

def Information_Of_Selected_Day(Fund_name, price_type, n_day, operation=None):
    fund = yf.Ticker(Fund_name)
    data = fund.history(period='max')

    if data.empty:
        return f"Ticker '{Fund_name}' is invalid or delisted."

    data['Average_Price'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    all_columns = ['Open', 'High', 'Low', 'Close', 'Average_Price']

    if isinstance(price_type, str):
        price_types_to_check = [pt.strip() for pt in price_type.split(',')]
    elif isinstance(price_type, list):
        price_types_to_check = price_type
    else:
        return "Invalid price_type. Must be string or list of strings."

    final_price_types = []
    for pt in price_types_to_check:
        pt_lower = pt.lower()
        if pt_lower == 'all':
            final_price_types = all_columns
            break
        elif pt_lower == 'average_price':
            final_price_types.append('Average_Price')
        elif pt.capitalize() in all_columns:
            final_price_types.append(pt.capitalize())
        else:
            return f"Invalid column '{pt}'. Available columns: {all_columns}"

    price_types_to_check = list(dict.fromkeys(final_price_types))

   
    if operation is None or 'All' in operation or 'all' in operation:
        operations_to_check = ['lowest', 'highest', 'average']
    else:
        operations_to_check = []
        for op in operation:
           op_lower = op.lower()
           if op_lower in ['lowest', 'low', 'smallest', 'min']:
              operations_to_check.append('lowest')
           elif op_lower in ['highest', 'high', 'max']:
                operations_to_check.append('highest')
           elif op_lower in ['average', 'mean']:
                 operations_to_check.append('average')
           else:
               return f"Invalid operation '{op}'. Choose from Lowest/Highest/Average."

    
    operations_to_check = list(dict.fromkeys(operations_to_check))


    if n_day is None:
        selected_range = slice(None)
    else:
        selected_range = slice(-n_day, None)

    results_dict = {}
    for pt in price_types_to_check:
        selected_data = data[pt][selected_range]
        results_dict[pt] = {}
        if 'lowest' in operations_to_check:
            results_dict[pt]['Lowest'] = selected_data.min()
        if 'highest' in operations_to_check:
            results_dict[pt]['Highest'] = selected_data.max()
        if 'average' in operations_to_check:
            results_dict[pt]['Average'] = selected_data.mean()

    df_results = pd.DataFrame(results_dict).T
    return df_results


# -------------------- Plot Function --------------------
def plot_price_type(price_type, Fund_name, n_day):
    fund = yf.Ticker(Fund_name)
    data = fund.history(period='max')

    data['Average_Price'] = data[['Open','High','Low','Close']].mean(axis=1)

    if n_day is not None:
        data = data.tail(n_day)

    data.index = pd.to_datetime(data.index)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data[price_type], label=price_type, color='blue')
    ax.set_title(f"{Fund_name} — {price_type} Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    return fig

# --------------------------------------------------------------- Interface ------------------------------------------------------------------

with st.expander("📊 Mutual Fund Analyzer For Recent Day's ", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    # ---------------- INPUTS ----------------
    Fund_name = st.text_input(
        "Enter Ticker/Fund Name:",
        key="fund"
    ).upper()

    price_type_options = ['Open', 'Close', 'High', 'Low', 'Average_Price',"All"]

    price_type_selected = st.multiselect(
        "Select Price Type(s) for Analysis:",
        options=price_type_options,
        default=["Close"],
        key="analysis"
    )

    # Convert selection To Actual Columns For Anlaysis
    if "All" in price_type_selected:
        price_type_selected = ['Open', 'Close', 'High', 'Low', 'Average_Price']

    n_day_input = st.text_input(
        "Enter Number of Days (or 'all'):",
        value="all"
    )

    operation_input = st.multiselect(
        "Select Operation:",
        ["All", "Lowest", "Highest", "Average"]
    )

    # ---------------- ANALYSIS ----------------
    if st.button("📊 Get Analysis"):
        if not Fund_name:
            st.error("Please enter a ticker.")
        else:
            # Convert days safely
            if n_day_input.lower() == "all":
                n_day = None
            else:
                try:
                    n_day = int(n_day_input)
                except ValueError:
                    st.error("Please enter a valid number for days.")
                    st.stop()

            operation = None if "All" in operation_input else operation_input

            result = Information_Of_Selected_Day(
                Fund_name,
                price_type_selected,
                n_day,
                operation
            )

            if isinstance(result, pd.DataFrame):
                st.subheader("📋 Analysis Result")
                st.dataframe(result)

                csv = result.to_csv().encode()
                st.download_button(
                    "📥 Download Analysis as CSV",
                    data=csv,
                    file_name=f"{Fund_name}_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.error(result)

    st.divider()

    # ---------------- PLOT ----------------
    plot_price_type_selected = st.selectbox(
        "Select ONE Price Type to Plot:",
        options=price_type_options,
        key="plot_only"
    )

    if st.button("📈 Show Plot"):
        # Convert days safely
        if n_day_input.lower() == "all":
            n_day = None
        else:
            try:
                n_day = int(n_day_input)
            except ValueError:
                st.error("Please enter a valid number for days.")
                st.stop()

        if not Fund_name:
            st.error("Please enter a ticker to plot.")
            st.stop()

        fig = plot_price_type(
            plot_price_type_selected,
            Fund_name,
            n_day
        )

        st.pyplot(fig)

        # ---------------- Download Options ----------------
        with st.expander("💾 Download Options", expanded=False):
            # PNG
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png")
            buf_png.seek(0)
            st.download_button(
                "📥 Download Plot (PNG)",
                data=buf_png,
                file_name=f"{Fund_name}_{plot_price_type_selected}_plot.png",
                mime="image/png"
            )

            # PDF
            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf")
            buf_pdf.seek(0)
            st.download_button(
                "🗂️ Download Plot (PDF)",
                data=buf_pdf,
                file_name=f"{Fund_name}_{plot_price_type_selected}_plot.pdf",
                mime="application/pdf"
            )

            # CSV for plot data
            data = yf.Ticker(Fund_name).history(period='max')
            data['Average_Price'] = data[['Open','High','Low','Close']].mean(axis=1)
            if n_day is not None:
                data = data.tail(n_day)
            st.download_button(
                "⬇️ Download Plot Data (CSV)",
                data=data[[plot_price_type_selected]].to_csv().encode(),
                file_name=f"{Fund_name}_{plot_price_type_selected}_data.csv",
                mime="text/csv"
            )

         # ---------------- Raw Data (Last n Days or All) ----------------
        with st.expander("### 📄 Raw Data (Selected Days)"):
            if data.empty:
                st.warning("No data available for the selected date range.")
            else:
                st.dataframe(data, use_container_width=True)
                st.download_button(
                    "⬇️ Download Raw Data (CSV)",
                    data=data.to_csv(index=True).encode("utf-8"),
                    file_name=f"{Fund_name}_raw_data.csv",
                    mime="text/csv"
                )


# ---------------------------------------------------------------Block-4------------------------------------------------------------------


def Operation_On_Selected_Day(Fund_name, price_type, start_date, end_date, operation=None):
    fund = yf.Ticker(Fund_name)
    data = fund.history(start=start_date, end=end_date)

    if data.empty:
        return f"Ticker '{Fund_name}' is invalid, delisted, or no data for this date range."

    # Create Average_Price column
    data['Average_Price'] = data[['Open','High','Low','Close']].mean(axis=1)

    # --------- Handle price_type ---------
    if isinstance(price_type, list):
        if any(pt.lower() == "all" for pt in price_type):
            price_types_to_check = ['Open', 'High', 'Low', 'Close', 'Average_Price']
        else:
            price_types_to_check = []
            for pt in price_type:
                if pt.lower() == "average":
                    price_types_to_check.append("Average_Price")
                elif pt in ['Open', 'High', 'Low', 'Close']:
                    price_types_to_check.append(pt)
                else:
                    return f"Price type '{pt}' not found. Available: Open, High, Low, Close, Average"
    else:
        if price_type.lower() == "all":
            price_types_to_check = ['Open', 'High', 'Low', 'Close', 'Average_Price']
        elif price_type.lower() == "average":
            price_types_to_check = ["Average_Price"]
        else:
            price_types_to_check = [price_type]

    # --------- Handle operation ---------
    if operation is None:
        operations_to_check = ['lowest', 'highest', 'average']
    else:
        if isinstance(operation, str):
            ops = [operation.lower()]
        else:
            ops = [op.lower() for op in operation]

        operations_to_check = []
        for op in ops:
            if op in ['lowest', 'low', 'smallest', 'min']:
                operations_to_check.append('lowest')
            elif op in ['highest', 'high', 'max']:
                operations_to_check.append('highest')
            elif op in ['average', 'mean']:
                operations_to_check.append('average')
            elif op == "all":
                operations_to_check.extend(['lowest', 'highest', 'average'])
            else:
                return "Invalid operation! Choose from: lowest/highest/average"

    # --------- Build results ---------
    results_dict = {}
    for pt in price_types_to_check:
        selected_data = data[pt]
        results_dict[pt] = {}
        if 'lowest' in operations_to_check:
            results_dict[pt]['Lowest'] = selected_data.min()
        if 'highest' in operations_to_check:
            results_dict[pt]['Highest'] = selected_data.max()
        if 'average' in operations_to_check:
            results_dict[pt]['Average'] = selected_data.mean()

    df_results = pd.DataFrame(results_dict).T
    return df_results

# -------------------- PLOT FUNCTION --------------------
def plot_selected_price_type(new_price_type, Fund_name, start_date, end_date):
    fund = yf.Ticker(Fund_name)
    data = fund.history(start=start_date, end=end_date)

    if data.empty:
        st.error(f"No data to plot for {Fund_name} in this date range.")
        return

    data['Average_Price'] = data[['Open','High','Low','Close']].mean(axis=1)

    if new_price_type.lower() == 'average':
        new_price_type = 'Average_Price'

    if new_price_type not in data.columns:
        st.error(f"Price type '{new_price_type}' not found in data.")
        return

    data.index = pd.to_datetime(data.index)

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(data.index, data[new_price_type], label=new_price_type, color='blue')
    ax.set_title(f"{Fund_name} — {new_price_type} Price from {start_date} to {end_date}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    return fig,data


#---------------------------------------Interface----------------------------------


with st.expander("📈 Mutual Fund Analyzer For Selected Date's", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    # ---------------- INPUTS ----------------
    Fund_name = st.text_input("Enter Ticker/Fund Name:", key="fund_op").upper()

    price_type = st.multiselect(
        "Select Price Type:",
        ["Open", "Close", "High", "Low", "Average", "All"],
        default=["Close"],
        key="price_type_op"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date:", key="start_date_op")
    with col2:
        end_date = st.date_input("End Date:", key="end_date_op")

    operation_input = st.multiselect(
        "Select Operation:",
        ["Lowest", "Highest", "Average", "All"],
        key="operation_op"
    )
    operation = None if any(op.lower() == "all" for op in operation_input) else operation_input

    st.divider()

    # ---------------- ANALYSIS ----------------
    if st.button("📊 Get Results", key="btn_results_op"):
        if not Fund_name:
            st.error("Please enter a valid ticker.")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            result = Operation_On_Selected_Day(
                Fund_name,
                price_type,
                start_str,
                end_str,
                operation
            )

            if isinstance(result, pd.DataFrame):
                st.success(f"Results for {Fund_name} ({start_str} → {end_str})")
                st.dataframe(result)

                csv = result.to_csv().encode()
                st.download_button(
                    "📥 Download Results as CSV",
                    data=csv,
                    file_name=f"{Fund_name}_{start_str}_to_{end_str}_results.csv",
                    mime="text/csv"
                )
            else:
                st.error(result)

    st.divider()

    # ---------------- PLOT ----------------
    plot_price_type_choice = st.selectbox(
        "Select Price Type to Plot:",
        ["Open", "Close", "High", "Low", "Average"],
        key="plot_price_type"
    )

    if st.button("📈 Show Plot", key="btn_plot"):
        if not Fund_name:
            st.error("Please enter a valid ticker.")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            plot_column = "Average_Price" if plot_price_type_choice.lower() == "average" else plot_price_type_choice

            fig, data_for_download = plot_selected_price_type(
                plot_column,
                Fund_name,
                start_str,
                end_str
            )

#------------------------------ Download Option For Plot------------------------------------

            with st.expander("💾 Download Plot Options", expanded=False):

                if fig is not None:
                    # PNG
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png")
                    buf_png.seek(0)
                    st.download_button(
                        "📥 Download Plot (PNG)",
                        data=buf_png,
                        file_name=f"{Fund_name}_{plot_price_type_choice}_plot.png",
                        mime="image/png"
                    )

                    # PDF
                    buf_pdf = io.BytesIO()
                    fig.savefig(buf_pdf, format="pdf")
                    buf_pdf.seek(0)
                    st.download_button(
                        "🗂️ Download Plot (PDF)",
                        data=buf_pdf,
                        file_name=f"{Fund_name}_{plot_price_type_choice}_plot.pdf",
                        mime="application/pdf"
                    )

                    # CSV
                    csv_data = data_for_download[[plot_column]].to_csv().encode()
                    st.download_button(
                        "⬇️ Download Data (CSV)",
                        data=csv_data,
                        file_name=f"{Fund_name}_{plot_price_type_choice}_data.csv",
                        mime="text/csv"
                    )



             # ------------------- RAW DATA FOR SELECTED DAYS -------------------
            with st.expander("📄 Raw Data (Selected Days)", expanded=False):

                        if not Fund_name:
                            st.warning("Please enter a valid ticker.")
                        else:
                            # Fetch data
                            fund = yf.Ticker(Fund_name)
                            df_raw = fund.history(start=start_date, end=end_date)

                            if df_raw.empty:
                                st.warning("No data available for the selected date range.")
                            else:
                                # Add average price column
                                df_raw['Average_Price'] = df_raw[['Open','High','Low','Close']].mean(axis=1)
                                df_raw.drop(["Dividends","Stock Splits"],axis=1)

                                # Show data
                                st.dataframe(df_raw, use_container_width=True)

                                # Download button
                                raw_csv = df_raw.to_csv(index=True).encode("utf-8")
                                st.download_button(
                                    label="⬇️ Download Selected Days Data (CSV)",
                                    data=raw_csv,
                                    file_name=f"{Fund_name}_raw_data_{start_date}_{end_date}.csv",
                                    mime="text/csv"
                                )


    
# --------------------------------------------------------------- Block-5 ------------------------------------------------------------------


# ------------------- FUNCTIONS -------------------

def fetch_fund_data(Fund_name):    
    dat = yf.Ticker(Fund_name)
    FUND_name = dat.history(period='max')

    # Drop Dividends and Stock Splits only if they exist

    df_prepare = FUND_name.drop(columns=[c for c in ['Dividends','Stock Splits'] if c in FUND_name.columns])
    df_prepare['Average'] = df_prepare[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df_prepare.index = pd.to_datetime(df_prepare.index)
    df_prepare.index = df_prepare.index.tz_convert(None)  # remove tz info
    return df_prepare


def price_analysis(df_prepare, Buy_date, Sell_date, price_type='Close'):
    """Analyze buy-sell price, ROI, and profit/loss."""
    nearest_buy_date = df_prepare.index.asof(Buy_date)
    nearest_sell_date = df_prepare.index.asof(Sell_date)

    if pd.isna(nearest_buy_date):
        return "❌ No data available before Buy date."
    if pd.isna(nearest_sell_date):
        return "❌ No data available before Sell date."

    Buy_price = float(df_prepare.loc[nearest_buy_date, price_type])
    Sell_price = float(df_prepare.loc[nearest_sell_date, price_type])

    Result = Sell_price - Buy_price
    ROI = (Result / Buy_price) * 100

    status = "🟢 PROFIT" if Result > 0 else ("🔴 LOSS" if Result < 0 else "⚪ NO PROFIT / NO LOSS")

    return {
        "status": status,
        "Buy_date": nearest_buy_date,
        "Sell_date": nearest_sell_date,
        "Buy_price": Buy_price,
        "Sell_price": Sell_price,
        "Result": Result,
        "ROI": ROI
    }

def plot_price(df_prepare, buy_date, sell_date, price_type='Close'):
    """Plot price movement between buy and sell dates."""
    buy_nearest = df_prepare.index.asof(buy_date)
    sell_nearest = df_prepare.index.asof(sell_date)
    df_slice = df_prepare.loc[buy_nearest:sell_nearest]

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_slice.index, df_slice[price_type], label='NAV Price', color='blue')
    ax.scatter(buy_nearest, df_prepare.loc[buy_nearest, price_type], color='orange', s=80, label='Buy')
    ax.scatter(sell_nearest, df_prepare.loc[sell_nearest, price_type], color='green', s=80, label='Sell')
    ax.set_title("Price Analysis (BUY → SELL)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig

# ------------------- Interface -------------------

with st.expander("🧮 Profit Or Loss Analyzer", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    Fund_name = st.text_input(
        "Enter Fund / Stock Ticker (e.g., TSLA, ARKK):"
    ).upper()

    price_type = st.selectbox(
        "Select Price Type:",
        ["Open", "High", "Low", "Close", "Average"],
        index=3
    )

    col1, col2 = st.columns(2)
    with col1:
        Buy_Date = st.date_input("Select Buy Date")
    with col2:
        Sell_Date = st.date_input("Select Sell Date")

    # Convert Streamlit dates to pandas timestamps
    if Buy_Date and Sell_Date:
        Buy_Date_ts = pd.Timestamp(Buy_Date)
        Sell_Date_ts = pd.Timestamp(Sell_Date)

    st.divider()

    # ---------------- ANALYSIS ----------------
    if st.button("📊 Analyze", key="analyze_btn"):
        if not Fund_name:
            st.error("Please enter a valid ticker/fund name.")
        else:
            try:
                df_prepare = fetch_fund_data(Fund_name)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

            analysis = price_analysis(
                df_prepare,
                Buy_Date_ts,
                Sell_Date_ts,
                price_type
            )

            if isinstance(analysis, dict):
                st.success(analysis["status"])

                st.write(
                    f"Buy Date: {analysis['Buy_date'].date()} → "
                    f"Price: {analysis['Buy_price']:.2f}"
                )
                st.write(
                    f"Sell Date: {analysis['Sell_date'].date()} → "
                    f"Price: {analysis['Sell_price']:.2f}"
                )

                if analysis['Result'] >= 0:
                    st.success(f"Profit: {analysis['Result']:.2f} ↑")
                else:
                    st.error(f"Loss: {analysis['Result']:.2f} ↓")

                st.write(f"ROI: {analysis['ROI']:.2f}%")

                df_analysis = pd.DataFrame([analysis])
                df_analysis['Buy_date'] = df_analysis['Buy_date'].dt.date
                df_analysis['Sell_date'] = df_analysis['Sell_date'].dt.date

                # CSV download
                csv = df_analysis.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download Buy-Sell Analysis as CSV",
                    data=csv,
                    file_name=f"{Fund_name}_buy_sell_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.error(analysis)

    st.divider()

    # ---------------- PLOT ----------------
    if st.button("📈 Show Plot", key="plot_btn"):
        if not Fund_name:
            st.error("Please enter a valid ticker/fund name.")
        else:
            try:
                df_prepare = fetch_fund_data(Fund_name)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

            fig = plot_price(
                df_prepare,
                Buy_Date_ts,
                Sell_Date_ts,
                price_type
            )
            st.pyplot(fig)



            with st.expander("💾 Download Options", expanded=False):

                if fig is not None:

                    # PNG
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png")
                    buf_png.seek(0)
                    st.download_button(
                        "📥 Download Plot (PNG)",
                        data=buf_png,
                        file_name=f"{Fund_name}_{price_type}_plot.png",
                        mime="image/png"
                    )

                    # PDF
                    buf_pdf = io.BytesIO()
                    fig.savefig(buf_pdf, format="pdf")
                    buf_pdf.seek(0)
                    st.download_button(
                        "🗂️ Download Plot (PDF)",
                        data=buf_pdf,
                        file_name=f"{Fund_name}_{price_type}_plot.pdf",
                        mime="application/pdf"
                    )

                    # CSV
                    csv_data = df_prepare[[price_type]].to_csv().encode()
                    st.download_button(
                        "⬇️ Download Plot Data (CSV)",
                        data=csv_data,
                        file_name=f"{Fund_name}_{price_type}_data.csv",
                        mime="text/csv"
                    )


            # ---------------- FILTER RAW DATA BY DATE ----------------
            filtered_df = df_prepare.copy()

            if Buy_Date_ts:
                filtered_df = filtered_df[filtered_df.index >= Buy_Date_ts]
            if Sell_Date_ts:
                filtered_df = filtered_df[filtered_df.index <= Sell_Date_ts]


            # ---------------- SHOW FILTERED RAW DATA ----------------
            with st.expander("📄 Raw Data (Selected Days Only)"):

                if filtered_df.empty:
                    st.warning("No data available for the selected date range.")
                else:
                    st.dataframe(filtered_df, use_container_width=True)

                    # Download filtered raw data
                    raw_csv = filtered_df.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Selected Days Data (CSV)",
                        data=raw_csv,
                        file_name=f"{Fund_name}_raw_data_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
#-------------------------------------------------------------------------- Block - 6 -------------------------------------------------------------------------------------

with st.expander("💹 Graph Plotter On Selected Day's", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    # --- Inputs ---
    f_name = st.text_input(
        "Enter the Ticker (e.g., AAPL, MSFT):",
        ""
    ).upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date (optional)")
    with col2:
        end_date = st.date_input("End Date (optional)")

    # --- Fetch data function ---
    @st.cache_data
    def get_data(ticker):
        try:
            val = yf.Ticker(ticker)
            df = val.history(period='max')
            if df.empty:
                return None

            df = df.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
            df.columns = [col.lower() for col in df.columns]
            df.index = df.index.tz_localize(None)
            df['average'] = df[['open', 'high', 'low', 'close']].mean(axis=1)

            return df
        except Exception:
            return None

    # --- Processing ---
    if f_name:
        df_ready = get_data(f_name)

        if df_ready is None:
            st.error(f"No data found for ticker '{f_name}'")
        else:
            # Filter by date
            if start_date:
                df_ready = df_ready[df_ready.index >= pd.to_datetime(start_date)]
            if end_date:
                df_ready = df_ready[df_ready.index <= pd.to_datetime(end_date)]

            st.subheader("Available Price Types")
            st.write(df_ready.columns.tolist())

            price_type = st.selectbox(
                "Select Price Type",
                df_ready.columns.tolist(),
                index=df_ready.columns.get_loc("close")
            )

            ma_input = st.text_input(
                "Enter Moving Average Window(s) separated by commas (e.g., 7,14):",
                "7"
            )

            ma_window = []
            for x in ma_input.split(","):
                try:
                    window = int(x.strip())
                    if window > 0:
                        ma_window.append(window)
                except ValueError:
                    st.warning(f"Ignored invalid MA input: '{x}'")

            title = st.text_input(
                "Enter Plot Title (optional):",
                f"{price_type.capitalize()} Price Over Time"
            )

            # --- Plot ---
            st.subheader("📊 Price Chart")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(
                df_ready.index,
                df_ready[price_type],
                label=f"{price_type} price",
                color='blue'
            )

            for window in ma_window:
                if window < len(df_ready):
                    ma = df_ready[price_type].rolling(window=window).mean()
                    ax.plot(ma.index, ma.values, label=f"{window}-Day MA")
                else:
                    st.warning(
                        f"MA window {window} is larger than "
                        f"number of data points ({len(df_ready)}). Skipping."
                    )

            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # --- Download Options ---
            with st.expander("### 💾 Download Options"):
                if fig is not None:

                    # PNG
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png")
                    buf_png.seek(0)
                    st.download_button(
                        "📥 Download Plot (PNG)",
                        data=buf_png,
                        file_name=f"{f_name}_{price_type}_plot.png",
                        mime="image/png"
                    )

                    # PDF
                    buf_pdf = io.BytesIO()
                    fig.savefig(buf_pdf, format="pdf")
                    buf_pdf.seek(0)
                    st.download_button(
                        "🗂️ Download Plot (PDF)",
                        data=buf_pdf,
                        file_name=f"{f_name}_{price_type}_plot.pdf",
                        mime="application/pdf"
                    )

                    # CSV
                    csv_data = df_ready[[price_type]].to_csv().encode()
                    st.download_button(
                        "⬇️ Download Plot Data (CSV)",
                        data=csv_data,
                        file_name=f"{f_name}_{price_type}_data.csv",
                        mime="text/csv"
                    )


            # --- Raw Data ---
            with st.expander("📄 Show Raw Data"):
                st.dataframe(df_ready)


                csv_data = df_ready.to_csv(index=True).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Raw Data (CSV)",
                    data=csv_data,
                    file_name=f"{f_name}_raw_data.csv",
                    mime="text/csv"
                )


# --------------------------------------------------------------- Block-7 ------------------------------------------------------------------


# --- Expander for LSTM Prediction ---
with st.expander("🧠 NeuralTicker", expanded=False):

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # adds top spacing

    # Input Panel
    ticker = st.text_input("Yahoo Finance Ticker", value="INFY.NS")
    sequence_length = 60
    n_future = 30

    # 1️⃣ Database connection
    @st.cache_resource(show_spinner=True)
    def get_connection():        
        cfg = st.secrets["mysql"]
        try:
            conn = mysql.connector.connect(
                host=cfg["host"],
                port=int(cfg["port"]),
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
             )
            conn.autocommit = True
            return conn
        except mysql.connector.Error as e:
            st.error(f"❌ Could not connect to the database: {e}")
            st.stop() 

    # 2️⃣ Cache ticker-specific data
    @st.cache_data(show_spinner=True)
    def get_fund_data(ticker: str, period: str = "10y") -> pd.DataFrame:
        conn = get_connection()
        cursor = conn.cursor()

        # Safe table name
        safe_ticker = re.sub(r"[^a-zA-Z0-9_]", "_", ticker).lower()
        table_name = f"ticker_{safe_ticker}"

        # Create table if not exists
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Date DATE PRIMARY KEY,
                Open FLOAT,
                High FLOAT,
                Low FLOAT,
                Close FLOAT,
                Volume BIGINT,
                Dividends FLOAT,
                Stock_Splits FLOAT
            )
        """)

        # Get last stored date
        cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
        last_date = cursor.fetchone()[0]

        # Fetch missing data from Yahoo
        if last_date:
            start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
            df_api = yf.download(ticker, start=start, progress=False)
        else:
            df_api = yf.download(ticker, period=period, progress=False)

        if not df_api.empty:
            df_api.reset_index(inplace=True)
            df_api["Dividends"] = 0.0
            df_api["Stock_Splits"] = 0.0
            df_api["Volume"] = df_api["Volume"].fillna(0).astype(int)
            df_api[["Open", "High", "Low", "Close"]] = df_api[["Open", "High", "Low", "Close"]].fillna(0.0)

            rows = list(
                df_api[["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock_Splits"]]
                .itertuples(index=False, name=None)
            )

            cursor.executemany(
                f"""
                INSERT INTO {table_name}
                (Date, Open, High, Low, Close, Volume, Dividends, Stock_Splits)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    Open=VALUES(Open),
                    High=VALUES(High),
                    Low=VALUES(Low),
                    Close=VALUES(Close),
                    Volume=VALUES(Volume),
                    Dividends=VALUES(Dividends),
                    Stock_Splits=VALUES(Stock_Splits)
                """,
                rows
            )

        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY Date ASC", conn, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        cursor.close()
        return df[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock_Splits"]]

    # --- LSTM Helper Functions ---
    def create_sequences_single(data, seq_len, n_future):
        X, Y = [], []
        for i in range(len(data) - seq_len - n_future + 1):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+n_future])
        return np.array(X), np.array(Y)

    def Signal(current_price, future_price, buy=0.02, sell=-0.02):
        change = (future_price - current_price) / current_price
        if change > buy:
            return f"BUY 📈 (+{change*100:.2f}%)"
        elif change < sell:
            return f"SELL 📉 (-{abs(change)*100:.2f}%)"
        return f"HOLD 🤝 ({change*100:.2f}%)"

    # --- Run Prediction ---
    if st.button("🚀 Run Prediction"):

        with st.spinner("Fetching & processing data..."):
            df = get_fund_data(ticker)
            df['Average_Price'] = df[['Open','High','Low','Close']].mean(axis=1)

            df_close = df[['Close']].ffill().bfill()
            if len(df_close) < sequence_length + n_future + 1:
                st.error("Not enough data for training.")
                st.stop()

            scaler = MinMaxScaler()
            df_close_scaled = scaler.fit_transform(df_close)

            # Train/test split
            split_idx = int(len(df_close_scaled) * 0.8)
            X_train, Y_train = create_sequences_single(df_close_scaled[:split_idx], sequence_length, n_future)
            X_test, Y_test = create_sequences_single(df_close_scaled[split_idx:], sequence_length, n_future)

            # Build LSTM model
            model = Sequential([
                Input(shape=(sequence_length, 1)),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(n_future)
            ])
            model.compile(optimizer="adam", loss="mse")
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=30, batch_size=16, callbacks=[early_stop], verbose=0)

            # Predict future
            last_seq = df_close_scaled[-sequence_length:].reshape(1, sequence_length, 1)
            future_scaled = model.predict(last_seq)[0]
            future_close = scaler.inverse_transform(future_scaled.reshape(-1,1)).flatten()
            future_dates = pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=n_future)

            future_df = pd.DataFrame(index=future_dates)
            future_df['Close'] = future_close
            future_df['Open'] = future_df['Close'].shift(1).fillna(df['Close'].iloc[-1])
            future_df['High'] = future_df[['Open','Close']].max(axis=1)
            future_df['Low'] = future_df[['Open','Close']].min(axis=1)
            future_df['Average_Price'] = future_df[['Open','High','Low','Close']].mean(axis=1)

            current_price = df['Close'].iloc[-1]
            day30_price = future_df['Close'].iloc[-1]
            signal = Signal(current_price, day30_price)

            st.session_state['future_df'] = future_df

            st.subheader("📊 Prediction Result")
            st.metric("Current Price", f"{current_price:.2f}")
            st.metric("Day-30 Prediction", f"{day30_price:.2f}")
            st.success(signal)

            # Plot predicted close
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(future_df.index, future_df['Close'], marker='o')
            ax.set_title("Predicted Close Price")
            ax.grid()
            st.pyplot(fig)

            # --- Download Options ---
            with st.expander("### 💾 Download Options"):

                # Plot PNG
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png")
                buf_png.seek(0)
                st.download_button(
                    label="📥 Download Plot (PNG)",
                    data=buf_png,
                    file_name=f"{ticker}_Prediction.png",
                    mime='image/png'
                )

                # Plot PDF
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf")
                buf_pdf.seek(0)
                st.download_button(
                    label="🗂️ Download Plot (PDF)",
                    data=buf_pdf,
                    file_name=f"{ticker}_Prediction.pdf",
                    mime='application/pdf'
                )

                # CSV without index
                csv_simple = future_df.to_csv(index=False).encode()
                st.download_button(
                    label="⬇️ Download Predicted Prices (CSV, no index)",
                    data=csv_simple,
                    file_name=f"{ticker}_Prediction.csv",
                    mime='text/csv'
                )

            # Display DataFrame with dates
            future_df_to_download = future_df.copy().reset_index()
            future_df_to_download.rename(columns={'index': 'Date'}, inplace=True)

            st.subheader("📄 Predicted Data")
            st.dataframe(future_df_to_download)

            # CSV with dates
            csv_with_dates = future_df_to_download.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Download Predicted Prices (CSV with Dates)",
                data=csv_with_dates,
                file_name=f"{ticker}_Prediction.csv",
                mime='text/csv'
            )

# --- Analysis Section ---
with st.expander("📊 Column Operations on Predicted Data", expanded=False):
    if 'future_df' not in st.session_state:
        st.warning("No Data Available. Run the prediction first.")
    else:
        future_df = st.session_state['future_df']

        with st.form("analysis_form"):
            cols_options = list(future_df.columns) + ["All Columns"]
            selected_cols = st.multiselect("Select Columns:", options=cols_options, default=["All Columns"])
            operation = st.multiselect("Select Operation:", ["Lowest", "Highest", "Average", "All"], default=["All"])
            submitted = st.form_submit_button("Compute Summary")

        if submitted:
            cols_to_use = future_df.columns if "All Columns" in selected_cols else selected_cols
            ops_to_apply = ["Lowest", "Highest", "Average"] if "All" in operation else operation

            summary = {}
            for col in cols_to_use:
                summary[col] = {}
                if "Lowest" in ops_to_apply:
                    summary[col]["Lowest"] = future_df[col].min()
                if "Highest" in ops_to_apply:
                    summary[col]["Highest"] = future_df[col].max()
                if "Average" in ops_to_apply:
                    summary[col]["Average"] = future_df[col].mean()

            df_summary = pd.DataFrame(summary).T
            st.dataframe(df_summary)

            csv_summary = df_summary.reset_index().to_csv(index=False).encode()
            st.download_button(
                "📥 Download Summary as CSV",
                data=csv_summary,
                file_name="predicted_summary.csv",
                mime="text/csv"
            )
