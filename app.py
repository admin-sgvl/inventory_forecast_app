import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Inventory Forecast Pro", layout="wide")

# --- 1. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Forecast & Supply Settings")
    st.markdown("Adjust these parameters to reflect your supply chain realities.")
    
    forecast_days = st.sidebar.slider("Days to Forecast", 30, 365, 90)
    
    st.divider()
    
    st.subheader("📦 Inventory Logic")
    lead_time = st.number_input("Lead Time (Days to receive stock)", min_value=1, value=7)
    service_level = st.selectbox("Desired Service Level", [0.90, 0.95, 0.99], index=1)
    
    # Z-score mapping for Safety Stock
    z_map = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z_score = z_map[service_level]

# --- 2. MAIN INTERFACE ---
st.title("📈 Inventory & Sales Forecasting Dashboard")
st.markdown("""
Upload your historical sales data to predict future demand and calculate 
optimal stock levels including **Safety Stock** and **Reorder Points**.
""")

uploaded_file = st.file_uploader("Upload Historical Sales (CSV)", type="csv")

if uploaded_file:
    # Load and prep data
    df = pd.read_csv(uploaded_file)
    
    # Ensure column names match Prophet requirements
    # Expecting: Column 0 = Date, Column 1 = Sales
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    
    with st.spinner('Calculating forecast and seasonality patterns...'):
        # 3. THE FORECAST MODEL
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        
        # Create future dates
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        # 4. INVENTORY CALCULATIONS
        # Calculate Standard Deviation of historic sales for Safety Stock
        std_dev = df['y'].std()
        
        # Safety Stock = Z * sqrt(Lead Time) * StdDev
        safety_stock = z_score * np.sqrt(lead_time) * std_dev
        
        # Average Daily Demand (from the forecast period)
        avg_daily_demand = forecast.tail(forecast_days)['yhat'].mean()
        
        # Reorder Point = (Lead Time Demand) + Safety Stock
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        
        # Total forecasted units needed for the chosen period
        total_demand = forecast.tail(forecast_days)['yhat'].sum()

        # 5. DASHBOARD METRICS
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Total Demand", f"{int(total_demand)} units")
        m2.metric("Safety Stock Buffer", f"{int(safety_stock)} units")
        m3.metric("Reorder Point", f"{int(reorder_point)}")
        m4.metric("Service Level", f"{service_level*100}%")

        # 6. VISUALIZATION
        st.subheader("Demand Projection & Safety Thresholds")
        
        fig = go.Figure()

        # Confidence Interval (Shaded area)
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            fill=None, mode='lines', line_color='rgba(0,0,255,0)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.1)',
            name="Forecast Confidence"
        ))

        # Actual Sales
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'], 
            name="Historical Sales", line=dict(color='#333', width=1.5)
        ))

        # Forecasted Sales
        fig.add_trace(go.Scatter(
            x=forecast.iloc[-forecast_days:]['ds'], 
            y=forecast.iloc[-forecast_days:]['yhat'], 
            name="Predicted Demand", line=dict(color='blue', width=3, dash='dash')
        ))

        # Reorder Point Horizontal Line
        fig.add_hline(y=reorder_point, line_dash="dot", line_color="red", 
                      annotation_text="Critical Reorder Point", annotation_position="top right")

        fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 7. EXPORT DATA
        st.divider()
        st.subheader("📥 Export Forecast Data")
        csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        csv_data.columns = ['Date', 'Predicted_Sales', 'Min_Range', 'Max_Range']
        
        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name=f'sales_forecast_{forecast_days}days.csv',
            mime='text/csv',
        )

else:
    st.info("👋 Welcome! Please upload your historical sales CSV to begin. Ensure columns are: Date, Sales Amount.")