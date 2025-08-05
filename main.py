import streamlit as st 
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd

st.title("🛒 Sales Forecasting Dashboard")

uploaded_file =st.file_uploader('Upload Excel File',type=["csv"])

@st.cache_data
def load_data(uploaded_file):
  df = pd.read_csv(uploaded_file,parse_dates=['date'])
  return df 


if uploaded_file:

  df = load_data(uploaded_file)


  store_options = sorted(df['store_nbr'].unique())
  family_options = sorted(df['family'].unique())

  # sidebar
  store_id = st.sidebar.selectbox('Select Store',store_options)
  family = st.sidebar.selectbox('Select Family',family_options)

  # Filtered Data 
  df_filtered = df[(df['store_nbr']==store_id) & (df['family']==family)]

  # Prophet 
  df_prophet = df_filtered[['date','sales','onpromotion']].rename(columns = {
    'date':'ds',
    'sales': 'y'
  })

  df_prophet['onpromotion'] = df_prophet['onpromotion'].fillna(0)

  # Train model 
  model = Prophet()
  model.add_regressor('onpromotion')
  model.fit(df_prophet)

  # Future Frame 
  future = model.make_future_dataframe(periods=720)
  future['onpromotion']=df_prophet['onpromotion'].mean()
  forecast = model.predict(future)

  # Plot Forecast 
  fig = plot_plotly(model,forecast)
  st.plotly_chart(fig)

  # raw forecast 
  with st.expander('📄 Show Forecast Data'):
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(60))
else:
  st.warning("Please upload an Excel file with columns: `date`, `store_nbr`, `family`, `sales`, `onpromotion`.")