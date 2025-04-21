
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Gerçek AI Tahmin Paneli", layout="centered")
st.title("📊 Gerçek Prophet AI Tahmin Paneli")

assets = [
    "SDTY", "PYTH-USD", "KLSER.IS", "ISGYO.IS", "EKSUN.IS", "HNT-USD", "SARKY.IS", "FRIG.IS",
    "ALGO-USD", "USOI", "USOY", "TIA-USD", "KRDMD.IS", "SOL-USD", "TARKM.IS", "ENERY.IS",
    "OYAKC.IS", "RENDER-USD", "CANTE.IS", "BIENY.IS"
]

selected_asset = st.selectbox("Varlık Seçin", assets)
period_input = st.selectbox("Tahmin Süresi", [30, 90, 180, 365], index=2)

with st.spinner("Veriler getiriliyor ve tahmin üretiliyor..."):
    df = yf.download(selected_asset, period="2y")
    if df.empty:
        st.warning("Veri alınamadı.")
    else:
        df = df[["Close"]].reset_index()
        df.columns = ["ds", "y"]
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=period_input)
        forecast = model.predict(future)

        # Grafik
        fig, ax = plt.subplots()
        ax.plot(df["ds"], df["y"], label="Gerçek Fiyat", color="black", linewidth=1.5)
        ax.plot(forecast["ds"], forecast["yhat"], label="Tahmin", color="blue")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Güven Aralığı")
        ax.set_title(f"{selected_asset} - {period_input} Günlük AI Tahmini")
        ax.legend()
        st.pyplot(fig)

        # Tahmin tablosu
        son = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period_input)
        son.columns = ["Tarih", "Tahmin", "Alt Band", "Üst Band"]
        st.dataframe(son.set_index("Tarih").round(2))
