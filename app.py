
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="AI Tahmin Paneli + Hata Analizi", layout="centered")
st.title("📊 Prophet Tahmin Paneli + Hata Skoru")

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

        # Hata analizi (gerçek ile tahmini karşılaştır)
        # Son 7 gün tahmini vs gerçekleşeni karşıla
        df_real = yf.download(selected_asset, period="10d")["Close"].dropna().reset_index()
        df_real.columns = ["Tarih", "Gerçek"]
        df_pred = forecast[["ds", "yhat"]].rename(columns={"ds": "Tarih", "yhat": "Tahmin"})
        merged = pd.merge(df_real, df_pred, on="Tarih", how="inner").tail(7)

        if len(merged) >= 3:
            mape = (abs((merged["Gerçek"] - merged["Tahmin"]) / merged["Gerçek"])).mean() * 100
            mae = mean_absolute_error(merged["Gerçek"], merged["Tahmin"])
            rmse = mean_squared_error(merged["Gerçek"], merged["Tahmin"], squared=False)

            st.subheader("📐 Hata Analizi (Son 7 Gün)")
            st.markdown(f"- **MAPE (Yüzde Hata)**: {mape:.2f} %")
            st.markdown(f"- **MAE (Ortalama Mutlak Hata)**: {mae:.2f}")
            st.markdown(f"- **RMSE (Karekök Ortalama Hata)**: {rmse:.2f}")
        else:
            st.info("Yeterli veri bulunamadı, hata skoru gösterilemiyor.")
