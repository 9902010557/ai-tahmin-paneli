
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Tahmin Paneli", layout="centered")
st.title("📊 AI Destekli Portföy Tahmin Paneli")

assets = [
    "SDTY", "PYTH", "KLSER.IS", "ISGYO.IS", "EKSUN.IS", "HNT", "SARKY.IS", "FRIG.IS",
    "ALGO", "USOI", "USOY", "TIA", "KRDMD.IS", "SOL", "TARKM.IS", "ENERY.IS",
    "OYAKC.IS", "RENDER", "CANTE.IS", "BIENY.IS"
]

selected_asset = st.selectbox("Varlık Seçin", assets)

süre = np.array([1, 7, 30, 90, 180, 365])
base = 100 + np.random.rand() * 20
tahmin = base + np.array([0, 5, 12.5, 30, 50, 75])
alt_band = tahmin * 0.97
üst_band = tahmin * 1.03

fig, ax = plt.subplots()
ax.plot(süre, tahmin, label="Tahmini Fiyat", linewidth=2)
ax.fill_between(süre, alt_band, üst_band, alpha=0.3, label="Güven Aralığı")
ax.set_title(f"{selected_asset} - AI Tahmini")
ax.set_xlabel("Tahmin Süresi (gün)")
ax.set_ylabel("Fiyat")
ax.legend()
st.pyplot(fig)

df = pd.DataFrame({
    "Süre (gün)": süre,
    "Tahmini Fiyat": tahmin.round(2),
    "Alt Band": alt_band.round(2),
    "Üst Band": üst_band.round(2)
})
st.dataframe(df.set_index("Süre (gün)"))
