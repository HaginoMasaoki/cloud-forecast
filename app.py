import time
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st

# ページ基本設定
st.set_page_config(page_title="48h Cloud Layer Forecast", layout="wide")
st.title("☁ 48-Hour Cloud Layer Analysis")

# ------------------------------------------------------------------------------
# 1. データ取得（キャッシュ化）
# ------------------------------------------------------------------------------
@st.cache_data
def load_geo_data():
    PREF_URL = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
    TOKYO_CITY_URL = "https://raw.githubusercontent.com/niiyz/JapanCityGeoJson/master/geojson/13.json"
    KANAGAWA_CITY_URL = "https://raw.githubusercontent.com/niiyz/JapanCityGeoJson/master/geojson/14.json"
    SAITAMA_CITY_URL = "https://raw.githubusercontent.com/niiyz/JapanCityGeoJson/master/geojson/11.json"
    CHIBA_CITY_URL = "https://raw.githubusercontent.com/niiyz/JapanCityGeoJson/master/geojson/12.json"

    gdf_p = gpd.read_file(PREF_URL)
    try:
        gdf_c = pd.concat([
            gpd.read_file(TOKYO_CITY_URL),
            gpd.read_file(KANAGAWA_CITY_URL),
            gpd.read_file(SAITAMA_CITY_URL),
            gpd.read_file(CHIBA_CITY_URL)
        ], ignore_index=True)
    except Exception:
        gdf_c = None
    return gdf_p, gdf_c

@st.cache_data(ttl=3600)
def fetch_weather_data():
    lats = np.linspace(35.2, 36.2, 12)
    lons = np.linspace(138.8, 140.2, 14)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()

    lat_str = ",".join(map(str, flat_lats))
    lon_str = ",".join(map(str, flat_lons))

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat_str}&longitude={lon_str}&"
        f"hourly=cloud_cover_high,cloud_cover_mid,cloud_cover_low,cloud_cover&"
        f"forecast_days=2&timezone=Asia%2FTokyo"
    )
    res = requests.get(url).json()
    return res, flat_lons, flat_lats

gdf_pref, gdf_cities = load_geo_data()
weather_response, flat_lons, flat_lats = fetch_weather_data()
time_list = weather_response[0]["hourly"]["time"]
total_hours = len(time_list)

fine_x, fine_y = np.meshgrid(
    np.linspace(138.8, 140.2, 120),
    np.linspace(35.2, 36.2, 120)
)

# 文字化け防止のためタイトルを完全英語表記に変更
layers_config = {
    "Low Cloud": {"key": "cloud_cover_low", "cmap": "Greys", "color": "black", "title": "Low Cloud"},
    "Mid Cloud": {"key": "cloud_cover_mid", "cmap": "Blues", "color": "#0055ff", "title": "Mid Cloud"},
    "High Cloud": {"key": "cloud_cover_high", "cmap": "Purples", "color": "#aa00ff", "title": "High Cloud"}
}

# ------------------------------------------------------------------------------
# 2. セッション状態（再生/停止状態）
# ------------------------------------------------------------------------------
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "hour_idx" not in st.session_state:
    st.session_state.hour_idx = 0

# ------------------------------------------------------------------------------
# 3. サイドバーUI
# ------------------------------------------------------------------------------
st.sidebar.header("Control Panel")

selected_layer_option = st.sidebar.selectbox(
    "Select Cloud Layer:",
    ["All Layers", "Low Cloud", "Mid Cloud", "High Cloud"]
)

# 再生・一時停止ボタン
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶ Play"):
        st.session_state.is_playing = True
with col2:
    if st.button("❚❚ Pause"):
        st.session_state.is_playing = False

# 時間スライダー
selected_hour_idx = st.sidebar.slider(
    "Select Time (0~47h):",
    min_value=0,
    max_value=total_hours - 1,
    value=st.session_state.hour_idx,
    key="slider_hour",
    format="%d h"
)

if selected_hour_idx != st.session_state.hour_idx and not st.session_state.is_playing:
    st.session_state.hour_idx = selected_hour_idx

current_idx = st.session_state.hour_idx

# ------------------------------------------------------------------------------
# 4. 地図描画処理
# ------------------------------------------------------------------------------
t_label = time_list[current_idx].replace("T", " ")
hour_val = int(t_label.split(" ")[1].split(":")[0])
is_daytime = 6 <= hour_val < 18
bg_color = "#FFFFF5" if is_daytime else "#F0F4F8"
day_night_tag = "[Daytime]" if is_daytime else "[Night]"

st.subheader(f"📅 Time: {t_label} JST  {day_night_tag}")

def plot_single_layer(ax, layer_key, cfg):
    ax.set_facecolor(bg_color)
    cloud_vals = [loc["hourly"][cfg["key"]][current_idx] for loc in weather_response]
    zi = griddata((flat_lons, flat_lats), cloud_vals, (fine_x, fine_y), method='cubic')

    if gdf_cities is not None:
        gdf_cities.plot(ax=ax, facecolor="none", edgecolor="#999999", linewidth=0.4, linestyle=":", zorder=4)
    gdf_pref.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.0, zorder=5)

    cf = ax.contourf(fine_x, fine_y, zi, levels=np.linspace(0, 100, 21), cmap=cfg["cmap"], alpha=0.7, zorder=2)
    ax.contour(fine_x, fine_y, zi, levels=[20, 50, 80], colors=cfg["color"], linewidths=0.8, linestyles="--", zorder=3)
    ax.plot(139.5337, 35.6762, 'ro', markersize=5, label="Mitaka", zorder=6)

    ax.set_xlim(138.9, 140.1)
    ax.set_ylim(35.3, 36.1)
    ax.set_title(f"{cfg['title']} Cover (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.12)
    fig.colorbar(cf, cax=cax)

# 地図表示
if selected_layer_option == "All Layers":
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=bg_color)
    for idx, (k, cfg) in enumerate(layers_config.items()):
        plot_single_layer(axes[idx], k, cfg)
    plt.tight_layout()
    st.pyplot(fig)
else:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), facecolor=bg_color)
        plot_single_layer(ax, selected_layer_option, layers_config[selected_layer_option])
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------------------------
# 5. 48時間サマリーデータ（表）の出力
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 48-Hour Cloud Forecast Summary")

summary_data = []
for idx, t_item in enumerate(time_list):
    avg_total = np.mean([loc["hourly"]["cloud_cover"][idx] for loc in weather_response])
    avg_low = np.mean([loc["hourly"]["cloud_cover_low"][idx] for loc in weather_response])
    avg_mid = np.mean([loc["hourly"]["cloud_cover_mid"][idx] for loc in weather_response])
    avg_high = np.mean([loc["hourly"]["cloud_cover_high"][idx] for loc in weather_response])
    
    # 観測適性判定 (〇 / △ / ×)
    rating = "〇 (Clear)" if avg_total <= 5 else ("△ (Partly)" if avg_total <= 10 else "× (Cloudy)")
    
    summary_data.append({
        "Time (JST)": t_item.replace("T", " "),
        "Total Cloud (%)": round(avg_total, 1),
        "Low Cloud (%)": round(avg_low, 1),
        "Mid Cloud (%)": round(avg_mid, 1),
        "High Cloud (%)": round(avg_high, 1),
        "Rating": rating
    })

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary, use_container_width=True, height=250)

# ------------------------------------------------------------------------------
# 6. アニメーション制御
# ------------------------------------------------------------------------------
if st.session_state.is_playing:
    time.sleep(0.4)
    st.session_state.hour_idx = (st.session_state.hour_idx + 1) % total_hours
    st.rerun()
