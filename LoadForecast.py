import requests, io, pandas as pd
import numpy as np
import datetime as dt
import joblib

# ---------- 1) NTMM: Pin -> mean power & energy Today ----------
ymd = dt.date.today().strftime("%Y-%m-%d")
url = f"https://ntmm.org/pqlog/data/{ymd}.txt"
r = requests.get(url, timeout=20); r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None, engine="python")

P = pd.to_numeric(df.iloc[:, 7], errors="coerce").fillna(0)   # Pin [W]
avg_power_today_W = float(P.mean())
energy_today_kWh  = float(P.sum() * 10 / 3600 / 1000)         # to check it 

# ---------- 2) Open-Meteo: Mean temperature today & tomorrow ----------
def get_avg_temps(lat=59.3293, lon=18.0686):
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m&timezone=Europe/Stockholm")
    r = requests.get(url, timeout=10); r.raise_for_status()
    data = r.json()
    temps = np.array(data["hourly"]["temperature_2m"])
    times = [dt.datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    T_today = float(np.mean([temps[i] for i, t in enumerate(times) if t.date()==today]))
    T_tom   = float(np.mean([temps[i] for i, t in enumerate(times) if t.date()==tomorrow]))
    return T_today, T_tom

T_today, T_tom = get_avg_temps()

# ---------- 3) Load day-model & create X with the right columns ----------
day_model = joblib.load("Models/days_prediction.joblib") 

cols = ["power_active_import_avg_current_day",
        "Temperature_avg_current_day",
        "temperature_avg_1d_future"]

X_day = pd.DataFrame([[avg_power_today_W, T_today, T_tom]], columns=cols)

# ---------- 4) Forecasting for tomorrow ----------
y_pred_avg_W = float(day_model.predict(X_day)[0])             # mean power for tomorrow in (W)
y_pred_energy_kWh = y_pred_avg_W * 24 / 1000                  # try to calculate the energy for tomorrow in kWh/day


import io, requests, pandas as pd, numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
import joblib

# -------------------- Config --------------------
TZ = ZoneInfo("Europe/Stockholm")
LAT, LON = 59.3293, 18.0686
HOURS_MODEL_PATH = "Models/hours_prediction.joblib"   # <== change the path!
y_pred_avg_W = float(y_pred_avg_W)  # από το day-model (σε W)

# -------------------- 1) NTMM TXT -> current-hour mean power --------------------
def fetch_ntmm_txt_df(ymd: str) -> pd.DataFrame:
    url = f"https://ntmm.org/pqlog/data/{ymd}.txt"
    r = requests.get(url, timeout=20); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None, engine="python")
    #  DatetimeIndex with first 6 columns (Y M D h m s)
    parts = df.iloc[:, :6].astype(int)
    dt_index = pd.to_datetime(
        dict(year=parts[0], month=parts[1], day=parts[2],
             hour=parts[3], minute=parts[4], second=parts[5]),
        utc=False
    ).dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    df.index = dt_index
    return df

def get_current_hour_avg_power_W(df: pd.DataFrame) -> float:
    # Column 7 is Pin (W)
    P = pd.to_numeric(df.iloc[:, 7], errors="coerce")
    # Hourly
    hourly = P.resample("1h").mean()
    # 
    now = dt.datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
    last_full_hour = now - dt.timedelta(hours=1)
    if last_full_hour in hourly.index:
        return float(hourly.loc[last_full_hour])
    # 
    if len(hourly) >= 2:
        return float(hourly.iloc[-2])
    return float(hourly.iloc[-1])

today = dt.date.today().strftime("%Y-%m-%d")
df_txt = fetch_ntmm_txt_df(today)
power_active_import_avg_current_hour = get_current_hour_avg_power_W(df_txt)

# -------------------- 2) Open-Meteo temps: current-hour + next 24h --------------------
def get_hourly_temps(lat: float, lon: float, tz: ZoneInfo) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        "&timezone=Europe%2FStockholm"
    )
    r = requests.get(url, timeout=20); r.raise_for_status()
    js = r.json()
    times = pd.to_datetime(js["hourly"]["time"]).tz_localize(tz)
    temps = pd.Series(js["hourly"]["temperature_2m"], index=times, name="temperature_2m")
    return temps.to_frame()

temps_df = get_hourly_temps(LAT, LON, TZ)

# now hour
now_hour = dt.datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
Temperature_avg_current_hour = float(temps_df.reindex([now_hour]).iloc[0, 0])

# Next 24 hours
future_hours_index = [now_hour + dt.timedelta(hours=h) for h in range(1, 25)]
temps_next_24 = temps_df.reindex(future_hours_index)["temperature_2m"].astype(float).tolist()

# -------------------- 3) Build features for hours model --------------------
feat = {
    "power_active_import_avg_1d_future_pred": float(y_pred_avg_W),
    "power_active_import_avg_current_hour":   float(power_active_import_avg_current_hour),
    "Temperature_avg_current_hour":           float(Temperature_avg_current_hour),
}
for h in range(1, 25):
    feat[f"temperature_avg_{h}h_future"] = float(temps_next_24[h-1])

X_hours = pd.DataFrame([feat])

hours_model = joblib.load(HOURS_MODEL_PATH)
needed_cols = list(getattr(hours_model, "feature_names_in_", X_hours.columns))
X_hours = X_hours.reindex(columns=needed_cols)

# -------------------- 4) Predict next-24h power (W) --------------------
y_hours_W = hours_model.predict(X_hours).ravel()   #  we are expecting 24 values
pred_index = pd.Index(future_hours_index, name="timestamp")
forecast_24h = pd.Series(y_hours_W, index=pred_index, name="Pin_pred_W")

# 15-min interpolation
# Create EXACT 15-min index of 96 steps starting at first timestamp
new_index = pd.date_range(start=forecast_24h.index[0], periods=96, freq="15min")

forecast_15min = forecast_24h.reindex(new_index).interpolate("linear")

# save to CSV 
forecast_15min_df = forecast_15min.reset_index()
forecast_15min_df.columns = ["timestamp", "predicted_power_W"]
forecast_15min_df.to_csv("Data/forecast_15min_load.csv", index=False)

# === 7) Cumulative energy over next 24h ===
# 15-min step = 0.25 h
energy_15min_kWh = (forecast_15min * 0.25 / 1000.0)   # W * h / 1000 -> kWh
cum_energy_kWh = energy_15min_kWh.cumsum()

# Make a DataFrame that includes the timestamp column explicitly
df_load = forecast_15min.rename("Pin_pred_W").reset_index()
df_load.columns = ["Time", "Pin_pred_W"]   # βάζουμε καθαρά ονόματα

# Save with Time column
df_load.to_csv("Data/predicted_load_24h.csv", index=False)