import pandas as pd
import requests
from datetime import datetime, timedelta
import asyncio, nest_asyncio
from forecast_solar import ForecastSolar
from astral import LocationInfo
from astral.sun import sun

# SPOT PRICES
# Tomorrow
tomorrow = datetime.today() + timedelta(days=1)
date_str = tomorrow.strftime("%Y/%m-%d") 
zone = "SE3"
url = f"https://www.elprisetjustnu.se/api/v1/prices/{date_str}_{zone}.json"

# JSON
response = requests.get(url)
data = response.json()


records = []
for hour in data:
    records.append({
        "Time": hour["time_start"],  #  time_start
        "Price_SEK_kWh": hour["SEK_per_kWh"],
        "Price_EUR_kWh": hour["EUR_per_kWh"]
    })

# DataFrame
df_Spot_Prices = pd.DataFrame(records)
df_Spot_Prices["Time"] = pd.to_datetime(df_Spot_Prices["Time"])
#df_Spot_Prices["Time"] = df_Spot_Prices["Time"].dt.tz_convert("UTC")
df_Spot_Prices = df_Spot_Prices.sort_values("Time").reset_index(drop=True)

# Solar Forecast


nest_asyncio.apply()

# --- Step 2: PV forecast (forecast.solar) ---
async def get_solar():
    async with ForecastSolar(latitude=59.33, longitude=18.06,
                             declination=30, azimuth=180, kwp=5.0) as fs:
        est = await fs.estimate()
        d = pd.DataFrame(list(est.watts.items()), columns=["Time","Solar_watt"])
        d["Time"] = pd.to_datetime(d["Time"], utc=True)  
        return d
try:
    df_solar = asyncio.run(get_solar())
except Exception as e:
    raise SystemExit(f"PV API error: {e}")

df_solar["Time"] = df_solar["Time"].dt.tz_convert("Europe/Stockholm")

# Light window and filtering 


#Filter only the day after values
TZ   = "Europe/Stockholm"
tomorrow  = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
day_after = tomorrow + pd.Timedelta(days=1)

df_solar_day = (df_solar[(df_solar["Time"] >= tomorrow) & (df_solar["Time"] < day_after)]
                  .sort_values("Time").reset_index(drop=True))

#Find the Light window for each day to interpolate the right values and approximate the kW and kWh 
city = LocationInfo("Stockholm", "Sweden", TZ, 59.33, 18.06)
suns = sun(city.observer, date=tomorrow.date(), tzinfo=TZ)
start_light = suns["dawn"]   #  suns["sunrise"]
end_light   = suns["dusk"]   #  suns["sunset"]

# 15' timeline for each day 
grid = pd.DataFrame({"Time": pd.date_range(tomorrow, day_after, freq="15min",
                                           tz=TZ, inclusive="left")})

# merge
d = pd.merge(grid, df_solar_day, on="Time", how="left")

# 0 outside the light window
d.loc[d["Time"] < start_light, "Solar_watt"] = 0.0
d.loc[d["Time"] > end_light,   "Solar_watt"] = 0.0

# Interpolate ony inside the light window
mask = (d["Time"] >= start_light) & (d["Time"] <= end_light)
d.loc[mask, "Solar_watt"] = d.loc[mask, "Solar_watt"].interpolate(method="linear")

# filtering
d["Solar_watt"] = d["Solar_watt"].fillna(0.0).clip(lower=0.0)
d["Solar_kWh_15min"] = d["Solar_watt"]/1000.0 * 0.25

df_solar_15 = d

# --- Step 4: Merge Spot Prices and Solar ---

prices = df_Spot_Prices[["Time", "Price_SEK_kWh"]].sort_values("Time")

df_combined = (pd.merge(prices, df_solar_15, on="Time", how="left")
                 .fillna({"Solar_watt":0.0, "Solar_kWh_15min":0.0})
                 .reset_index(drop=True))


# I am creating the water usage profile in Liters and after that in kWh
PULSE_L = 0.002 # liters per pulse
TZ = "Europe/Stockholm"
START = "2023-05-08"
END   = "2024-05-08"

# --- Load CSV with raw meter data ---
df = pd.read_csv(
    "Models/water_data.csv", #IF YOU RUN IT YOU HAVE TO CHANGE THIS STEP 
    header=None,
    names=["timestamp","pulses","flow_ml_s"],
    on_bad_lines='skip'
)

# timestamps -> datetime Europe/Stockholm
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
# Sort chronologically and keep last sample if duplicates exist
df = (df.sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True))

# keep only desired period (1 year)
start = pd.Timestamp(START, tz=TZ)
end   = pd.Timestamp(END, tz=TZ) + pd.Timedelta(days=1)
df = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()

# Convert pulses (cumulative counter) -> liters per sample
# diff() turns the cumulative counter into per-sample increments
# clip(lower=0) protects against counter resets (negative diffs)
delta_pulses = df["pulses"].diff().clip(lower=0).fillna(0)
df["delta_L"] = delta_pulses * PULSE_L

# Daily aggregation (total liters per day)
daily = df.set_index("datetime")["delta_L"].resample("D").sum().to_frame("daily_L")

# Monthly totals (sum of daily)
monthly = daily.resample("MS").sum().rename(columns={"daily_L":"monthly_L"})

# --- 1. Resample in 15minutes and in Liters ---
df_15 = df.set_index("datetime")["delta_L"].resample("15T").sum().to_frame("L")

# --- 2. Time columns (00:00–23:45) ---
df_15["time_of_day"] = df_15.index.time

# --- 3. mean per time block (96 blocks per day) ---
mean_profile = df_15.groupby("time_of_day")["L"].mean()

#Liters in kWh
kwh_per_liter = 4.186 * 50 / 3600  # ≈ 0.0581 kWh per liter

df_energy = (mean_profile * kwh_per_liter).to_frame(name="kWh_per_15min")

# Here I am loading the Load forecasting for the next day

df_load = pd.read_csv("Data/predicted_load_24h.csv", parse_dates=["Time"])
# ensure tz if needed (the saved ISO may include +01:00)
df_load["Time"] = pd.to_datetime(df_load["Time"])  # parse
# if you want tz-aware:
# df_load["Time"] = df_load["Time"].dt.tz_convert("Europe/Stockholm")

# assume df_combined has a Time column parsed and tz-aware same as df_load
df_combined["Time"] = pd.to_datetime(df_combined["Time"])
df_all = pd.merge(df_combined, df_load, on="Time", how="left")

step_hours = 0.25

df_all["Solar_kWh"]     = (df_all["Solar_watt"]   / 1000.0) * step_hours
df_all["Pin_pred_kWh"]  = (df_all["Pin_pred_W"]   / 1000.0) * step_hours

# Add boiler energy column directly (same length, same ordering)
df_all["Boiler_kWh_per_15min"] = df_energy["kWh_per_15min"].values
#df_all.drop(columns=["time_of_day"], inplace=True)

# Compute load minus boiler
df_all["Load_minus_Boiler_kWh"] = df_all["Pin_pred_kWh"] - df_all["Boiler_kWh_per_15min"]

df_all.to_csv("Data/forecasts.csv", index=False)