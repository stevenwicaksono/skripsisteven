"""Referensi koordinat provinsi Indonesia:
  Aceh                : LAT=5.5483,   LON=95.3238
  Bali                : LAT=-8.4095,  LON=115.1889
  Banten              : LAT=-6.4058,  LON=106.0640
  Bengkulu            : LAT=-3.7928,  LON=102.2608
  DI Yogyakarta       : LAT=-7.8754,  LON=110.4262
  DKI Jakarta         : LAT=-6.2088,  LON=106.8456
  Gorontalo           : LAT=0.6999,   LON=122.4467
  Jambi               : LAT=-1.6101,  LON=103.6131
  Jawa Barat          : LAT=-6.9147,  LON=107.6098
  Jawa Tengah         : LAT=-7.1500,  LON=110.1403
  Jawa Timur          : LAT=-7.5360,  LON=112.2384
  Kalimantan Barat    : LAT=-0.2787,  LON=111.4753
  Kalimantan Selatan  : LAT=-3.0926,  LON=115.2838
  Kalimantan Tengah   : LAT=-1.6814,  LON=113.3824
  Kalimantan Timur    : LAT=0.5387,   LON=116.4194
  Kalimantan Utara    : LAT=3.0731,   LON=116.0413
  Kep. Bangka Belitung: LAT=-2.7411,  LON=106.4406
  Kep. Riau           : LAT=3.9457,   LON=108.1429
  Lampung             : LAT=-4.5585,  LON=105.4068
  Maluku              : LAT=-3.2385,  LON=130.1453
  Maluku Utara        : LAT=1.5709,   LON=127.8088
  Nusa Tenggara Barat : LAT=-8.6529,  LON=117.3616
  Nusa Tenggara Timur : LAT=-8.6574,  LON=121.0794
  Papua               : LAT=-4.2699,  LON=138.0804
  Papua Barat         : LAT=-1.3361,  LON=133.1747
  Papua Barat Daya    : LAT=-1.3500,  LON=132.0000
  Papua Pegunungan    : LAT=-4.0000,  LON=138.5000
  Papua Selatan       : LAT=-6.5000,  LON=140.0000
  Papua Tengah        : LAT=-3.5000,  LON=135.5000
  Sulawesi Barat      : LAT=-2.8441,  LON=119.2321
  Sulawesi Selatan    : LAT=-3.6687,  LON=119.9740
  Sulawesi Tengah     : LAT=-1.4300,  LON=121.4456
  Sulawesi Tenggara   : LAT=-4.1449,  LON=122.1746
  Sulawesi Utara      : LAT=1.4300,   LON=124.8420
  Sumatera Barat      : LAT=-0.7399,  LON=100.8000
  Sumatera Selatan    : LAT=-3.3194,  LON=104.9144
  Sumatera Utara      : LAT=2.1154,   LON=99.5451
"""

import requests
import pandas as pd
import numpy as np
import ephem

LAT   = 3.9457
LON   = 108.1429
NAME  = "Kep. Riau"
START = "20180101"
END   = "20241231"

_name_file = NAME.replace(" ", "_").replace(".", "")
_start_yr  = START[:4]
_end_yr    = END[:4]
OUTPUT     = f"{_name_file}_{_start_yr}_{_end_yr}.csv"

print("=" * 60)
print(f"  Download Data Cuaca: {NAME}")
print(f"  Sumber: NASA POWER API")
print("=" * 60)
print(f"\n  Provinsi  : {NAME}")
print(f"  Koordinat : {LAT}, {LON}")
print(f"  Rentang   : {START} - {END}")
print(f"  Output    : {OUTPUT}")
print(f"\n  Mengambil data dari NASA POWER...")

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M_MAX,T2M_MIN,T2M,T2MDEW,RH2M,PRECTOTCORR,WS10M_MAX,WS10M,WD10M,PS,ALLSKY_SFC_SW_DWN,ALLSKY_KT",
    "community": "AG",
    "longitude": LON,
    "latitude":  LAT,
    "start":     START,
    "end":       END,
    "format":    "JSON",
    "time-standard": "LST",
}

resp = requests.get(url, params=params, timeout=180)
resp.raise_for_status()
props = resp.json()["properties"]["parameter"]
print(f"  Data berhasil diambil dari NASA POWER")

dates = list(props["T2M_MAX"].keys())
df = pd.DataFrame({"date_str": dates})
df["datetime_dt"] = pd.to_datetime(df["date_str"], format="%Y%m%d")

col_map = {
    "T2M_MAX"          : "tempmax",
    "T2M_MIN"          : "tempmin",
    "T2M"              : "temp",
    "T2MDEW"           : "dew",
    "RH2M"             : "humidity",
    "PRECTOTCORR"      : "precip",
    "WS10M_MAX"        : "windspeed_ms",
    "WS10M"            : "windspeed_mean_ms",
    "WD10M"            : "winddir",
    "PS"               : "pressure_kpa",
    "ALLSKY_SFC_SW_DWN": "solar_raw",
    "ALLSKY_KT"        : "clearness",
}
for nasa_key, col in col_map.items():
    df[col] = df["date_str"].map(props[nasa_key])
    df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999.0, np.nan)

df["sealevelpressure"] = (df["pressure_kpa"] * 10).round(1)
df["windspeed"]        = (df["windspeed_ms"] * 3.6).round(1)
df["windgust"]         = (df["windspeed_ms"] * 3.6 * 1.3).round(1)
df["solarradiation"]   = df["solar_raw"].round(1)
df["solarenergy"]      = (df["solar_raw"] * 86400 / 1e6).round(2)
df["cloudcover"]       = ((1 - df["clearness"].fillna(0.5)) * 100).clip(0, 100).round(1)
df["visibility"]       = 16.1
df["snow"]             = 0.0
df["snowdepth"]        = 0.0
df["severerisk"]       = 0

def heat_index(T, RH):
    if pd.isna(T) or pd.isna(RH):
        return T
    HI = (-8.78469475556 + 1.61139411*T + 2.33854883889*RH
          - 0.14611605*T*RH - 0.012308094*T**2
          - 0.0164248277778*RH**2 + 0.002211732*T**2*RH
          + 0.00072546*T*RH**2 - 0.000003582*T**2*RH**2)
    return round(float(HI), 1)

df["feelslike"]    = df.apply(lambda r: heat_index(r["temp"],    r["humidity"]), axis=1)
df["feelslikemax"] = df.apply(lambda r: heat_index(r["tempmax"], r["humidity"]), axis=1)
df["feelslikemin"] = df.apply(lambda r: heat_index(r["tempmin"], r["humidity"]), axis=1)

df["preciptype"]  = df["precip"].apply(lambda x: "rain" if pd.notna(x) and x > 0 else "")
df["precipprob"]  = df["precip"].apply(lambda x: min(100, round(x * 10, 0)) if pd.notna(x) and x > 0 else 0)
df["precipcover"] = df["precip"].apply(lambda x: min(100, round(x * 4,  2)) if pd.notna(x) and x > 0 else 0)
df["uvindex"]     = (df["solar_raw"].fillna(0) / 25).clip(0, 15).round(0).astype(int)

def get_cond(row):
    p = row["precip"]     if pd.notna(row["precip"])     else 0
    c = row["cloudcover"] if pd.notna(row["cloudcover"]) else 0
    if p > 0:
        if c > 70:
            return "Rain, Overcast", "Cloudy skies throughout the day with rain.", "rain"
        return "Rain, Partially cloudy", "Partly cloudy with rain throughout the day.", "rain"
    if c > 80:
        return "Overcast", "Cloudy skies throughout the day.", "cloudy"
    if c > 40:
        return "Partially cloudy", "Partly cloudy throughout the day.", "partly-cloudy-day"
    return "Clear", "Clear conditions throughout the day.", "clear-day"

cond = df.apply(get_cond, axis=1, result_type="expand")
df["conditions"]  = cond[0]
df["description"] = cond[1]
df["icon"]        = cond[2]

print(f"\n  Menghitung sunrise, sunset, moonphase ({len(df)} hari)...")
print("  Harap tunggu ~2 menit...")

obs = ephem.Observer()
obs.lat       = str(LAT)
obs.lon       = str(LON)
obs.elevation = 15
obs.pressure  = 0

sunrises, sunsets, moonphases = [], [], []
for i, dt in enumerate(df["datetime_dt"]):
    obs.date = dt.strftime("%Y/%m/%d 06:00:00")
    sun  = ephem.Sun()
    moon = ephem.Moon()
    try:
        sr = ephem.localtime(obs.next_rising(sun)).strftime("%Y-%m-%dT%H:%M:%S")
        ss = ephem.localtime(obs.next_setting(sun)).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        sr = ss = ""
    moon.compute(obs)
    sunrises.append(sr)
    sunsets.append(ss)
    moonphases.append(round(float(moon.phase) / 100, 2))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{len(df)} hari selesai...")

df["sunrise"]   = sunrises
df["sunset"]    = sunsets
df["moonphase"] = moonphases

df["datetime_fmt"] = df["datetime_dt"].apply(lambda d: f"{d.month}/{d.day}/{d.year}")

out = pd.DataFrame()
out["name"]             = [NAME] * len(df)
out["datetime"]         = df["datetime_fmt"]
out["tempmax"]          = df["tempmax"].round(1)
out["tempmin"]          = df["tempmin"].round(1)
out["temp"]             = df["temp"].round(1)
out["feelslikemax"]     = df["feelslikemax"]
out["feelslikemin"]     = df["feelslikemin"]
out["feelslike"]        = df["feelslike"]
out["dew"]              = df["dew"].round(1)
out["humidity"]         = df["humidity"].round(1)
out["precip"]           = df["precip"].fillna(0).round(2)
out["precipprob"]       = df["precipprob"]
out["precipcover"]      = df["precipcover"]
out["preciptype"]       = df["preciptype"]
out["snow"]             = df["snow"]
out["snowdepth"]        = df["snowdepth"]
out["windgust"]         = df["windgust"]
out["windspeed"]        = df["windspeed"]
out["winddir"]          = df["winddir"].round(1)
out["sealevelpressure"] = df["sealevelpressure"]
out["cloudcover"]       = df["cloudcover"]
out["visibility"]       = df["visibility"]
out["solarradiation"]   = df["solarradiation"]
out["solarenergy"]      = df["solarenergy"]
out["uvindex"]          = df["uvindex"]
out["severerisk"]       = df["severerisk"]
out["sunrise"]          = df["sunrise"]
out["sunset"]           = df["sunset"]
out["moonphase"]        = df["moonphase"]
out["conditions"]       = df["conditions"]
out["description"]      = df["description"]
out["icon"]             = df["icon"]
out["stations"]         = [f"NASA-POWER"] * len(df)

out.to_csv(OUTPUT, index=False)

print(f"\n  SELESAI!")
print(f"  Provinsi  : {NAME}")
print(f"  File      : {OUTPUT}")
print(f"  Total     : {len(out)} baris")
print(f"  Rentang   : {out['datetime'].iloc[0]} - {out['datetime'].iloc[-1]}")
print(f"\n  Cek kolom name: {out['name'].unique()}")
print(f"\n  Contoh 3 baris pertama:")
print(out.head(3)[['name','datetime','temp','humidity','precip']].to_string(index=False))