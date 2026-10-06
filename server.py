"""Bengaluru Earth-2 dashboard server.

  * /api/point     live data for a clicked point: reverse geocode (OSM Nominatim),
                   current weather + hourly outlook (Open-Meteo), air quality
                   (Open-Meteo / CAMS), and the Earth2Studio FourCastNet forecast
                   interpolated to that point
  * /api/forecast  the FourCastNet run over Bengaluru (from forecast.py)
  * a scheduler re-runs forecast.py every 6 h so it follows each new GFS cycle

Run:  .venv/bin/python server.py   (then open http://localhost:8050)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
FORECAST = os.path.join(HERE, "forecast_bengaluru.json")
STATUS = os.path.join(HERE, "forecast_status.json")
UA = "earth2-bengaluru-dashboard/1.0 (local research dashboard)"
RERUN_SECONDS = 6 * 3600

with open(os.path.join(HERE, "bengaluru.geojson")) as f:
    BOUNDARY = json.load(f)
RING = BOUNDARY["features"][0]["geometry"]["coordinates"][0]

app = FastAPI(title="Bengaluru Earth-2 dashboard")
client: httpx.AsyncClient | None = None
geocode_cache: dict[tuple, dict] = {}
geocode_lock = asyncio.Lock()
last_geocode = 0.0
forecast_proc: subprocess.Popen | None = None


def inside(lon, lat, ring=RING):
    """Ray-casting point-in-polygon test against the city boundary."""
    hit = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def load_forecast():
    try:
        with open(FORECAST) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def bilinear(fc, var, lat, lon):
    """Interpolate one forecast variable to (lat, lon) for every lead time."""
    lats, lons, field = fc["lat"], fc["lon"], fc["fields"][var]

    def bracket(axis, x):
        desc = axis[0] > axis[-1]
        for k in range(len(axis) - 1):
            a, b = axis[k], axis[k + 1]
            if (a >= x >= b) if desc else (a <= x <= b):
                return k, (x - a) / (b - a)
        raise ValueError("point outside forecast box")

    i, ty = bracket(lats, lat)
    j, tx = bracket(lons, lon)
    out = []
    for step in field:
        v00, v01 = step[i][j], step[i][j + 1]
        v10, v11 = step[i + 1][j], step[i + 1][j + 1]
        out.append((v00 * (1 - tx) + v01 * tx) * (1 - ty) + (v10 * (1 - tx) + v11 * tx) * ty)
    return out


def point_forecast(fc, lat, lon):
    if not fc:
        return None
    t2m = bilinear(fc, "t2m", lat, lon)
    u = bilinear(fc, "u10m", lat, lon)
    v = bilinear(fc, "v10m", lat, lon)
    msl = bilinear(fc, "msl", lat, lon)
    tcwv = bilinear(fc, "tcwv", lat, lon)
    r850 = bilinear(fc, "r850", lat, lon)
    return {
        "init_time": fc["init_time"],
        "lead_hours": fc["lead_hours"],
        "t2m_c": [round(x - 273.15, 2) for x in t2m],
        "wind_kmh": [round(((a * a + b * b) ** 0.5) * 3.6, 1) for a, b in zip(u, v)],
        "msl_hpa": [round(x / 100, 1) for x in msl],
        "tcwv": [round(x, 1) for x in tcwv],
        "r850": [round(x, 1) for x in r850],
    }


async def reverse_geocode(lat, lon):
    """Nominatim allows one request per second; cache by ~100 m cell."""
    global last_geocode
    key = (round(lat, 3), round(lon, 3))
    if key in geocode_cache:
        return geocode_cache[key]
    async with geocode_lock:
        wait = 1.05 - (time.monotonic() - last_geocode)
        if wait > 0:
            await asyncio.sleep(wait)
        r = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
            headers={"User-Agent": UA, "Accept-Language": "en"},
        )
        last_geocode = time.monotonic()
    r.raise_for_status()
    d = r.json()
    a = d.get("address", {})
    res = {
        "name": d.get("name") or a.get("road") or a.get("neighbourhood") or "",
        "road": a.get("road"),
        "locality": a.get("suburb") or a.get("neighbourhood") or a.get("quarter") or a.get("village"),
        "district": a.get("city_district") or a.get("county"),
        "city": a.get("city") or a.get("town") or "Bengaluru",
        "postcode": a.get("postcode"),
        "display_name": d.get("display_name"),
        "osm": f"https://www.openstreetmap.org/{d.get('osm_type')}/{d.get('osm_id')}" if d.get("osm_id") else None,
    }
    geocode_cache[key] = res
    return res


async def weather(lat, lon):
    r = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,dew_point_2m,"
                       "precipitation,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,"
                       "wind_direction_10m,wind_gusts_10m,weather_code,is_day,uv_index",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m",
            "daily": "sunrise,sunset,uv_index_max,precipitation_sum",
            "forecast_days": 6,
        },
    )
    r.raise_for_status()
    return r.json()


async def air(lat, lon):
    r = await client.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,"
                       "aerosol_optical_depth,dust",
        },
    )
    r.raise_for_status()
    return r.json()


async def safe(coro):
    try:
        return {"ok": True, "data": await coro}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/boundary")
def boundary():
    return BOUNDARY


@app.get("/api/point")
async def point(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    if not inside(lon, lat):
        raise HTTPException(400, "That point is outside the Bengaluru city boundary.")
    geo, wx, aq = await asyncio.gather(safe(reverse_geocode(lat, lon)), safe(weather(lat, lon)), safe(air(lat, lon)))
    fc = load_forecast()
    try:
        e2 = {"ok": True, "data": point_forecast(fc, lat, lon)} if fc else {"ok": False, "error": "No forecast yet"}
    except Exception as e:
        e2 = {"ok": False, "error": str(e)}
    return {
        "lat": lat, "lon": lon,
        "fetched": datetime.now(timezone.utc).isoformat(),
        "geocode": geo, "weather": wx, "air": aq, "earth2": e2,
    }


@app.get("/api/forecast")
def forecast():
    fc = load_forecast()
    if not fc:
        raise HTTPException(404, "No forecast has finished yet.")
    return JSONResponse(fc, headers={"Cache-Control": "no-store"})


def read_status():
    try:
        with open(STATUS) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {"state": "idle"}
    s["running"] = forecast_proc is not None and forecast_proc.poll() is None
    return s


@app.get("/api/forecast/status")
def forecast_status():
    return read_status()


def start_forecast():
    global forecast_proc
    if forecast_proc is not None and forecast_proc.poll() is None:
        return False
    log = open(os.path.join(HERE, "forecast.log"), "a")
    forecast_proc = subprocess.Popen([sys.executable, os.path.join(HERE, "forecast.py")], cwd=HERE,
                                     stdout=log, stderr=subprocess.STDOUT)
    return True


@app.post("/api/forecast/run")
def forecast_run():
    return {"started": start_forecast(), **read_status()}


async def scheduler():
    while True:
        fc = load_forecast()
        age = time.time() - os.path.getmtime(FORECAST) if fc else None
        if age is None or age > RERUN_SECONDS:
            start_forecast()
        await asyncio.sleep(600)


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=20)
    if os.environ.get("E2_SCHEDULE", "1") == "1":
        asyncio.create_task(scheduler())


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8050)))
