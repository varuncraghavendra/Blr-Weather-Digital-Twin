# Blr'Weather Digital Twin

A click-anywhere weather and air-quality dashboard for Bengaluru. It is built on
[NVIDIA Earth2Studio](https://nvidia.github.io/earth2studio/) and runs a
FourCastNet forecast on a local GPU.

The map covers only the Bengaluru city boundary (OSM relation 7902476). Click any
point inside it and the panel loads:

| Panel | Source | Freshness |
|---|---|---|
| Address, PIN, elevation | OpenStreetMap Nominatim, Open-Meteo | Looked up on each click |
| Current conditions + hourly outlook | Open-Meteo (model blend, not stations) | Live, page refreshes every 5 min |
| Air quality (US AQI, PM2.5, PM10, NO₂, O₃, SO₂, CO) | CAMS via Open-Meteo (model, not sensors) | Live, hourly |
| 5-day forecast + map grid | **Earth2Studio FourCastNet (0.25°)** started from NOAA GFS | Re-run every 6 h on the local GPU |

The India AQI badge is an estimate from the current PM2.5 and PM10 using CPCB
breakpoints. The official index uses 24-hour averages from CPCB stations.

## How the Earth-2 part works

`forecast.py` follows Earth2Studio's *Running Deterministic Inference* example
(`examples/01_getting_started`), with two changes:

- **FourCastNet** (AFNO, 0.25°) replaces DLWP, so the city spans real grid cells.
- The run starts from the **newest GFS data cycle** on AWS instead of a fixed
  date. The output is cropped to 12–14°N, 76.5–78.75°E.

It rolls forward 20 six-hour steps (5 days) and writes `forecast_bengaluru.json`.
The server interpolates that file to whatever point you click. FourCastNet cells
are about 28 km wide, so values change smoothly across the city rather than
neighbourhood by neighbourhood.

## Run it

You need an NVIDIA GPU (tested on an RTX 4080 Laptop GPU, where one run takes
about 10 s) and Python 3.12. Python 3.11.0rc1 breaks `torch._dynamo`.

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python forecast.py --nsteps 20   # first forecast; the server also schedules it
.venv/bin/python server.py                 # http://localhost:8050
```

Set `E2_SCHEDULE=0` to turn off the 6-hourly re-run, and `PORT` or `HOST` to
change where the server binds.

## Files

- `server.py`: FastAPI backend that proxies live feeds, serves forecasts and schedules runs
- `forecast.py`: the Earth2Studio FourCastNet run
- `index.html`: the Leaflet + Chart.js dashboard
- `bengaluru.geojson`: the city boundary

Map tiles © OpenStreetMap contributors. Forecast initial conditions: NOAA GFS.
Air quality: Copernicus CAMS.
