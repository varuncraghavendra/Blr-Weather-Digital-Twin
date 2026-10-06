"""Earth2Studio "Running Deterministic Inference" example, run for Bengaluru.

Same workflow as examples/01_getting_started/01_deterministic_workflow.py
(prognostic model + GFS initial conditions + run.deterministic), with two changes:
  * FourCastNet (FCN, 0.25 deg) instead of DLWP, so the city spans real grid cells
  * the newest GFS cycle instead of a fixed 2024 date, cropped to a box around Bengaluru

Writes forecast_bengaluru.json for the dashboard.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "forecast_bengaluru.json")
STATUS = os.path.join(HERE, "forecast_status.json")

# Box around Bengaluru city (city bbox is 77.46-77.78 E, 12.83-13.14 N)
LAT_MIN, LAT_MAX = 12.0, 14.0
LON_MIN, LON_MAX = 76.5, 78.75
VARIABLES = ["t2m", "u10m", "v10m", "msl", "sp", "tcwv", "r850", "t850"]


def write_status(state, **extra):
    with open(STATUS, "w") as f:
        json.dump({"state": state, "updated": datetime.now(timezone.utc).isoformat(), **extra}, f)


def candidate_cycles(n=4):
    """GFS cycles newest first. A cycle lands on AWS roughly 4-5 h after its nominal time."""
    now = datetime.now(timezone.utc) - timedelta(hours=4, minutes=30)
    cyc = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0, microsecond=0)
    return [cyc - timedelta(hours=6 * i) for i in range(n)]


def main(nsteps):
    write_status("loading model")
    import torch
    import earth2studio.run as run
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import FCN

    package = FCN.load_default_package()
    model = FCN.load_model(package)
    data = GFS()

    last_err = None
    for cycle in candidate_cycles():
        init = cycle.replace(tzinfo=None)
        try:
            write_status("running", init=init.isoformat() + "Z", nsteps=nsteps)
            io = ZarrBackend()
            io = run.deterministic(
                [np.datetime64(init)], nsteps, model, data, io,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            break
        except Exception as e:  # cycle not on AWS yet; fall back to the previous one
            last_err = e
            print(f"GFS cycle {init} failed: {e!r}")
    else:
        write_status("error", error=repr(last_err))
        raise SystemExit(1)

    lat = np.asarray(io["lat"][:])
    lon = np.asarray(io["lon"][:])
    li = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lj = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    lead = np.asarray(io["lead_time"][:]).astype("timedelta64[h]").astype(int).tolist()

    fields = {}
    for v in VARIABLES:
        arr = np.asarray(io[v][0, :, li.min(): li.max() + 1, lj.min(): lj.max() + 1], dtype=np.float32)
        fields[v] = np.round(arr, 3).tolist()

    out = {
        "model": "FourCastNet (FCN, AFNO 0.25 deg) via Earth2Studio",
        "source": "NOAA GFS analysis (earth2studio.data.GFS)",
        "init_time": init.isoformat() + "Z",
        "generated": datetime.now(timezone.utc).isoformat(),
        "lead_hours": lead,
        "lat": lat[li].round(3).tolist(),
        "lon": lon[lj].round(3).tolist(),
        "units": {"t2m": "K", "u10m": "m/s", "v10m": "m/s", "msl": "Pa", "sp": "Pa",
                  "tcwv": "kg/m2", "r850": "%", "t850": "K"},
        "fields": fields,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, OUT)
    write_status("done", init=out["init_time"], nsteps=nsteps)
    print(f"wrote {OUT}: init {out['init_time']}, {len(lead)} lead times, grid {len(li)}x{len(lj)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nsteps", type=int, default=20, help="6-hour steps (20 = 5 days)")
    main(p.parse_args().nsteps)
