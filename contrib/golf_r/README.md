# Experimental Golf R dynamics demonstration

The model is a research prototype. Hardpoint origins, spring mounting,
inertia, damping and other parameters are assumptions; neither its geometry
nor its response has been calibrated against a measured vehicle. Do not use
its numerical output as validated Golf R data.

Supply your own extracted vehicle data and GLB model. These assets are not
included in modmesh. From the repository root, with a built modmesh and
PySide6, run:

```sh
PYTHONPATH=. python3 contrib/golf_r/step_steer.py \
  --data-dir /path/to/extracted/car/data \
  --model /path/to/car.glb --output body_dynamics.csv
```

The data directory must contain `suspensions.ini`; the GLB must contain the
four named wheel hierarchies `WHEEL_LF`, `WHEEL_RF`, `WHEEL_LR` and `WHEEL_RR`.
The example defaults to 43.2 km/h and -4 mm steering-rack travel. Change them
with `--speed-kmh` and `--rack-mm`. The program overwrites the requested CSV.

The demonstration uses `solvcon.multibody.suspension.car.step_steer` and
shows the same solved chassis and upright poses in the Model Viewer and the
Suspension K&C view. Playback is in the vehicle-following frame. The dashboard
and CSV contain body heave, roll, pitch and four wheel travels relative to the
first sampled pose; wheel bump is upward in the chassis frame. Wheel rotation
about the axle is a visual estimate from speed, not a solved wheel spin.

The case is a steering transient with a simplified lateral tire, not a
trail-braking simulation. There is no coupled longitudinal braking force,
calibrated tire curve, ABS, differential dynamics or full nonlinear suspension
stops. The inputs and limits are documented in the model modules. The model
requires measured geometry, mass and tire data, then independent validation,
before it can support quantitative conclusions about the actual vehicle.

The application modules are headless and accept a path to the external data:

```python
from solvcon.apps.golf_r import body_dynamics
result = body_dynamics.run('/path/to/extracted/car/data')
body_dynamics.export_csv(result, 'body_dynamics.csv')
```

The standard library tests use synthetic inputs and do not require vehicle
assets:

```sh
PYTHONPATH=. python3 -m unittest discover -s tests/apps/golf_r -v
```
