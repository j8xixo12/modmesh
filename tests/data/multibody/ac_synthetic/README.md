# Synthetic Assetto Corsa car data

A made-up compact car in the Assetto Corsa file layout (`car.ini`,
`suspensions.ini`, `tyres.ini`, `brakes.ini`, `drivetrain.ini`) for the
tests of the AC conversion.  The values are plausible for a road car but
belong to no real vehicle; the hardpoints are from the wheel centre (X
inboard, Y up, Z forward), as AC defines them.

`car.ini` [CONTROLS] gives the steering ratios (the rod moves
LINEAR_STEER_ROD_RATIO m per degree of steering wheel angle / STEER_RATIO).
`lods.ini` names the KN5 models by level of detail; `synthetic_b.kn5` (the
second level, the one the 3D view shows) is the model of
`tests/apps/golf_r/test_kn5.py`: four wheel nodes and one triangle as the
body.  The first level `synthetic.kn5` is left out on purpose.
