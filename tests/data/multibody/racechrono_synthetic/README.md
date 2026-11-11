# Synthetic RaceChrono exports

Made-up logs in the layout of the RaceChrono CSV exports, for the tests of
`solvcon.multibody.suspension.racechrono`.  RaceChrono publishes no
specification; the layout follows third-party importers of the format.

* `session_v3.csv`: CSV v3 ("Format 3"): metadata lines, the row of channel
  keys starting with `timestamp`, a row of units, a row of sources, then 10 s
  of samples at 20 Hz of a slalom at 20 m/s (yaw rate 0.3 sin(pi t) rad/s,
  a_y = V r).  The lateral acceleration is written + to the right, the
  bearing starts at 355 deg and wraps past 360, and every fourth gyro sample
  is missing.
* `session_v2.csv`: the older single-header export with the unit in
  parentheses, the first 2 s of the same slalom (+ left).
