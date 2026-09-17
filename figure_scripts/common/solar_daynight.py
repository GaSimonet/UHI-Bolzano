#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astronomical sunrise/sunset day/night classification -- no external deps (no
astral/ephem/pyephem needed on this system).

Self-contained NOAA-style low-precision solar position formulas (Meeus-based),
accurate to within about a minute for sunrise/sunset -- more than enough for
classifying hourly WRF output. Used by seasonal_valley_lapse_rate.py
and seasonal_UHI_WRF_harmonized_dynamiclapse_20260731.py so BOTH the lapse-rate
fit and the UHI maps use the exact same, per-calendar-day day/night split.

WHY NOT A FIXED HOUR WINDOW (e.g. 06-17 UTC year-round): at Bolzano's latitude
(~46.5N) daylight runs from ~8h (December) to ~15.5h (June), so a fixed window
either clips real daytime in summer or counts pre-dawn/post-dusk hours as "day"
in winter. Boundary-layer mixing -- and hence the T2-vs-elevation lapse rate --
is driven by solar heating, not by clock time, so the day/night split should
track the sun, not a fixed UTC range.
"""

import numpy as np
import pandas as pd

BOLZANO_LAT = 46.4983
BOLZANO_LON = 11.3548


def sun_times_utc(dates, lat=BOLZANO_LAT, lon=BOLZANO_LON):
    """Return (sunrise_hour_utc, sunset_hour_utc) as fractional-UTC-hour arrays,
    one pair per date in `dates` (array-like of datetime64/Timestamp -- only the
    calendar date is used, time-of-day is ignored)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    doy = dates.dayofyear.values.astype(np.float64)

    gamma = 2 * np.pi / 365.0 * (doy - 1)   # fractional year, radians

    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))   # minutes
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))   # radians

    lat_r = np.radians(lat)
    cos_ha = (np.cos(np.radians(90.833)) / (np.cos(lat_r) * np.cos(decl))
              - np.tan(lat_r) * np.tan(decl))
    cos_ha = np.clip(cos_ha, -1.0, 1.0)   # guards polar day/night; never hit at this latitude
    ha_deg = np.degrees(np.arccos(cos_ha))

    solar_noon_min = 720.0 - 4.0 * lon - eqtime   # UTC minutes-from-midnight
    sunrise_min = solar_noon_min - 4.0 * ha_deg
    sunset_min  = solar_noon_min + 4.0 * ha_deg
    return sunrise_min / 60.0, sunset_min / 60.0


def classify_day_night(times, lat=BOLZANO_LAT, lon=BOLZANO_LON):
    """Vectorized 'day'/'night' classification of a UTC timestamp array, using
    each timestamp's OWN calendar date's real sunrise/sunset (see sun_times_utc)
    -- i.e. the window shifts smoothly day by day across the year, not just
    season by season."""
    times = pd.DatetimeIndex(pd.to_datetime(times))
    sunrise_h, sunset_h = sun_times_utc(times, lat=lat, lon=lon)
    hour_frac = times.hour.values + times.minute.values / 60.0
    is_day = (hour_frac >= sunrise_h) & (hour_frac < sunset_h)
    return np.where(is_day, 'day', 'night')
