#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared LAPSE_MODE resolution for the elevation-correction rate.

Every one of the six methodology scripts (wrf_uhi/, heatwave2023/, suhi_lcz/,
urbclim/) exposes a per-script CONFIG constant, LAPSE_MODE, choosing between three
ways of correcting a source's temperature field for elevation:

  'none'     -- no elevation correction at all (raw urban-minus-rural difference).
  'constant' -- a single, spatio-temporally fixed rate (LAPSE_RATE_CONSTANT below,
                the classical/standard-atmosphere value), applied everywhere
                regardless of season or day/night.
  'valley'   -- the dynamic, per-(season, day/night) rate fitted from WRF over the
                representative valley domain (seasonal_valley_lapse_rate.py) --
                this is the default and what all six scripts used before this
                toggle was added.

Edit LAPSE_MODE near the top of a script and re-run it to get that mode's figure;
see each script's own OUTPUT_DIR/filename handling for how the mode is tagged onto
outputs so the three don't overwrite each other.
"""

LAPSE_RATE_CONSTANT = 0.0065   # K/m -- classical/standard-atmosphere lapse rate, used
                                 # uniformly (no season/day-night dependence) in 'constant' mode

LAPSE_MODES = ('none', 'constant', 'valley')


def resolve_rate(mode, dynamic_rate):
    """Return the rate (K/m) actually used for the elevation correction, given the
    selected LAPSE_MODE and the dynamic (valley-fitted) rate for the current
    (season, day/night) bucket.

    `dynamic_rate` may be None (e.g. a bucket missing from the CSV) -- in 'valley'
    mode a missing rate resolves to 0.0 (no-op for that bucket), matching the
    existing `if rate is not None else 0.0` guards already used at call sites.
    """
    if mode == 'none':
        return 0.0
    if mode == 'constant':
        return LAPSE_RATE_CONSTANT
    if mode == 'valley':
        return dynamic_rate if dynamic_rate is not None else 0.0
    raise ValueError(f"Unknown LAPSE_MODE {mode!r}, expected one of {LAPSE_MODES}")
