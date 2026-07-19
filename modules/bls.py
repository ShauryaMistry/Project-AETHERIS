"""
==========================================================
AETHERIS
Box Least Squares (BLS) Transit Search

Searches a light curve for periodic transit signals.
==========================================================
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares

from config import (
    MIN_PERIOD,
    MAX_PERIOD,
    N_PERIODS,
    MIN_DURATION,
    MAX_DURATION,
    N_DURATIONS,
    OVERSAMPLE,
)


class BLSAnalyzer:

    def __init__(self, lightcurve):

        self.lc = lightcurve
        self.time = lightcurve.time.value
        self.flux = lightcurve.flux.value

        self.model = BoxLeastSquares(self.time, self.flux)

    def build_grid(self):

        periods = np.linspace(MIN_PERIOD, MAX_PERIOD, N_PERIODS)
        durations = np.linspace(MIN_DURATION, MAX_DURATION, N_DURATIONS)

        return periods, durations

    def search(self):

        print("\nRunning Box Least Squares search...")

        periods, durations = self.build_grid()

        results = self.model.power(periods, durations, oversample=OVERSAMPLE)

        print("BLS search complete.")

        return results

    @staticmethod
    def best_candidate(results):

        index = np.argmax(results.power)

        return {
            "period": float(results.period[index]),
            "duration": float(results.duration[index]),
            "transit_time": float(results.transit_time[index]),
            "depth": float(results.depth[index]),
            "power": float(results.power[index]),
            "index": int(index),
        }


def run_bls(lightcurve):

    analyzer = BLSAnalyzer(lightcurve)

    return analyzer.search()


def get_best_candidate(results):

    return BLSAnalyzer.best_candidate(results)
