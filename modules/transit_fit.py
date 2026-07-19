"""
==========================================================
AETHERIS
Transit Characterization Module

Refines and characterizes the strongest transit candidate.

Future versions will include:
    * MCMC transit fitting
    * Inclination
    * Impact parameter
==========================================================
"""

import numpy as np


class TransitFitter:

    def __init__(self, results):

        self.results = results
        self.best = np.argmax(results.power)

    def period(self):
        return float(self.results.period[self.best])

    def duration(self):
        return float(self.results.duration[self.best])

    def transit_time(self):
        return float(self.results.transit_time[self.best])

    def depth(self):
        return float(self.results.depth[self.best])

    def power(self):
        return float(self.results.power[self.best])

    def depth_ppm(self):
        return self.depth() * 1_000_000

    def duration_hours(self):
        return self.duration() * 24

    def duration_minutes(self):
        return self.duration() * 24 * 60

    def candidate(self):

        return {
            "period": self.period(),
            "duration": self.duration(),
            "duration_hours": self.duration_hours(),
            "duration_minutes": self.duration_minutes(),
            "transit_time": self.transit_time(),
            "depth": self.depth(),
            "depth_ppm": self.depth_ppm(),
            "power": self.power(),
        }


def characterize_transit(results):

    fitter = TransitFitter(results)

    return fitter.candidate()
