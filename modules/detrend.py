"""
==========================================================
AETHERIS
Detrending Module

Removes long-term stellar variability from light curves.

Current Methods
---------------
- Median flattening (Lightkurve)

Future Methods
--------------
- Savitzky-Golay
- Gaussian Process
- Cotrending Basis Vectors (CBVs)
- Polynomial fitting
==========================================================
"""

from config import FLATTEN_WINDOW


class Detrender:

    def __init__(self, lightcurve):

        self.lc = lightcurve

    def flatten(self):

        print("\nDetrending light curve...")

        detrended = self.lc.flatten(window_length=FLATTEN_WINDOW)

        print("Detrending complete.")

        return detrended

    def savgol(self):
        raise NotImplementedError("Savitzky-Golay detrending not implemented yet.")

    def gaussian_process(self):
        raise NotImplementedError("Gaussian Process detrending not implemented yet.")

    def cbv(self):
        raise NotImplementedError("CBV detrending not implemented yet.")


def detrend_lightcurve(lightcurve):

    detrender = Detrender(lightcurve)

    return detrender.flatten()
