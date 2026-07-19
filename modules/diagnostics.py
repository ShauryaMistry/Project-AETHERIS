"""
==========================================================
AETHERIS
Transit Diagnostics Module

Computes additional vetting diagnostics beyond the core BLS
search:

    * Signal-to-noise ratio (SNR)
    * Observed transit count
    * Odd/even transit depth test
      (the classic eclipsing-binary discriminator)

These feed both vetting.py's rule-based score and
ml_classifier.py's feature set.
==========================================================
"""

import numpy as np

from config import ODD_EVEN_SIGMA_THRESHOLD


class TransitDiagnostics:

    def __init__(self, lc, candidate):

        self.lc = lc
        self.candidate = candidate

        self.time = lc.time.value
        self.flux = lc.flux.value

        self.period = candidate["period"]
        self.duration = candidate["duration"]
        self.t0 = candidate["transit_time"]

    # ======================================================
    # Phase / in-transit mask
    # ======================================================

    def _phase(self):

        phase = (
            (self.time - self.t0 + 0.5 * self.period) % self.period
        ) - 0.5 * self.period

        return phase

    def _in_transit_mask(self):

        phase = self._phase()

        return np.abs(phase) <= (self.duration / 2.0)

    def _cycle_numbers(self):

        return np.round((self.time - self.t0) / self.period)

    # ======================================================
    # Signal-to-noise ratio
    # ======================================================

    def snr(self):

        in_transit = self._in_transit_mask()

        if in_transit.sum() < 2:
            return 0.0

        noise = np.nanstd(self.flux[~in_transit])

        if noise == 0 or np.isnan(noise):
            return 0.0

        n_in = in_transit.sum()
        depth = self.candidate["depth"]

        return float((depth / noise) * np.sqrt(n_in))

    # ======================================================
    # Transit count
    # ======================================================

    def transit_count(self):

        in_transit = self._in_transit_mask()

        cycles = self._cycle_numbers()[in_transit]

        if len(cycles) == 0:
            return 0

        return int(len(np.unique(cycles)))

    # ======================================================
    # Odd/even transit depth test
    # ======================================================

    def odd_even_depth(self):

        in_transit = self._in_transit_mask()

        if in_transit.sum() < 4:
            return {
                "odd_depth": None,
                "even_depth": None,
                "difference_sigma": 0.0,
                "flag": False,
            }

        cycles = self._cycle_numbers()

        odd_mask = in_transit & (np.mod(cycles, 2) == 1)
        even_mask = in_transit & (np.mod(cycles, 2) == 0)

        if odd_mask.sum() < 2 or even_mask.sum() < 2:
            return {
                "odd_depth": None,
                "even_depth": None,
                "difference_sigma": 0.0,
                "flag": False,
            }

        baseline = np.nanmedian(self.flux[~in_transit])

        odd_depth = baseline - np.nanmean(self.flux[odd_mask])
        even_depth = baseline - np.nanmean(self.flux[even_mask])

        odd_err = np.nanstd(self.flux[odd_mask]) / np.sqrt(odd_mask.sum())
        even_err = np.nanstd(self.flux[even_mask]) / np.sqrt(even_mask.sum())

        combined_err = np.sqrt(odd_err ** 2 + even_err ** 2)

        sigma = 0.0

        if combined_err and not np.isnan(combined_err):
            sigma = abs(odd_depth - even_depth) / combined_err

        return {
            "odd_depth": float(odd_depth),
            "even_depth": float(even_depth),
            "difference_sigma": float(sigma),
            "flag": bool(sigma >= ODD_EVEN_SIGMA_THRESHOLD),
        }

    # ======================================================
    # Run all diagnostics
    # ======================================================

    def run(self):

        print("\nRunning transit diagnostics...")

        result = {
            "snr": self.snr(),
            "transit_count": self.transit_count(),
            "odd_even": self.odd_even_depth(),
        }

        print(f"SNR              : {result['snr']:.2f}")
        print(f"Transit Count    : {result['transit_count']}")

        oe = result["odd_even"]

        if oe["odd_depth"] is not None:
            print(f"Odd-Even Sigma   : {oe['difference_sigma']:.2f}")

        print("Diagnostics complete.")

        return result


def run_diagnostics(lc, candidate):

    diag = TransitDiagnostics(lc, candidate)

    return diag.run()
