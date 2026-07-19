"""
==========================================================
AETHERIS
Candidate Vetting Module

Rule-based sanity check on a detected transit signal. This
runs alongside (not instead of) the learned classifier in
ml_classifier.py -- disagreement between the two is itself
worth surfacing in a demo.

Tests
-----
* Transit depth
* Orbital period
* Transit duration
* BLS power
* Signal-to-noise ratio
* Transit count
* Odd/even depth consistency

Future Tests
------------
* Secondary eclipse search
* Centroid shift
* Gaia contamination
==========================================================
"""

from config import (
    MIN_DEPTH,
    MIN_PERIOD,
    MAX_PERIOD,
    MIN_SNR,
    MIN_TRANSITS,
)


class CandidateVetter:

    def __init__(self, candidate, diagnostics=None):

        self.candidate = candidate
        self.diagnostics = diagnostics or {}

        self.score = 0
        self.max_score = 7
        self.notes = []

    def check_depth(self):

        if self.candidate["depth"] >= MIN_DEPTH:
            self.score += 1
            self.notes.append(" Transit depth acceptable")
        else:
            self.notes.append(" Transit depth too shallow")

    def check_period(self):

        p = self.candidate["period"]

        if MIN_PERIOD <= p <= MAX_PERIOD:
            self.score += 1
            self.notes.append(" Period within search limits")
        else:
            self.notes.append(" Unusual orbital period")

    def check_duration(self):

        duration_hours = self.candidate["duration_hours"]

        if 0.5 <= duration_hours <= 15:
            self.score += 1
            self.notes.append(" Transit duration reasonable")
        else:
            self.notes.append(" Transit duration suspicious")

    def check_power(self):

        if self.candidate["power"] > 0:
            self.score += 1
            self.notes.append(" Positive BLS detection")
        else:
            self.notes.append(" Weak BLS signal")

    def check_snr(self):

        snr = self.diagnostics.get("snr", 0.0)

        if snr >= MIN_SNR:
            self.score += 1
            self.notes.append(f" SNR acceptable ({snr:.1f})")
        else:
            self.notes.append(f" SNR too low ({snr:.1f})")

    def check_transit_count(self):

        count = self.diagnostics.get("transit_count", 0)

        if count >= MIN_TRANSITS:
            self.score += 1
            self.notes.append(f" Sufficient transits observed ({count})")
        else:
            self.notes.append(f" Too few transits observed ({count})")

    def check_odd_even(self):

        oe = self.diagnostics.get("odd_even", {})

        if oe.get("odd_depth") is None:
            self.notes.append(" Odd/even test inconclusive (too few transits)")
            return

        if oe.get("flag"):
            self.notes.append(
                f" Odd/even depth mismatch "
                f"({oe.get('difference_sigma', 0):.1f} sigma) "
                f"- possible eclipsing binary"
            )
        else:
            self.score += 1
            self.notes.append(" Odd/even transit depths consistent")

    def confidence(self):

        return round(100 * self.score / self.max_score, 1)

    def classification(self):

        c = self.confidence()

        if c >= 90:
            return "Excellent Candidate"
        elif c >= 75:
            return "Likely Candidate"
        elif c >= 50:
            return "Possible Candidate"

        return "Poor Candidate"

    def run(self):

        print("\nRunning candidate vetting...")

        self.check_depth()
        self.check_period()
        self.check_duration()
        self.check_power()
        self.check_snr()
        self.check_transit_count()
        self.check_odd_even()

        self.candidate["confidence"] = self.confidence()
        self.candidate["classification"] = self.classification()
        self.candidate["vetting_notes"] = self.notes
        self.candidate["diagnostics"] = self.diagnostics

        print("Vetting complete.")

        return self.candidate


def vet_candidate(candidate, diagnostics=None):

    vetter = CandidateVetter(candidate, diagnostics)

    return vetter.run()
