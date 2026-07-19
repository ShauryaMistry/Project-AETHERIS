"""
==========================================================
AETHERIS
Output Module

Saves candidate information to disk.

Supported formats:
    * JSON
    * CSV
==========================================================
"""

import csv
import json

from config import (
    OUTPUT_DIR,
    SAVE_CANDIDATE_JSON,
    SAVE_CSV,
)


class OutputManager:

    def __init__(self, target):

        self.target = target.replace(" ", "_")
        self.output_dir = OUTPUT_DIR / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, candidate):

        filepath = self.output_dir / "candidate.json"

        with open(filepath, "w") as file:
            json.dump(candidate, file, indent=4, default=str)

        print(f"Saved candidate : {filepath}")

    def save_csv(self, candidate):

        filepath = self.output_dir / "candidate.csv"

        with open(filepath, "w", newline="") as file:

            writer = csv.writer(file)
            writer.writerow(["Parameter", "Value"])

            for key, value in candidate.items():
                writer.writerow([key, value])

        print(f"Saved CSV : {filepath}")

    def save(self, candidate):

        if SAVE_CANDIDATE_JSON:
            self.save_json(candidate)

        if SAVE_CSV:
            self.save_csv(candidate)


def save_candidate(target, candidate):

    manager = OutputManager(target)

    manager.save(candidate)
