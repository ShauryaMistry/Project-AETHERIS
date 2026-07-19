"""
==========================================================
AETHERIS
Utility Functions

Shared helper functions used throughout the project.
==========================================================
"""

import json
import time
from pathlib import Path


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name):
    return str(name).replace(" ", "_")


def save_json(data, filepath):
    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(filepath):
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def format_seconds(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}m {seconds}s"


def depth_to_ppm(depth):
    return depth * 1_000_000


def duration_to_hours(duration_days):
    return duration_days * 24


def duration_to_minutes(duration_days):
    return duration_days * 24 * 60


class Timer:

    def __init__(self):
        self.start_time = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start_time

    def elapsed_string(self):
        return format_seconds(self.elapsed())


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def candidate_summary(candidate):

    depth_ppm = depth_to_ppm(candidate["depth"])
    duration_hours = duration_to_hours(candidate["duration"])

    return f"""
Period        : {candidate['period']:.6f} days
Duration      : {duration_hours:.2f} hours
Transit Time  : {candidate['transit_time']:.6f}
Depth         : {depth_ppm:.1f} ppm
BLS Power     : {candidate['power']:.8f}
"""


def file_exists(filepath):
    return Path(filepath).exists()
