"""
==========================================================
AETHERIS
Preprocessing Module

Cleans, normalizes and caches light curves.
==========================================================
"""

import pickle

from config import (
    PROCESSED_DIR,
    USE_PROCESSED_CACHE,
    SAVE_PROCESSED,
    REMOVE_NANS,
    REMOVE_OUTLIERS,
    OUTLIER_SIGMA,
    NORMALIZE,
    FLATTEN,
    FLATTEN_WINDOW,
)


class Preprocessor:

    def __init__(self, target: str):

        self.target = target
        self.cache_file = PROCESSED_DIR / f"{target.replace(' ', '_')}.pkl"

    def cache_exists(self):
        return self.cache_file.exists()

    def load_cache(self):

        print("\nLoading processed light curve...")

        with open(self.cache_file, "rb") as file:
            lc = pickle.load(file)

        print("Processed cache loaded.")

        return lc

    def save_cache(self, lc):

        with open(self.cache_file, "wb") as file:
            pickle.dump(lc, file)

        print(f"Saved processed cache -> {self.cache_file}")

    def process(self, lc):

        print("\nPreprocessing light curve...")

        original_points = len(lc)

        if REMOVE_NANS:
            before = len(lc)
            lc = lc.remove_nans()
            print(f"Removed NaNs       : {before - len(lc)}")

        if REMOVE_OUTLIERS:
            before = len(lc)
            lc = lc.remove_outliers(sigma=OUTLIER_SIGMA)
            print(f"Removed Outliers   : {before - len(lc)}")

        if FLATTEN:
            print("Flattening light curve...")
            lc = lc.flatten(window_length=FLATTEN_WINDOW)

        if NORMALIZE:
            print("Normalizing...")
            lc = lc.normalize()

        print("\nPreprocessing Summary")
        print("------------------------------")
        print(f"Original Points : {original_points}")
        print(f"Remaining Points: {len(lc)}")
        print(f"Removed Total   : {original_points - len(lc)}")

        return lc

    def run(self, lc):

        if USE_PROCESSED_CACHE and self.cache_exists():
            return self.load_cache()

        lc = self.process(lc)

        if SAVE_PROCESSED:
            self.save_cache(lc)

        return lc


def preprocess_lightcurve(lc, target):

    processor = Preprocessor(target)

    return processor.run(lc)
