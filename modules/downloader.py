"""
==========================================================
AETHERIS
TESS Downloader Module

Downloads TESS light curves using Lightkurve.
==========================================================
"""

import lightkurve as lk

from config import MISSION


class TESSDownloader:
    """
    Handles downloading TESS light curves.
    """

    def __init__(self, target):

        self.target = target
        self.search = None

    def search_target(self):

        print(f"\nSearching for {self.target}...")

        self.search = lk.search_lightcurve(
            self.target,
            mission=MISSION,
        )

        if len(self.search) == 0:
            raise ValueError(
                f"No light curve found for '{self.target}'."
            )

        print(f"Found {len(self.search)} products.\n")
        print(self.search[:5])

        return self.search

    def download(self):

        if self.search is None:
            self.search_target()

        print("\nDownloading data...")

        product = self.search[0]
        lc = product.download()

        print(f"Downloaded {len(lc)} observations.")
        print(
            f"Time Range : "
            f"{lc.time.value.min():.2f} - "
            f"{lc.time.value.max():.2f} days"
        )

        return lc, product, self.search


def download_target(target):
    """
    Download a TESS light curve.

    Returns
    -------
    tuple : (LightCurve, SearchResult row, full SearchResult)
    """

    downloader = TESSDownloader(target)
    downloader.search_target()

    return downloader.download()
