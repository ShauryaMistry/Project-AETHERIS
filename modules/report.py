"""
==========================================================
AETHERIS
Report Module

Generates scientific reports for detected candidates.
==========================================================
"""

from datetime import datetime

from config import (
    PROJECT_NAME,
    VERSION,
    AUTHOR,
    REPORT_DIR,
    REPORT_TITLE,
)


class ReportGenerator:

    def __init__(self, target):

        self.target = target.replace(" ", "_")
        self.output_dir = REPORT_DIR / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, candidate):

        lines = []

        lines.append("=" * 60)
        lines.append(REPORT_TITLE)
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Generated : {datetime.now()}")
        lines.append(f"Software  : {PROJECT_NAME}")
        lines.append(f"Version   : {VERSION}")
        lines.append(f"Author    : {AUTHOR}")
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Target : {self.target}")
        lines.append("-" * 60)
        lines.append("")
        lines.append("Detected Candidate")
        lines.append("")
        lines.append(f"Orbital Period     : {candidate['period']:.6f} days")
        lines.append(f"Transit Duration   : {candidate['duration']:.6f} days")
        lines.append(f"Transit Midpoint   : {candidate['transit_time']:.6f}")
        lines.append(f"Transit Depth      : {candidate['depth']:.8f}")
        lines.append(f"BLS Power          : {candidate['power']:.8f}")

        diag = candidate.get("diagnostics", {})

        if diag:
            lines.append("")
            lines.append("-" * 60)
            lines.append("Diagnostics")
            lines.append("-" * 60)
            lines.append(f"SNR                : {diag.get('snr', 0):.2f}")
            lines.append(f"Transit Count      : {diag.get('transit_count', 0)}")

            oe = diag.get("odd_even", {})
            if oe.get("odd_depth") is not None:
                lines.append(
                    f"Odd/Even Sigma     : {oe['difference_sigma']:.2f} "
                    f"{'(FLAGGED)' if oe.get('flag') else ''}"
                )

        if candidate.get("planet_radius"):
            pr = candidate["planet_radius"]
            lines.append("")
            lines.append("-" * 60)
            lines.append("Estimated Planet Radius")
            lines.append("-" * 60)
            lines.append(f"Earth Radii        : {pr['radius_earth']:.2f}")
            lines.append(f"Jupiter Radii      : {pr['radius_jupiter']:.3f}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("Rule-Based Vetting")
        lines.append("-" * 60)
        lines.append(f"Confidence         : {candidate.get('confidence', 'N/A')} %")
        lines.append(f"Classification     : {candidate.get('classification', 'N/A')}")

        if candidate.get("ml_probability_planet") is not None:
            lines.append("")
            lines.append("-" * 60)
            lines.append("ML Classifier")
            lines.append("-" * 60)
            lines.append(
                f"Probability Planet : "
                f"{candidate['ml_probability_planet'] * 100:.1f} %"
            )
            lines.append(f"Label              : {candidate['ml_label']}")
            lines.append(f"Explanation        : {candidate['ml_explanation']}")

            if candidate.get("is_anomaly"):
                lines.append("Anomaly Flag       : YES - unusual feature profile")

        lines.append("")
        lines.append("-" * 60)
        lines.append("Interpretation")
        lines.append("-" * 60)
        lines.append("")

        depth_ppm = candidate["depth"] * 1_000_000

        lines.append(
            f"The strongest periodic signal detected has an orbital "
            f"period of {candidate['period']:.4f} days."
        )
        lines.append(f"The estimated transit depth is {depth_ppm:.1f} ppm.")
        lines.append(
            "This candidate should be validated using additional "
            "diagnostics such as centroid analysis and comparison "
            "with known exoplanet catalogs."
        )

        lines.append("")
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, candidate):

        report = self.build(candidate)

        filepath = self.output_dir / "report.txt"

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(report)

        print(f"Saved report : {filepath}")


def generate_report(target, candidate):

    ReportGenerator(target).save(candidate)
