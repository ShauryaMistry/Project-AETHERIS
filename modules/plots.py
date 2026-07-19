"""
==========================================================
AETHERIS
Plotting Module

Creates a professional ExoFOP-style vetting dashboard.
Uses SciPy to generate a dependency-free, smoothed U-shaped transit model.
Compatible with Streamlit UI by returning the figure object.
==========================================================
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import gaussian_filter1d

from config import (
    FIGURE_DIR,
    FIG_WIDTH,
    FIG_HEIGHT,
    FIG_DPI,
    FIG_FORMAT,
    SAVE_PLOTS,
)


class Plotter:

    def __init__(self, target):

        self.target = target.replace(" ", "_")
        self.output_dir = FIGURE_DIR / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename):

        if not SAVE_PLOTS:
            return

        filepath = self.output_dir / f"{filename}.{FIG_FORMAT}"
        plt.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
        print(f"Saved figure -> {filepath}")

    def dashboard(self, lc, results, candidate, diagnostics):
        """
        Generates a professional 5-panel TESS SPOC-style layout.
        Returns the matplotlib figure for Streamlit compatibility.
        """
        period = candidate["period"]
        
        # Robustly extract the transit epoch (center time)
        epoch = candidate.get("epoch", candidate.get("transit_time", candidate.get("t0")))
        if epoch is None and hasattr(results, "transit_time_at_max_power"):
            epoch = results.transit_time_at_max_power
            if hasattr(epoch, "value"):
                epoch = epoch.value
        if epoch is None:
            epoch = lc.time.value[0] # Fallback

        duration_days = candidate.get("duration_hours", 2.0) / 24.0
        depth = candidate.get("depth_ppm", 1000) / 1e6

        # Set up a highly controlled grid layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        fig.suptitle(f"Aetheris Data Validation Report - {self.target}", fontsize=20, fontweight="bold", y=0.97)

        # ---------------------------------------------------------
        # 1. Row 1: Full Detrended Light Curve (Spans both columns)
        # ---------------------------------------------------------
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(lc.time.value, lc.flux.value, ".", color="black", markersize=1.5, alpha=0.7)
        
        # --- Generate the U-Shaped Transit Model ---
        t_min, t_max = np.min(lc.time.value), np.max(lc.time.value)
        t_model = np.linspace(t_min, t_max, 10000) 
        
        phases_model = (t_model - epoch) % period
        phases_model = np.where(phases_model > period / 2, phases_model - period, phases_model)
        
        # Create a raw box model
        model_flux = np.ones_like(t_model)
        in_transit = np.abs(phases_model) <= (duration_days / 2.0)
        model_flux[in_transit] -= depth
        
        # Smooth it into a U-shape using SciPy (simulating ingress/egress and limb darkening)
        dt = t_model[1] - t_model[0]
        ingress_days = duration_days * 0.15 # Approx 15% of transit is ingress
        sigma = ingress_days / dt
        
        if sigma > 0:
            model_flux = gaussian_filter1d(model_flux, sigma=sigma)
            
        # Rescale so the bottom of the curve exactly hits our calculated depth
        current_depth = 1.0 - np.min(model_flux)
        if current_depth > 0:
            model_flux = 1.0 - ((1.0 - model_flux) * (depth / current_depth))
        
        # Plot the smoothed model line
        ax_full.plot(t_model, model_flux, color="red", linewidth=2, alpha=0.8, label="Smoothed Transit Model")
        
        ax_full.set_xlabel("Time (days)")
        ax_full.set_ylabel("Normalized Flux")
        ax_full.set_title("Full Detrended Light Curve", fontsize=14)
        ax_full.legend(loc="lower right")
        ax_full.grid(alpha=0.3)

        # ---------------------------------------------------------
        # 2. Row 2, Col 1: BLS Periodogram
        # ---------------------------------------------------------
        ax_bls = fig.add_subplot(gs[1, 0])
        ax_bls.plot(results.period, results.power, color="royalblue", linewidth=1.5)
        ax_bls.axvline(period, color="red", linestyle="--", label=f"Best: {period:.4f} d")
        ax_bls.set_xlabel("Period (days)")
        ax_bls.set_ylabel("BLS Power")
        ax_bls.set_title("Box Least Squares Periodogram", fontsize=14)
        ax_bls.legend(loc="upper right")
        ax_bls.grid(alpha=0.3)

        # ---------------------------------------------------------
        # 3. Row 2, Col 2: Global Phase-Folded Light Curve
        # ---------------------------------------------------------
        folded = lc.fold(period=period, epoch_time=epoch)
        binned = folded.bin(time_bin_size=0.01)
        
        ax_fold = fig.add_subplot(gs[1, 1])
        ax_fold.plot(folded.phase.value, folded.flux.value, ".", color="lightgray", markersize=3, alpha=0.5, label="Raw Folded")
        ax_fold.plot(binned.phase.value, binned.flux.value, "o", color="black", markersize=4, label="Binned Model")
        
        ax_fold.set_xlim(-0.5, 0.5)
        ax_fold.set_xlabel("Orbital Phase")
        ax_fold.set_ylabel("Normalized Flux")
        ax_fold.set_title(f"Global Phase Fold (P = {period:.5f} d)", fontsize=14)
        ax_fold.legend(loc="lower right")
        ax_fold.grid(alpha=0.3)

        # ---------------------------------------------------------
        # 4. Row 3, Col 1: Zoomed Primary Transit
        # ---------------------------------------------------------
        ax_zoom = fig.add_subplot(gs[2, 0])
        ax_zoom.plot(folded.phase.value, folded.flux.value, ".", color="lightgray", markersize=4, alpha=0.8)
        
        binned_fine = folded.bin(time_bin_size=0.002)
        ax_zoom.plot(binned_fine.phase.value, binned_fine.flux.value, "o", color="red", markersize=5, markeredgecolor="black")
        
        # --- Generate U-Shaped Model for the Zoomed View ---
        phase_zoom = np.linspace(-0.05, 0.05, 1000)
        zoom_flux = np.ones_like(phase_zoom)
        
        in_transit_zoom = np.abs(phase_zoom) <= (duration_days / period / 2.0)
        zoom_flux[in_transit_zoom] -= depth
        
        dp = phase_zoom[1] - phase_zoom[0]
        ingress_phase = (duration_days * 0.15) / period
        sigma_zoom = ingress_phase / dp
        
        if sigma_zoom > 0:
            zoom_flux = gaussian_filter1d(zoom_flux, sigma=sigma_zoom)
            
        c_depth = 1.0 - np.min(zoom_flux)
        if c_depth > 0:
            zoom_flux = 1.0 - ((1.0 - zoom_flux) * (depth / c_depth))
            
        ax_zoom.plot(phase_zoom, zoom_flux, color="darkred", linewidth=2, label="Smoothed Model Fit")
        ax_zoom.legend(loc="lower right")

        ax_zoom.set_xlim(-0.05, 0.05)
        ax_zoom.set_xlabel("Orbital Phase")
        ax_zoom.set_ylabel("Normalized Flux")
        ax_zoom.set_title("Primary Transit (Zoomed)", fontsize=14)
        ax_zoom.grid(alpha=0.3)

        # ---------------------------------------------------------
        # 5. Row 3, Col 2: Zoomed Secondary Eclipse Check
        # ---------------------------------------------------------
        phase_shifted = np.where(folded.phase.value < 0, folded.phase.value + 1.0, folded.phase.value)
        binned_shifted = np.where(binned_fine.phase.value < 0, binned_fine.phase.value + 1.0, binned_fine.phase.value)
        
        ax_sec = fig.add_subplot(gs[2, 1])
        ax_sec.plot(phase_shifted, folded.flux.value, ".", color="lightgray", markersize=4, alpha=0.8)
        ax_sec.plot(binned_shifted, binned_fine.flux.value, "o", color="orange", markersize=5, markeredgecolor="black")
        
        ax_sec.set_xlim(0.45, 0.55)
        ax_sec.set_ylim(ax_zoom.get_ylim()) 
        
        ax_sec.set_xlabel("Orbital Phase")
        ax_sec.set_ylabel("Normalized Flux")
        ax_sec.set_title("Secondary Eclipse Check (Phase 0.5)", fontsize=14)
        ax_sec.grid(alpha=0.3)

        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
        
        self.save("dv_report_dashboard")
        return fig

# ======================================================
# Public API
# ======================================================

def plot_dashboard(lc, results, candidate, diagnostics, target):
    return Plotter(target).dashboard(lc, results, candidate, diagnostics)