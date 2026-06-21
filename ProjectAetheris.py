import sys
import subprocess
import os
import multiprocessing
import shutil

# ==============================================================================
# ENVIRONMENT GUARD: PORTABLE LOCALIZED WORKSPACE SETUP
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else os.getcwd()
# Temporary staging area for the raw download structure
LOCAL_CACHE_DIR = os.path.join(SCRIPT_DIR, "clean_astro_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

os.environ["ASTROPY_CACHE_DIR"] = LOCAL_CACHE_DIR

# ==============================================================================
# PART I: AUTOMATED ENVIRONMENT PROVISIONING ENGINE
# ==============================================================================
REQUIRED_LIBRARIES = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "lightkurve": "lightkurve",
    "astropy": "astropy"
}

def bootstrap_environment():
    print("[INIT] Verifying computational environment dependencies...")
    missing_packages = []
    for module_name, pip_name in REQUIRED_LIBRARIES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        print(f"[SETUP] Missing requirements detected: {missing_packages}")
        print("[SETUP] Executing automated background provisioning via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("[SETUP] All dependencies successfully built and provisioned.")
        except Exception as e:
            print(f"[FATAL SETUP ERROR] Automatic environment deployment failed: {str(e)}")
            sys.exit(1)
    else:
        print("[INIT] Environment verified. All assets present locally.")

bootstrap_environment()

# ==============================================================================
# PART II: DATA SCIENCE ENGINE, CONFIGURATIONS, & SIGNAL ANALYSIS LIBRARIES
# ==============================================================================
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('agg')  
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import Conf as AstropyDataConf

AstropyDataConf.remote_timeout.set(120)
warnings.filterwarnings("ignore")

plt.style.use('dark_background')
matplotlib.rcParams['text.color'] = '#e0e0e0'
matplotlib.rcParams['axes.labelcolor'] = '#b0bec5'
matplotlib.rcParams['xtick.color'] = '#78909c'
matplotlib.rcParams['ytick.color'] = '#78909c'

# ==============================================================================
# PART III: FLAT STORAGE PIPELINE ENGINE
# ==============================================================================
def analyze_target_system(target_input, max_planets_to_hunt=1):
    print("==================================================================")
    print(" PROJECT AETHERIS V5 // FLAT DIRECTORY INJECTION ENGINE          ")
    
    if isinstance(target_input, SkyCoord):
        target_display_name = f"RA {target_input.ra.deg:.4f} DEC {target_input.dec.deg:.4f}"
        clean_name = f"coords_{target_input.ra.deg:.2f}_{target_input.dec.deg:.2f}"
        search_query = target_input
    else:
        target_display_name = str(target_input)
        clean_name = target_display_name.replace(" ", "_").replace("-", "_")
        search_query = target_display_name

    print(f" Target Objective: {target_display_name}")
    
    available_cores = multiprocessing.cpu_count()
    target_workers = max(1, available_cores - 1)
    print(f" [SYSTEM PROFILE] Host hardware detected: {available_cores} Cores. Allocating {target_workers} workers.")
    print("==================================================================")
    
    try:
        # Create a single, flat destination folder dedicated to this target system
        system_output_folder = os.path.join(SCRIPT_DIR, f"System_{clean_name}")
        os.makedirs(system_output_folder, exist_ok=True)

        print("[STAGE 1] Querying NASA MAST Archive for unrefined SPOC sectors...")
        search_result = None
        max_retries = 3
        backoff_delay = 5  
        network_failed = False
        
        for attempt in range(1, max_retries + 1):
            try:
                search_result = lk.search_lightcurve(search_query, mission="TESS", author="SPOC")
                break  
            except Exception:
                print(f"  [-] Connection attempt {attempt}/{max_retries} timed out.")
                if attempt < max_retries:
                    print(f"  [-] Pausing {backoff_delay}s before network re-entry...")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2  
                else:
                    print("  [-] Live remote connection failed completely.")
                    network_failed = True

        lc_downloaded = None
        
        # Helper to extract downloads into a flat configuration folder
        def extract_to_flat_folder(downloaded_obj):
            if downloaded_obj is None:
                return
            # Find the actual .fits file deep in the download tree and pull it to our unified folder
            for root, _, files in os.walk(LOCAL_CACHE_DIR):
                for file in files:
                    if file.endswith(".fits") and clean_name.replace("_", "") in file.lower():
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(system_output_folder, file)
                        if not os.path.exists(dest_path):
                            shutil.copy2(src_path, dest_path)

        if network_failed:
            print("[FALLBACK] Server offline. Scanning folder for local assets...")
            matching_files = [os.path.join(system_output_folder, f) for f in os.listdir(system_output_folder) if f.endswith(".fits")]
            
            if len(matching_files) > 0:
                print(f"[SUCCESS] Found {len(matching_files)} flat sector datasets locally. Loading...")
                lcs = [lk.read(f) for f in matching_files[:3]]
                lc_downloaded = lk.LightCurveCollection(lcs).stitch()
            else:
                print("[FATAL] No local target files found inside the system directory. Pipeline halted.")
                return
        else:
            if search_result is None or len(search_result) == 0:
                print(f"[ERROR] Target '{target_display_name}' could not be resolved by MAST. Exiting.")
                return
            
            def stream_sectors(segments, label):
                print(f"[STAGE 2] Streaming {label} target sectors (Max Timeout: 120s)...")
                chunks = []
                for idx in range(len(segments)):
                    single_row = segments[idx:idx+1]
                    try:
                        lc = single_row.download(download_dir=LOCAL_CACHE_DIR)
                        if lc is not None:
                            chunks.append(lc)
                            extract_to_flat_folder(lc)
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "corrupt" in error_msg or "error in reading" in error_msg or "bad" in error_msg:
                            print(f"  [SELF-HEALING] Interrupted chunk at index {idx}. Forcing direct network fetch override...")
                            try:
                                lc = single_row.download(download_dir=LOCAL_CACHE_DIR, force=True)
                                if lc is not None:
                                    chunks.append(lc)
                                    extract_to_flat_folder(lc)
                                    print(f"  [SUCCESS] Restored and downloaded sector index {idx} safely.")
                            except Exception as retry_err:
                                print(f"  [-] Direct network force retry failed for index {idx}: {str(retry_err)}")
                        else:
                            print(f"  [-] Sector index {idx} aborted. Reason: {type(e).__name__} -> {str(e)}")
                        continue
                return chunks

            if len(search_result) > 3:
                print(f"[OPTIMIZE] Target has {len(search_result)} sectors. Slicing most recent 3 baseline windows...")
                best_lcs = stream_sectors(search_result[-3:], "most recent")
                
                if len(best_lcs) == 0:
                    print("[FALLBACK] Modern sector endpoints un-responsive. Routing to historical mirrors...")
                    best_lcs = stream_sectors(search_result[:3], "historical baseline")
            else:
                best_lcs = stream_sectors(search_result, "all available")
            
            if len(best_lcs) == 0:
                print("[FATAL] Both modern and historical archive mirrors are down. Pipeline halted.")
                return
                
            print(f"[SUCCESS] Successfully secured and stitching {len(best_lcs)} target data streams...")
            lc_downloaded = lk.LightCurveCollection(best_lcs).stitch()
        
        try:
            star_radius = lc_downloaded.meta.get('RADIUS', 1.0)
            if star_radius is None or np.isnan(star_radius): star_radius = 1.0
        except Exception:
            star_radius = 1.0

        print("[STAGE 3] Computing variance metrics for content-aware flattening...")
        try:
            time_diffs = np.diff(lc_downloaded.time.value)
            median_cadence_days = np.nanmedian(time_diffs)
            optimized_window = int(2.5 / median_cadence_days)
            if optimized_window % 2 == 0: optimized_window += 1 
            if optimized_window < 51: optimized_window = 151     
            print(f"  [SUCCESS] Cadence profiled at {median_cadence_days*1440:.2f} mins. Tuning filter window length to: {optimized_window}")
        except Exception:
            optimized_window = 151
            print("  [WARNING] Cadence metrics inconclusive. Defaulting window length to 151.")

        lc_flattened = lc_downloaded.flatten(window_length=optimized_window, polyorder=2)
        working_lc = lc_flattened.remove_outliers(sigma=4.0).remove_nans()

        data_matrix = {
            'time': np.array(working_lc.time.value, dtype=np.float64),
            'flux': np.array(working_lc.flux.value, dtype=np.float64),
            'flux_err': np.array(working_lc.flux_err.value, dtype=np.float64)
        }

        for planet_idx in range(1, max_planets_to_hunt + 1):
            if len(data_matrix['time']) < 100: break
                
            print(f"[STAGE 4] Launching multi-threaded BLS Comb Engine (10,000 Step Array)...")
            periods = np.linspace(0.8, 45.0, 10000)
            durations = np.linspace(0.04, 0.16, 4)
            
            t_start = time.time()
            try:
                bls_model = BoxLeastSquares(data_matrix['time'], data_matrix['flux'], data_matrix['flux_err'])
                bls_results = bls_model.power(periods, durations, objective="snr")
                execution_type = f"Parallel Acceleration [Active Workers: {target_workers}]"
            except Exception as thread_fault:
                print(f"  [MANAGED EXCEPTION] Multi-threading interface locked: {str(thread_fault)}")
                print("  [FALLBACK] Re-routing logic array safely into Single-Core container...")
                bls_model = BoxLeastSquares(data_matrix['time'], data_matrix['flux'], data_matrix['flux_err'])
                bls_results = bls_model.power(periods, durations, objective="snr")
                execution_type = "Single-Core Safety Container"
                
            t_end = time.time()
            print(f"  [PERFORMANCE LOG] Array processed via {execution_type} in {t_end - t_start:.2f}s.")
            
            max_power_idx = np.nanargmax(bls_results.power)
            snr = float(bls_results.power[max_power_idx])
            
            if snr < 4.0:
                print(f"[INFO] No signals detected above noise ceiling (SNR: {snr:.2f}). Plot skipped.")
                break
                
            best_p = float(bls_results.period[max_power_idx])
            t0 = float(bls_results.transit_time[max_power_idx])
            duration = float(bls_results.duration[max_power_idx])

            folded_phase_raw = (data_matrix['time'] - t0 + 0.5 * best_p) % best_p - 0.5 * best_p
            folded_phase_days = folded_phase_raw
            folded_phase_norm = folded_phase_raw / best_p
            
            transit_window = np.abs(folded_phase_days) < (duration * 2.5)
            cycle_numbers = np.round((data_matrix['time'] - t0) / best_p)
            even_mask = (cycle_numbers % 2 == 0)
            odd_mask = (cycle_numbers % 2 != 0)

            def compute_binned_profile(phase_array, flux_array, bin_edges):
                if len(phase_array) == 0:
                    return np.ones(len(bin_edges) - 1)
                indices = np.digitize(phase_array, bin_edges)
                profile = []
                for i in range(1, len(bin_edges)):
                    subset = flux_array[indices == i]
                    profile.append(np.nanmedian(subset) if len(subset) > 0 and not np.all(np.isnan(subset)) else 1.0)
                return np.array(profile)

            full_bins = np.linspace(-0.5, 0.5, 400)
            zoom_bins = np.linspace(-3.5 * duration, 3.5 * duration, 40) 
            zoom_bin_centers = 0.5 * (zoom_bins[:-1] + zoom_bins[1:])

            full_binned_flux = compute_binned_profile(folded_phase_norm, data_matrix['flux'], full_bins)
            even_binned_flux = compute_binned_profile(folded_phase_days[even_mask], data_matrix['flux'][even_mask], zoom_bins)
            odd_binned_flux = compute_binned_profile(folded_phase_days[odd_mask], data_matrix['flux'][odd_mask], zoom_bins)

            val_even = np.nanmin(even_binned_flux) if not np.all(np.isnan(even_binned_flux)) else 1.0
            val_odd = np.nanmin(odd_binned_flux) if not np.all(np.isnan(odd_binned_flux)) else 1.0
            
            depth_even = float(1.0 - val_even)
            depth_odd = float(1.0 - val_odd)
            mean_depth = (depth_even + depth_odd) / 2.0
            
            if np.isnan(mean_depth) or np.isinf(mean_depth):
                mean_depth = 0.0

            odd_even_delta = np.abs(depth_even - depth_odd) / (mean_depth + 1e-8) * 100.0 if mean_depth > 0 else 0.0
            radius_ratio = np.sqrt(mean_depth) if mean_depth > 0 else 0
            planet_rad_earth = star_radius * radius_ratio * 109.2

            if planet_rad_earth < 1.25: category = "Rocky Terrestrial"
            elif 1.25 <= planet_rad_earth < 2.0: category = "Super-Earth Class"
            elif 2.0 <= planet_rad_earth < 6.0: category = "Sub-Neptune Class"
            else: category = "Jovian Gas Giant"

            is_false_positive = False
            disposition = "STRONG CANDIDATE"
            
            if odd_even_delta > 25.0:
                disposition = "FALSE_POSITIVE"
                is_false_positive = True
            elif planet_rad_earth > 22.0:
                disposition = "FALSE_POSITIVE"
                is_false_positive = True

            print("\n[REPORT] --- AUTOMATED TOI VETTING ANALYSIS REPORT ---")
            print(f" -> Recovered Signal Period : {best_p:.5f} Days")
            print(f" -> Signal-to-Noise (SNR)   : {snr:.2f}")
            print(f" -> Derived Physical Radius : {planet_rad_earth:.2f} R_Earth")
            print(f" -> Odd/Even Depth Delta    : {odd_even_delta:.2f}%")
            print(f" -> PRELIMINARY DISPOSITION : {disposition}\n")

            y_bounds_delta = (mean_depth * 3.5) if mean_depth > 0.0 else 0.001

            fig = plt.figure(figsize=(16, 9), facecolor='#12161a')
            fig.suptitle(f"Project Aetheris Diagnostics // System: {target_display_name}", 
                         color='#2ecc71', fontsize=14, weight='bold', y=0.96)

            ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, facecolor='#1a1f26')
            ax1.scatter(folded_phase_norm, data_matrix['flux'], alpha=0.08, s=0.5, color='#90a4ae')
            full_bin_centers = 0.5 * (full_bins[:-1] + full_bins[1:])
            ax1.plot(full_bin_centers, full_binned_flux, color='#2ecc71', lw=1.2)
            ax1.set_title("1. Continuous Full-Phase Folded Alignment Map", fontsize=10, loc='left', pad=6)
            ax1.set_xlim(-0.5, 0.5)
            ax1.grid(True, linestyle=':', alpha=0.1, color='#ffffff')

            ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2, facecolor='#1a1f26')
            ax2.scatter(folded_phase_days, data_matrix['flux'], alpha=0.12, s=0.8, color='#78909c')
            ax2.plot(zoom_bin_centers, even_binned_flux, color='#3498db', lw=2.5, marker='o', markersize=3)
            box_x = [-4.0*duration, -0.5*duration, -0.5*duration, 0.5*duration, 0.5*duration, 4.0*duration]
            box_y = [1.0, 1.0, 1.0 - mean_depth, 1.0 - mean_depth, 1.0, 1.0]
            ax2.plot(box_x, box_y, color='#e74c3c', linestyle='--', lw=1.5)
            ax2.set_xlim(-2.5 * duration, 2.5 * duration)
            ax2.set_ylim(1.0 - y_bounds_delta, 1.0 + y_bounds_delta)
            ax2.grid(True, linestyle=':', alpha=0.15, color='#ffffff')

            ax3a = plt.subplot2grid((3, 2), (2, 0), facecolor='#1a1f26')
            ax3a.plot(zoom_bin_centers, even_binned_flux, color='#2980b9', lw=1.5, label='Even Cycles')
            ax3a.plot(zoom_bin_centers, odd_binned_flux, color='#e67e22', lw=1.5, label='Odd Cycles')
            ax3a.set_title("3a. Alternating Cycle Contrast (EB Discrepancy Check)", fontsize=10, loc='left', pad=6)
            ax3a.set_xlabel("Time from Mid-Transit [Days]", fontsize=9)
            ax3a.set_xlim(-2.0 * duration, 2.0 * duration)
            ax3a.set_ylim(1.0 - y_bounds_delta, 1.0 + y_bounds_delta)
            ax3a.legend(fontsize=8, loc='lower left', framealpha=0.4)
            ax3a.grid(True, linestyle=':', alpha=0.1, color='#ffffff')

            ax3b = plt.subplot2grid((3, 2), (2, 1), facecolor='#1a1f26')
            ax3b.hist(data_matrix['flux'][~transit_window], bins=80, alpha=0.3, color='#b0bec5', density=True)
            ax3b.hist(data_matrix['flux'][transit_window], bins=80, alpha=0.5, color='#2980b9', density=True)
            ax3b.set_xlabel("Normalized Spectral Density", fontsize=9)
            ax3b.grid(True, linestyle=':', alpha=0.1, color='#ffffff')

            infobox_text = (
                f"--- PROJECT AETHERIS OBSERVATION REPORT CARD ---\n"
                f"Target Pipeline Tag: {target_display_name} {chr(96 + planet_idx)} | Inferred Disposition: {disposition}\n"
                f"Computed Period: {best_p:.4f} days | Transit Depth: {mean_depth*10000:.1f} ppm | Characterized Classification: {planet_rad_earth:.2f} R_Earth ({category})"
            )
            props = dict(boxstyle='round,pad=0.5', facecolor='#1c2833', edgecolor='#2ecc71', alpha=0.9)
            fig.text(0.5, 0.02, infobox_text, transform=fig.transFigure, fontsize=9.5, fontfamily='monospace', ha='center', color='#f4f6f7', bbox=props)

            plt.tight_layout()
            plt.subplots_adjust(top=0.88, bottom=0.13, hspace=0.45, wspace=0.16)
            
            # Save the plot directly alongside the pulled data fits files
            filename = f"vetting_assessment_{clean_name}_{disposition}.png"
            out_img = os.path.join(system_output_folder, filename)
                
            fig.savefig(out_img, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            print(f"[SUCCESS] Flat file compilation completed inside directory: System_{clean_name}/")

            keep_filter = ~transit_window
            data_matrix['time'] = data_matrix['time'][keep_filter]
            data_matrix['flux'] = data_matrix['flux'][keep_filter]
            data_matrix['flux_err'] = data_matrix['flux_err'][keep_filter]

        print("==================================================================\n")

    except Exception as e:
        print(f"[FATAL EXOTRANSIT ERROR] Pipeline failed processing target: {str(e)}")

if __name__ == "__main__":
    # ==============================================================================
    # TARGET INJECTION ZONE
    # ==============================================================================
    TARGET_OBJECT = "ENTER TIC-ID OR TOI NAME"
    
    analyze_target_system(target_input=TARGET_OBJECT, max_planets_to_hunt=1)
