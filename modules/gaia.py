"""
==========================================================
AETHERIS
Unified Metadata Query Module (SIMBAD + Gaia DR3)

Resolves targets using SIMBAD name cross-matching, matches
coordinates to Gaia DR3, and pulls critical astrometric, 
photometric, and astrophysical parameters.
==========================================================
"""

import numpy as np
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

from config import (
    DEFAULT_ST_RADIUS,
    DEFAULT_ST_TEFF,
    DEFAULT_ST_LOGG,
)

SOLAR_RADIUS_EARTH = 109.2
SOLAR_RADIUS_JUPITER = 9.73


class GaiaQuery:

    def __init__(self, ra, dec):
        self.coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # ======================================================
    # Astrometry / photometry
    # ======================================================

    def query_astrometry(self):
        try:
            job = Gaia.cone_search_async(self.coord, radius=5 * u.arcsec)
            table = job.get_results()

            if len(table) == 0:
                print("No Gaia source found.")
                return {}

            star = table[0]

            # Helper to safely parse potentially masked Gaia values
            def safe_float(val):
                return float(val) if val is not None and not np.ma.is_masked(val) else np.nan

            return {
                "source_id": int(star["source_id"]),
                "ra": safe_float(star["ra"]),
                "dec": safe_float(star["dec"]),
                "parallax": safe_float(star["parallax"]),
                "parallax_error": safe_float(star["parallax_error"]),
                "g_mag": safe_float(star["phot_g_mean_mag"]),
                "bp_mag": safe_float(star["phot_bp_mean_mag"]),
                "rp_mag": safe_float(star["phot_rp_mean_mag"]),
                "bp_rp": safe_float(star["bp_rp"]),
            }

        except Exception as e:
            print(f"Gaia astrometry query failed: {e}")
            return {}

    # ======================================================
    # Astrophysical parameters (radius, Teff, logg)
    # ======================================================

    def query_astrophysical_parameters(self):
        defaults = {
            "radius": DEFAULT_ST_RADIUS,
            "teff": DEFAULT_ST_TEFF,
            "logg": DEFAULT_ST_LOGG,
        }

        try:
            radius_deg = 5 / 3600.0

            query = f"""
                SELECT ap.radius_gspphot, ap.teff_gspphot, ap.logg_gspphot
                FROM gaiadr3.astrophysical_parameters AS ap
                JOIN gaiadr3.gaia_source AS gs USING (source_id)
                WHERE 1=CONTAINS(
                    POINT('ICRS', gs.ra, gs.dec),
                    CIRCLE('ICRS', {self.coord.ra.deg}, {self.coord.dec.deg}, {radius_deg})
                )
            """

            job = Gaia.launch_job(query)
            table = job.get_results()

            if len(table) == 0:
                print("No astrophysical parameters found, using defaults.")
                return defaults

            row = table[0]

            radius = row["radius_gspphot"]
            teff = row["teff_gspphot"]
            logg = row["logg_gspphot"]

            return {
                "radius": float(radius) if radius is not None and not np.ma.is_masked(radius) else DEFAULT_ST_RADIUS,
                "teff": float(teff) if teff is not None and not np.ma.is_masked(teff) else DEFAULT_ST_TEFF,
                "logg": float(logg) if logg is not None and not np.ma.is_masked(logg) else DEFAULT_ST_LOGG,
            }

        except Exception as e:
            print(f"Astrophysical parameter query failed: {e}")
            return defaults

    # ======================================================
    # Combined Gaia query
    # ======================================================

    def query(self):
        print("\nQuerying Gaia DR3...")
        data = self.query_astrometry()

        if not data:
            return {}

        physical = self.query_astrophysical_parameters()
        data.update(physical)

        print("Gaia query complete.")
        return data


def get_gaia_data(ra, dec):
    return GaiaQuery(ra, dec).query()


# ==========================================================
# New Unified Metadata Infrastructure Entry Point
# ==========================================================

def get_stellar_metadata(target_name, ra=None, dec=None):
    """
    Orchestrates target identification across SIMBAD and Gaia DR3.
    """
    print(f"\n🔍 Resolving identity metadata via SIMBAD for target: '{target_name}'...")
    
    # Configure and run SIMBAD query
    sb = Simbad()
    sb.add_votable_fields('sptype')
    
    sb_table = None
    try:
        sb_table = sb.query_object(target_name)
    except Exception as e:
        print(f"⚠️ SIMBAD query connection error: {e}")

    # Fallback assignment setups
    main_id = target_name
    sp_type = "N/A"
    
    if sb_table is not None and len(sb_table) > 0:
        main_id = sb_table['MAIN_ID'][0]
        # Clean bytes types if returned by astroquery
        if 'SP_TYPE' in sb_table.colnames and sb_table['SP_TYPE'][0]:
            raw_sp = sb_table['SP_TYPE'][0]
            sp_type = raw_sp.decode('utf-8') if isinstance(raw_sp, bytes) else str(raw_sp)
        
        # Use coordinates from SIMBAD if they aren't provided by Lightkurve
        if ra is None or dec is None:
            coord = SkyCoord(ra=sb_table['RA'][0], dec=sb_table['DEC'][0], unit=(u.hourangle, u.deg), frame='icrs')
            ra = coord.ra.deg
            dec = coord.dec.deg
            
        print(f"✅ SIMBAD Resolved: '{main_id}' [Spectral Type: {sp_type}]")
    else:
        print("⚠️ SIMBAD could not resolve designation; continuing with raw target name.")

    # Execute Gaia lookup using coordinate space mappings
    if ra is not None and dec is not None:
        gaia_profile = get_gaia_data(ra, dec)
    else:
        print("❌ Coordinate cross-match omitted: Missing spatial coordinates.")
        gaia_profile = {}
        
    # Append SIMBAD structural strings to the dictionary returned to main.py
    gaia_profile['simbad_id'] = main_id
    gaia_profile['spectral_type'] = sp_type
    
    return gaia_profile


def estimate_planet_radius(depth, stellar_radius_solar):
    """
    Rp = R_star * sqrt(depth)

    Returns planet radius in Earth radii and Jupiter radii.
    """
    rp_solar = stellar_radius_solar * np.sqrt(max(depth, 0.0))

    return {
        "radius_earth": float(rp_solar * SOLAR_RADIUS_EARTH),
        "radius_jupiter": float(rp_solar * SOLAR_RADIUS_JUPITER),
    }