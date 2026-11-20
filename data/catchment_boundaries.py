"""
Step 2: Obtain catchment boundary polygons for the five gauges from the USGS
Network-Linked Data Index (NLDI), check them against the CAMELS-US areas, and
write a shapefile that can be uploaded to Google Earth Engine (Step 3).

Requirements:  pip install dataretrieval geopandas

Outputs (in ./data/basins/):
  study_basins.shp (+ .dbf, .shx, .prj, .cpg)   one polygon per gauge, field 'gauge_id'
  area_check.csv                                  delineated vs. CAMELS area
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from dataretrieval import nldi

# CAMELS-US drainage areas (km^2), camels_topo.txt, column area_gages2.
CAMELS_AREA_KM2 = {
    "11264500": 467.98,
    "14305500": 526.33,
    "01047000": 909.10,
    "04127997": 487.16,
    "06906800": 1414.75,
}
MAX_AREA_DEVIATION_PCT = 10.0  # flag polygons deviating more than this from CAMELS

OUT = Path("data") / "basins"
OUT.mkdir(parents=True, exist_ok=True)

polygons = []
checks = []

FEATURE_SOURCES = ["nwissite", "WQP"]  # streamgage index first, water-quality index as fallback

for gauge, camels_area in CAMELS_AREA_KM2.items():
    print(f"Delineating basin upstream of USGS-{gauge} ...")
    gdf, used_source = None, None
    for source in FEATURE_SOURCES:
        try:
            gdf = nldi.get_basin(feature_source=source, feature_id=f"USGS-{gauge}")
            used_source = source
            break
        except Exception as err:  # e.g. HTTP 404: gauge not indexed in this source
            print(f"  not found via '{source}': {err}")
    if gdf is None:
        print(f"  FAILED for USGS-{gauge}; skipping it (see messages above).")
        checks.append({"gauge_id": gauge, "source": None, "area_nldi_km2": None,
                       "area_camels_km2": camels_area, "deviation_pct": None, "flag": True})
        continue
    print(f"  found via '{used_source}'")
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union

    basin = gpd.GeoDataFrame({"gauge_id": [gauge]}, geometry=[geom], crs=gdf.crs)
    # Area in an equal-area projection for the contiguous US (NAD83 / Conus Albers).
    area_km2 = basin.to_crs(epsg=5070).area.iloc[0] / 1e6
    deviation = 100 * (area_km2 - camels_area) / camels_area

    checks.append({
        "gauge_id": gauge,
        "source": used_source,
        "area_nldi_km2": round(area_km2, 1),
        "area_camels_km2": camels_area,
        "deviation_pct": round(deviation, 1),
        "flag": abs(deviation) > MAX_AREA_DEVIATION_PCT,
    })
    polygons.append(basin.to_crs(epsg=4326))

basins = pd.concat(polygons, ignore_index=True)
basins.to_file(OUT / "study_basins.shp")

check_df = pd.DataFrame(checks)
check_df.to_csv(OUT / "area_check.csv", index=False)
print("\nArea check (delineated vs. CAMELS):")
print(check_df.to_string(index=False))
if check_df["flag"].any():
    print("\nWARNING: at least one polygon deviates strongly from the CAMELS area. "
          "Inspect it before uploading to Earth Engine.")
