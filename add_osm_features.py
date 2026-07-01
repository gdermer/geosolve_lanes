import osmnx as ox
import pandas as pd
import numpy as np
import time
from pathlib import Path

# use alternative Overpass server (office network blocks default)
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.timeout      = 300
ox.settings.log_console  = False
ox.settings.use_cache    = True
ox.settings.cache_folder = "Data/osm_cache"


# cache_folder saves API responses to disk
# so the same area is never queried twice
# Java equivalent: a HashMap that persists to disk

Path("Data/osm_cache").mkdir(exist_ok=True)

# ================================================================
# COORDINATE PARSING
# ================================================================

def parse_coordinate(coord_str, hemisphere):
    """
    Convert DDMM.MMMM format to decimal degrees.
    e.g. "4627.5690S" → -46.4595
    """
    try:
        coord_clean = str(coord_str).replace(hemisphere, "").strip()
        coord_float = float(coord_clean)
        degrees     = int(coord_float / 100)
        minutes     = coord_float - (degrees * 100)
        decimal     = degrees + (minutes / 60)
        if hemisphere == "S":
            decimal = -decimal
        return decimal
    except:
        return None


# ================================================================
# OSM CACHE — stores road data for grid squares
# ================================================================

# divide NZ into 0.1° × 0.1° grid squares (~10km each)
# when we need a GPS point, we load its grid square
# each grid square is only loaded once then cached in memory

road_cache = {}
# dictionary: "lat_grid,lon_grid" → GeoDataFrame of roads
# Java equivalent: HashMap<String, GeoDataFrame>

def get_grid_key(lat, lon, grid_size=0.1):
    """Get the grid square key for a GPS point."""
    lat_grid = round(int(lat / grid_size) * grid_size, 2)
    lon_grid = round(int(lon / grid_size) * grid_size, 2)
    return f"{lat_grid},{lon_grid}"

def get_roads_for_point(lat, lon):
    """
    Get road network for the grid square containing this point.
    Uses in-memory cache to avoid repeated API calls.
    """
    grid_key = get_grid_key(lat, lon)

    if grid_key not in road_cache:
        # not in cache — fetch from OSM
        try:
            lat_grid = float(grid_key.split(",")[0])
            lon_grid = float(grid_key.split(",")[1])

            # download roads for this 0.1° grid square
            G = ox.graph_from_bbox(
                bbox=(lon_grid, lat_grid, lon_grid + 0.1, lat_grid + 0.1),
                network_type="drive",
                simplify=True
            )
            edges = ox.graph_to_gdfs(G, nodes=False)
            road_cache[grid_key] = edges

        except Exception as e:
            # no roads in this area or API error
            road_cache[grid_key] = None

    return road_cache[grid_key]


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def get_osm_features(lat, lon):
    """
    Get lanes_forward, oneway, road_type for a GPS point.
    """
    try:
        edges = get_roads_for_point(lat, lon)

        if edges is None or len(edges) == 0:
            return {"lanes_forward": 1, "oneway": 0, "road_type": "unknown"}

        from shapely.geometry import Point
        point  = Point(lon, lat)
        buffer = point.buffer(0.001)
        # 0.001 degrees ≈ 100 metres

        nearby = edges[edges.geometry.intersects(buffer)]

        if len(nearby) == 0:
            return {"lanes_forward": 1, "oneway": 0, "road_type": "unknown"}

        nearest = nearby.iloc[0]

        # lanes_forward
        lanes_forward_explicit = nearest.get("lanes:forward", None)
        if lanes_forward_explicit is not None:
            try:
                lanes_forward = int(float(str(lanes_forward_explicit)))
            except:
                lanes_forward = None

        if lanes_forward_explicit is None or lanes_forward is None:
            raw_lanes = nearest.get("lanes", None)
            if isinstance(raw_lanes, list):
                raw_lanes = raw_lanes[0]
            try:
                total_lanes = int(float(str(raw_lanes)))
            except:
                total_lanes = 2

            is_oneway = bool(nearest.get("oneway", False))
            if is_oneway:
                lanes_forward = total_lanes
            else:
                lanes_forward = max(1, total_lanes // 2)

        # oneway
        is_oneway = int(bool(nearest.get("oneway", False)))

        # road type
        highway = nearest.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]
        road_type_map = {
            "motorway": "motorway", "motorway_link": "motorway",
            "trunk": "primary", "trunk_link": "primary",
            "primary": "primary", "primary_link": "primary",
            "secondary": "secondary", "secondary_link": "secondary",
            "tertiary": "secondary", "tertiary_link": "secondary",
            "residential": "residential", "living_street": "residential",
            "unclassified": "residential", "service": "residential",
        }
        road_type = road_type_map.get(str(highway).lower(), "unknown")

        return {
            "lanes_forward": int(lanes_forward),
            "oneway":        is_oneway,
            "road_type":     road_type,
        }

    except Exception as e:
        return {"lanes_forward": 1, "oneway": 0, "road_type": "unknown"}


# ================================================================
# PROCESS CSV
# ================================================================

def process_csv(input_path, output_path, sample_size=None):
    print(f"\nProcessing {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

    if sample_size:
        df = df.head(sample_size)
        print(f"Using sample: {sample_size:,} rows")

    print(f"Total rows: {len(df):,}")

    df["lanes_forward"] = 1
    df["oneway"]        = 0
    df["road_type"]     = "unknown"

    start_time   = time.time()
    success      = 0
    failed       = 0
    cache_hits   = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            lat = parse_coordinate(str(row.get("Latitude",  "0")), "S")
            lon = parse_coordinate(str(row.get("Longitude", "0")), "E")

            if lat is None or lon is None:
                failed += 1
                continue

            grid_key    = get_grid_key(lat, lon)
            was_cached  = grid_key in road_cache

            features = get_osm_features(lat, lon)

            df.at[idx, "lanes_forward"] = features["lanes_forward"]
            df.at[idx, "oneway"]        = features["oneway"]
            df.at[idx, "road_type"]     = features["road_type"]
            success += 1

            if was_cached:
                cache_hits += 1

        except Exception as e:
            failed += 1

        if (i + 1) % 1000 == 0:
            elapsed   = time.time() - start_time
            rate      = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate / 60
            pct       = (i + 1) / len(df) * 100
            cache_size = len(road_cache)
            print(f"  {i+1:,}/{len(df):,} ({pct:.1f}%) | "
                  f"Rate: {rate:.0f}/s | "
                  f"Remaining: {remaining:.1f}m | "
                  f"Cache: {cache_size} squares | "
                  f"Failed: {failed}")

    df.to_csv(output_path, index=False)
    elapsed = time.time() - start_time
    print(f"\nDone! {success:,} enriched, {failed:,} failed")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} mins)")
    print(f"Cache squares used: {len(road_cache)}")
    print(f"Saved: {output_path}")

    print(f"\nlanes_forward distribution:")
    print(df["lanes_forward"].value_counts().sort_index().to_string())
    print(f"\nroad_type distribution:")
    print(df["road_type"].value_counts().to_string())

    return df


if __name__ == "__main__":
    print("=" * 50)
    print("ADD OSM FEATURES — Overpass API approach")
    print("=" * 50)

    # ---- test on 100 rows first ----
    print("\nSTEP 1 — Test on 100 rows")
    df_test = process_csv(
        input_path  = "Data/train_small.csv",
        output_path = "Data/test_osm_100.csv",
        sample_size = 100
    )

    print("\nSample output:")
    cols = ["Latitude", "Longitude", "Bearing",
            "lanes_forward", "oneway", "road_type", "Lane"]
    print(df_test[cols].head(10).to_string())

    non_default = (df_test["lanes_forward"] != 1).sum()
    print(f"\nRows with non-default lanes_forward: {non_default}/100")

    # ---- process real files ----
    print("\nSTEP 2 — Process train_small.csv (229K rows)")
    print("Estimated time: 30-60 minutes")
    process_csv(
        input_path  = "Data/train_small.csv",
        output_path = "Data/train_osm.csv",
    )

    print("\nSTEP 3 — Process val.csv (sample of 5K)")
    process_csv(
        input_path  = "Data/val.csv",
        output_path = "Data/val_osm.csv",
        sample_size = 5402
    )

    print("\nSTEP 4 — Create tiny test set (500 rows)")
    process_csv(
        input_path  = "Data/train_small.csv",
        output_path = "Data/train_tiny_osm.csv",
        sample_size = 500
    )

    print("\nALL DONE!")
    print("Update config.py:")
    print("  TRAIN_CSV = DATA_DIR / 'train_osm.csv'")
    print("  VAL_CSV   = DATA_DIR / 'val_osm.csv'")