# download_osm.py
import pyrosm
from pathlib import Path

PBF_PATH    = Path("Data/new-zealand-latest.osm.pbf")
OUTPUT_PATH = Path("Data/nz_roads_edges.gpkg")

# bounding box covering South Island + Wellington only
# (min_lon, min_lat, max_lon, max_lat)
BOUNDING_BOX = (166.0, -47.0, 175.0, -40.5)

print(f"Reading {PBF_PATH} with bounding box filter...")
print(f"Covering South Island + Wellington only")

osm = pyrosm.OSM(str(PBF_PATH), bounding_box=list(BOUNDING_BOX))

print("Extracting drivable roads...")
edges = osm.get_network(network_type="driving")
print(f"Edges: {len(edges):,}")

print(f"\nAvailable columns:")
print(edges.columns.tolist())

lane_cols = [c for c in ["highway", "lanes", "oneway", "lanes:forward"]
             if c in edges.columns]
print(f"\nSample lanes data:")
print(edges[lane_cols].head(10).to_string())

print(f"\nSaving to {OUTPUT_PATH}...")
edges.to_file(OUTPUT_PATH, driver="GPKG")
print(f"Saved! Size: {OUTPUT_PATH.stat().st_size/1024/1024:.1f} MB")
print("\nDone!")