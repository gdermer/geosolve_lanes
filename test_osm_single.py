import osmnx as ox

# use alternative Overpass servers
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.timeout      = 60
ox.settings.log_console  = True

lat = -46.459483
lon = 168.627837

print(f"Testing with alternative Overpass server...")

try:
    G = ox.graph_from_bbox(
        bbox         = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05),
        network_type = "drive",
        simplify     = True
    )
    print(f"Success! Nodes: {len(G.nodes):,}")
    edges = ox.graph_to_gdfs(G, nodes=False)
    lane_cols = [c for c in ["highway", "lanes", "oneway", "lanes:forward"]
                 if c in edges.columns]
    print(edges[lane_cols].head(5).to_string())

except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")