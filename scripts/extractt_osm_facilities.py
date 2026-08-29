from pathlib import Path
import osmium
import csv
import sys
import time

BASE = Path(__file__).resolve().parent.parent
OSM_DIR = BASE / "osm_data"
OUT_DIR = BASE / "processed"

OUT_DIR.mkdir(exist_ok=True)

PBF_FILES = [
    "central-zone-260828.osm.pbf",
    "eastern-zone-260828.osm.pbf",
    "north-eastern-zone-260828.osm.pbf",
    "northern-zone-260828.osm.pbf",
    "southern-zone-260828.osm.pbf",
    "western-zone-260828.osm.pbf",
]

OUTPUT = OUT_DIR / "osm_facilities.csv"


class FacilityHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.rows = []

    def check_tags(self, obj, feature_type):

        tags = {k: v for k, v in obj.tags}

        power = tags.get("power", "")
        industrial = tags.get("industrial", "")
        landuse = tags.get("landuse", "")

        # Only keep genuinely relevant features
        relevant = (
            power in {"plant", "generator", "substation"}
            or industrial not in {"", "no"}
            or landuse == "industrial"
        )

        if not relevant:
            return

        lat = None
        lon = None

        # NODE coordinates
        if feature_type == "node":
            try:
                if obj.location.valid():
                    lat = obj.location.lat
                    lon = obj.location.lon
            except Exception:
                return

        # WAY coordinates: use node reference coordinates only
        elif feature_type == "way":
            try:
                if len(obj.nodes) > 0:
                    node = obj.nodes[len(obj.nodes) // 2]

                    if node.location.valid():
                        lat = node.location.lat
                        lon = node.location.lon
            except Exception:
                return

        if lat is None or lon is None:
            return

        self.rows.append({
            "osm_id": obj.id,
            "feature_type": feature_type,
            "name": tags.get("name", ""),
            "power": power,
            "industrial": industrial,
            "landuse": landuse,
            "operator": tags.get("operator", ""),
            "plant_source": tags.get("plant:source", ""),
            "plant_method": tags.get("plant:method", ""),
            "latitude": lat,
            "longitude": lon
        })

    def node(self, n):
        self.check_tags(n, "node")

    def way(self, w):
        self.check_tags(w, "way")


def main():

    print("=" * 60)
    print("FAST OSM FACILITY EXTRACTION")
    print("=" * 60)

    all_rows = []

    for i, filename in enumerate(PBF_FILES, 1):

        path = OSM_DIR / filename

        if not path.exists():
            print(f"\nMISSING: {filename}")
            continue

        size_mb = path.stat().st_size / (1024 * 1024)

        print(f"\n[{i}/6] Processing: {filename}")
        print(f"Size: {size_mb:.1f} MB")

        start = time.time()

        handler = FacilityHandler()

        try:

            # IMPORTANT:
            # locations=False makes this much faster.
            # We don't build complete geometries.
            handler.apply_file(
                str(path),
                locations=True,
                idx="flex_mem"
            )

            elapsed = time.time() - start

            print(
                f"Facilities found: {len(handler.rows)} "
                f"| Time: {elapsed / 60:.1f} min"
            )

            all_rows.extend(handler.rows)

        except Exception as e:
            print("ERROR:", e)

    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES")
    print("=" * 60)

    unique = {}

    for row in all_rows:
        key = (
            row["feature_type"],
            row["osm_id"]
        )
        unique[key] = row

    rows = list(unique.values())

    print("Total unique facilities:", len(rows))

    if not rows:
        print("No facilities found.")
        sys.exit(1)

    fields = [
        "osm_id",
        "feature_type",
        "name",
        "power",
        "industrial",
        "landuse",
        "operator",
        "plant_source",
        "plant_method",
        "latitude",
        "longitude"
    ]

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n==============================================")
    print("OSM FACILITIES CREATED")
    print("==============================================")
    print("Rows:", len(rows))
    print("Saved:", OUTPUT)

    print("\nFirst 20 facilities:")

    for row in rows[:20]:
        print(
            f"{row['name'] or 'Unnamed'} | "
            f"{row['power']} | "
            f"{row['industrial']} | "
            f"{row['latitude']:.5f}, "
            f"{row['longitude']:.5f}"
        )


if __name__ == "__main__":
    main()