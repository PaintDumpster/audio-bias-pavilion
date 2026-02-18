"""Creates recording folder tree inside data/recordings."""

import os
from typing import Iterable, List


DAYS_OF_WEEK: List[str] = [
	"monday",
	"tuesday",
	"wednesday",
	"thursday",
	"friday",
	"saturday",
	"sunday",
]

DAY_PERIODS: List[str] = [
	"morning",
	"midday",
	"evening",
	"night",
]

# Update this list to match the neighborhoods or sites you are surveying.
RECORDING_ZONES: List[str] = [
	"Ciutat vella",
	"Sants-montjuic",
	"Gracia",
	"Les corts",
	"Eixample",
	"Sarria-sant gervasi",
	"San marti",
	"Nou barris",
	"Horta-guinardo",
	"Sant andreu"
]


def create_recording_structure(base_path: str, zones: Iterable[str]) -> None:
	"""Creates the day/period folder hierarchy for every zone name."""

	os.makedirs(base_path, exist_ok=True)

	for zone in zones:
		if not zone:
			continue

		zone_folder = os.path.join(base_path, zone)
		if os.path.isdir(zone_folder):
			print(f"Skipping existing folder: {zone_folder}")
			continue

		os.makedirs(zone_folder, exist_ok=True)
		for day in DAYS_OF_WEEK:
			day_folder = os.path.join(zone_folder, day)
			os.makedirs(day_folder, exist_ok=True)
			for period in DAY_PERIODS:
				os.makedirs(os.path.join(day_folder, period), exist_ok=True)

		print(f"Created folder tree for: {zone_folder}")


def main(zone_names: Iterable[str]) -> None:
	"""Entry point used by the CLI to build folders under data/recordings."""

	base_dir = os.path.join("data", "recordings")
	create_recording_structure(base_dir, zone_names)


if __name__ == "__main__":
	import sys

	# Use CLI arguments if provided, otherwise fall back to the list above.
	selected_zones = sys.argv[1:] or RECORDING_ZONES
	if not selected_zones:
		print("Add zone names as CLI arguments or inside RECORDING_ZONES.")
		sys.exit(1)

	main(selected_zones)
