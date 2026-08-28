"""Split the SD19 by_class checksum log into one hash list per digit."""

import os
from pathlib import Path


def split_hashes_by_digit(log_path, output_dir):
    """
    Reads a log file with <hash> <path> lines.
    Extracts hashes for digits 0–9 (from class_id 30–39).
    Writes them to <output_dir>/zero.log, one.log, ..., nine.log
    """
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Mapping from class_id (30–39) to digit (0–9)
    valid_class_ids = {str(i): i - 30 for i in range(30, 40)}
    log_filenames = [
        "zero.log", "one.log", "two.log", "three.log", "four.log",
        "five.log", "six.log", "seven.log", "eight.log", "nine.log"
    ]

    # Open log files for writing
    out_files = {
        i: open(output_dir / log_filenames[i], "w")
        for i in range(10)
    }

    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            hash_value, path = parts
            if "/by_class/" not in path:
                continue

            class_id = path.split("/by_class/")[1].split("/")[0]
            if class_id in valid_class_ids:
                digit = valid_class_ids[class_id]
                out_files[digit].write(hash_value + "\n")

    # Close everything
    for f in out_files.values():
        f.close()
