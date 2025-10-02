import os

results_dir = r"D:\SSD Performance\ssd_project3_kit\results"

for fname in os.listdir(results_dir):
    if fname.lower().endswith(".json"):
        fpath = os.path.join(results_dir, fname)

        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print(f"[SKIP] Empty file: {fname}")
            continue

        # If first line is not JSON (doesn't start with '{'), drop it
        if not lines[0].lstrip().startswith("{"):
            print(f"[FIX] Removing first line from {fname}")
            new_lines = lines[1:]
            with open(fpath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        else:
            print(f"[OK] {fname} already valid JSON")
