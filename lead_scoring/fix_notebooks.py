from pathlib import Path
import nbformat
from nbformat.validator import normalize

count = 0

for nb_file in Path(r"C:\Users\kuchbhe\Desktop\workspace_1\LeadScoringML\lead_scoring\experiments").rglob("*.ipynb"):
    try:
        nb = nbformat.read(nb_file, as_version=4)
        _, nb = normalize(nb)
        nbformat.write(nb, nb_file)
        count += 1
        print(f"Fixed: {nb_file}")
    except Exception as e:
        print(f"ERROR: {nb_file} -> {e}")

print(f"\nProcessed {count} notebooks")