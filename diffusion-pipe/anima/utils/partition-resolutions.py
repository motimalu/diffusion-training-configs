import os
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

INPUT_DIR = Path("./1_collection_name3")
RESOLUTIONS = [[1536, 1536], [1280, 1280], [1024, 1024], [768], [512, 512]]
SIZE_DELTA_RATIO = 0.9
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def get_target_dir(width, height):
    """Determines the correct bucket based on image area."""
    image_area = width * height
    for res in RESOLUTIONS:
        if image_area >= (res[0] * res[1] * SIZE_DELTA_RATIO):
            return INPUT_DIR / f"{res[0]}x{res[1]}"
    return None

def process_group(stem, files):
    """Processes a single group of files (image + txts)."""
    image_path = next((f for f in files if f.suffix.lower() in EXTENSIONS), None)
    if not image_path:
        return

    try:
        with Image.open(image_path) as img:
            width, height = img.size
        
        dest_dir = get_target_dir(width, height)
        if dest_dir:
            for file_path in files:
                try:
                    # Atomic OS rename is magnitudes faster than shutil.move for same-drive ops
                    file_path.rename(dest_dir / file_path.name)
                except OSError:
                    # Fallback for cross-drive moves if necessary
                    shutil.move(str(file_path), str(dest_dir / file_path.name))
    except Exception as e:
        print(f"Error processing {stem}: {e}")
        pass

def main():
    print("Pre-creating resolution directories...")
    for res in RESOLUTIONS:
        (INPUT_DIR / f"{res[0]}x{res[1]}").mkdir(parents=True, exist_ok=True)

    print("Scanning directory and mapping file groups...")
    file_groups = defaultdict(list)
    
    for file_path in INPUT_DIR.rglob("*"):
        if file_path.is_file():
            # Drops the extension, then strips "_nl" and anything after it
            stem = file_path.stem 
            stem = re.sub(r'_nl.*$', '', stem)
            
            file_groups[stem].append(file_path)

    print(f"Found {len(file_groups)} groups. Starting parallel move...")
    
    # Increased workers to push I/O limits
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        list(tqdm(executor.map(lambda x: process_group(*x), file_groups.items()), total=len(file_groups)))

    print("Partitioning image resolutions complete.")

if __name__ == "__main__":
    main()