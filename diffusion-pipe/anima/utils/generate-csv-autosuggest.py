from pathlib import Path
from collections import Counter

INPUT_FOLDER = "./1_collection_name3"
OUTPUT_FILENAME = "./kirazuri-3-autosuggest.csv"

def scan_dataset():
    dataset_path = Path(INPUT_FOLDER)
    
    if not dataset_path.exists():
        return None, f"[ERROR] Dataset folder not found: {dataset_path}"
    
    tag_counter = Counter()
    image_count = 0
    
    for txt_file in dataset_path.rglob("*.txt"):
        # Exclude '_nl' sidecar files which contain natual language captions
        if "_nl" in txt_file.name:
            continue
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                tag_counter.update(tags)
                image_count += 1
        except Exception as e:
            print(f"Warning: Could not read file {txt_file}. Error: {e}")
            continue
    
    if image_count == 0:
        return None, f"[ERROR] No caption files found in {dataset_path}"
    
    return tag_counter
        
def main():
    print("Starting tag analysis")
    tag_counter = scan_dataset()
    filtered_counter = Counter({k: v for k, v in tag_counter.items() if v >= 1})

    with open(OUTPUT_FILENAME, "w") as f:
        f.write('tag,category,count,alias')
        for item, count in filtered_counter.most_common():
            f.write(f"{item},5,{count},9999999\n")

    print(f"Autosuggest tag values written to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
