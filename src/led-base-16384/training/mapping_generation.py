import os
import json
import random

# ✅ Define Base Paths for Cluster
BASE_PATH = "/home/gadde/Thesis/data"
OUTPUT_PATH = os.path.join(BASE_PATH, "mappings")

# ✅ Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ✅ Define dataset categories and their respective folders
DATASETS = {
    "train": "train/easy",
    "val": "val/easy",
    "test": "test/easy",
}

CATEGORIES = ["structural_incompleteness", "requirement_coverage", "traceability", "requirement_quality_attributes"]

# ✅ Define dataset splits for complete files
TRAIN_COMPLETE_RANGE = range(1, 165)
VAL_COMPLETE_RANGE = range(165, 209)
TEST_COMPLETE_RANGE = range(209, 220)

# ✅ Function to scan folders and create mappings
def generate_mappings(dataset_type, shuffle=False):
    mappings = []
    mapping_id = 1
    complete_srs_path = os.path.join(BASE_PATH, "complete")
    
    # ✅ Add mappings for incomplete SRS files in the dataset
    for category in CATEGORIES:
        category_path = os.path.join(BASE_PATH, DATASETS[dataset_type], category)

        if os.path.exists(category_path):
            for filename in sorted(os.listdir(category_path)):  # Ensuring consistent ordering
                if filename.startswith("incomplete_srs") and filename.endswith(".json"):
                    file_number = "_".join(filename.split("_")[2:4])  # Extracting full file number
                    input_path = f"data/{dataset_type}/easy/{category}/{filename}"
                    output_path = f"data/complete/complete_srs_{file_number.split('_')[0]}.json"

                    mappings.append({
                        "mapping_id": mapping_id,
                        "input": input_path,
                        "output": output_path
                    })
                    mapping_id += 1

    # ✅ Special handling for training dataset (shuffle with repeatable randomness)
    if shuffle:
        random.seed(42)  # Ensures repeatability
        random.shuffle(mappings)

    # ✅ Add complete SRS mappings only within the dataset split
    if os.path.exists(complete_srs_path):
        for filename in sorted(os.listdir(complete_srs_path)):
            if filename.startswith("complete_srs") and filename.endswith(".json"):
                file_number = int(filename.split("_")[2].split(".")[0])  # Ensure only the number is extracted correctly
                complete_path = f"data/complete/{filename}"
                
                # Ensure complete file is added only to its respective dataset
                if (dataset_type == "train" and file_number in TRAIN_COMPLETE_RANGE) or \
                   (dataset_type == "val" and file_number in VAL_COMPLETE_RANGE) or \
                   (dataset_type == "test" and file_number in TEST_COMPLETE_RANGE):
                    mappings.append({
                        "mapping_id": mapping_id,
                        "input": complete_path,
                        "output": complete_path
                    })
                    mapping_id += 1

    # ✅ Save mappings to JSON file
    output_file = os.path.join(OUTPUT_PATH, f"{dataset_type}_mapping.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4)

    print(f"✅ {dataset_type}_mapping.json generated successfully!")

# ✅ Generate mappings for all datasets
generate_mappings("train", shuffle=True)  # Training set (shuffled)
generate_mappings("val", shuffle=False)   # Validation set (ordered)
generate_mappings("test", shuffle=False)  # Test set (ordered)

print("\n🚀 All mappings generated successfully!")
