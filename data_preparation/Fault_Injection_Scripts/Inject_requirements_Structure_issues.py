# Inject structural incompleteness into the input SRS content with ISO/IEC/IEEE 29148:2018 Standard Template with 33% missing tags, 33% jumbled template, 34% both missing tags and jumbled template.
# Set your path appropriately based on your input data directory.

import os
import json
import random
import re
import pandas as pd

# Define Base Paths
BASE_FOLDER = r"D:\Datasets\Thesis_data"
OUTPUT_BASE_FOLDER = r"D:\Datasets\Thesis_data\structural_incompleteness"

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# Function to Load JSON Files
def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename.startswith("Complete"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                json_files.append((filename, data))
    return json_files

# Function to Apply Swaps with Logging
def apply_swaps(json_data, swap_log, filename, num_swaps):
    for sec_key, sec_content in json_data.items():
        if isinstance(sec_content, dict):
            subsec_keys = list(sec_content.keys())
            if len(subsec_keys) > 1:
                for _ in range(min(num_swaps, len(subsec_keys) - 1)):
                    i = random.randint(0, len(subsec_keys) - 2)
                    subsec_keys[i], subsec_keys[i + 1] = subsec_keys[i + 1], subsec_keys[i]
                    swap_log.append([filename, "Swap", f"Swapped {subsec_keys[i]} with {subsec_keys[i + 1]} in {sec_key}"])
                json_data[sec_key] = {k: sec_content[k] for k in subsec_keys}
    return json_data

# Function to Remove Section Names While Keeping Numbering
def remove_section_names(json_data, missing_log, filename):
    missing_count = 0
    intro_section = json_data.get("[SEC] 1. Introduction", {})
    definitions = intro_section.get("[SUBSEC] 1.8 Definitions", [])
    acronyms = intro_section.get("[SUBSEC] 1.9 Acronyms and Abbreviations", [])

    grouped_sections = {"1.x": [], "1.3.x": [], "2.x": []}

    for sec_key in ["[SEC] 1. Introduction", "[SEC] 2. Requirements"]:
        if sec_key in json_data and isinstance(json_data[sec_key], dict):
            for subsec_key, subsec_content in json_data[sec_key].items():
                if subsec_key.startswith("[SUBSEC] 1.3") and isinstance(subsec_content, dict):
                    for subsubsec_key in subsec_content:
                        grouped_sections["1.3.x"].append((sec_key, subsec_key, subsubsec_key))
                elif subsec_key.startswith("[SUBSEC] 1.") and subsec_key not in ["[SUBSEC] 1.8 Definitions", "[SUBSEC] 1.9 Acronyms and Abbreviations"] and not subsec_key.startswith("[SUBSEC] 1.3"):
                    grouped_sections["1.x"].append((sec_key, subsec_key))
                elif subsec_key.startswith("[SUBSEC] 2."):
                    grouped_sections["2.x"].append((sec_key, subsec_key))

    for category, sections in grouped_sections.items():
        if sections:
            if category == "1.3.x":
                sec_key, subsec_key, subsubsec_key = random.choice(sections)
                match = re.search(r'(\d+\.\d+(\.\d+)?)', subsubsec_key)
                section_number = match.group(1) if match else "UNKNOWN"
                new_key = f"[SUBSUBSEC] {section_number} MISSING-TAG{missing_count+1}"
                json_data[sec_key][subsec_key][new_key] = json_data[sec_key][subsec_key].pop(subsubsec_key)
                missing_log.append([filename, "RemoveTag", f"Removed {subsubsec_key}, Replaced with {new_key}"])
            else:
                sec_key, subsec_key = random.choice(sections)
                match = re.search(r'(\d+\.\d+(\.\d+)?)', subsec_key)
                section_number = match.group(1) if match else "UNKNOWN"
                new_key = f"[SUBSEC] {section_number} MISSING-TAG{missing_count+1}"
                json_data[sec_key][new_key] = json_data[sec_key].pop(subsec_key)
                missing_log.append([filename, "RemoveTag", f"Removed {subsec_key}, Replaced with {new_key}"])
            missing_count += 1

    json_data["[SEC] 1. Introduction"] = intro_section
    return json_data

# Function to Process Structural Folders (33% swap-only, 33% missing-only, 34% both)
def process_structural_folders():
    subfolders = ["Infer", "Train", "Val"]
    for subfolder in subfolders:
        input_folder = os.path.join(BASE_FOLDER, subfolder)
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, subfolder)
        excel_log_file = os.path.join(OUTPUT_BASE_FOLDER, f"{subfolder}_structural_log.xlsx")

        os.makedirs(output_folder, exist_ok=True)
        log_data = []
        inference_files = load_json_files(input_folder)
        total_files = len(inference_files)
        random.shuffle(inference_files)

        split1 = total_files // 3
        split2 = (total_files * 2) // 3
        split3 = total_files

        swap_only_files = inference_files[:split1]
        tag_only_files = inference_files[split1:split2]
        both_files = inference_files[split2:split3]

        for filename, json_data in swap_only_files:
            swap_log = []
            modified_json = apply_swaps(json_data, swap_log, filename, 2)

            file_number = filename.split("_")[1].split(".")[0]
            output_filename = f"incomplete_srs_{file_number}_00001.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(modified_json, file, indent=2, ensure_ascii=False)

            log_data.extend(swap_log)

        for filename, json_data in tag_only_files:
            missing_log = []
            modified_json = remove_section_names(json_data, missing_log, filename)

            file_number = filename.split("_")[1].split(".")[0]
            output_filename = f"incomplete_srs_{file_number}_00001.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(modified_json, file, indent=2, ensure_ascii=False)

            log_data.extend(missing_log)

        for filename, json_data in both_files:
            swap_log = []
            missing_log = []

            modified_json = apply_swaps(json_data, swap_log, filename, 2)
            modified_json = remove_section_names(modified_json, missing_log, filename)

            file_number = filename.split("_")[1].split(".")[0]
            output_filename = f"incomplete_srs_{file_number}_00001.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(modified_json, file, indent=2, ensure_ascii=False)

            log_data.extend(swap_log + missing_log)

        log_df = pd.DataFrame(log_data, columns=["Filename", "Issue Type", "Details"])
        log_df.to_excel(excel_log_file, index=False, engine="openpyxl")
        print(f"{subfolder} processing complete! Logs saved.")

# Run the Program
process_structural_folders()
print("\n Structural Processing Complete!")
