# Inject tracibility link issues into the input SRS content with removal, duplicate, incorrect tracelinks.
# Set your path appropriately based on your input data directory.

import os
import json
import re
import pandas as pd
import random

# Define Base Paths
BASE_FOLDER = r"D:\Datasets\Thesis_data"
OUTPUT_BASE_FOLDER = r"D:\Datasets\Thesis_data\traceability"

TRACEABILITY_DISTRIBUTION_FILE = r"D:\Datasets\Thesis_data\traceability\SRS_Filewise_Traceability_Distribution.xlsx"

# Regex pattern for traceability links inside `()`
TRACEABILITY_PATTERN = r"\((REQ-[A-Z]+-\d+(?:, REQ-[A-Z]+-\d+)*)\)"

# Define thresholds for categorizing files
HIGH_DENSITY_THRESHOLD = 10
BALANCED_DENSITY_THRESHOLD = 5

# Define swapping rules for traceability links
SWAP_CANDIDATES = {
    "REQ-DB": "REQ-PERF",
    "REQ-FUNC": "REQ-INTF",
    "REQ-INTF": "REQ-FUNC",
    "REQ-PERF": "REQ-DB"
}

# Load Traceability Distribution File
df_traceability_counts = pd.read_excel(TRACEABILITY_DISTRIBUTION_FILE, sheet_name="Filewise Traceability Counts")

# Categorize files based on traceability counts
file_categories = {}
for _, row in df_traceability_counts.iterrows():
    file_name = row["Filename"]
    numeric_values = pd.to_numeric(row[1:], errors='coerce').fillna(0)
    total_links = int(numeric_values.sum())

    if total_links >= HIGH_DENSITY_THRESHOLD:
        file_categories[file_name] = "Mixed"
    elif total_links >= BALANCED_DENSITY_THRESHOLD:
        file_categories[file_name] = "Swapped"
    elif total_links > 0:
        file_categories[file_name] = "Missing"
    else:
        file_categories[file_name] = "No_Modification"

# Function to Process Files in a Folder
def process_traceability_issues(input_folder, output_folder, report_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    modifications_log = []

    # Process all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json") and filename in file_categories:
            input_path = os.path.join(input_folder, filename)

            # New Naming Convention
            file_number = filename.split("_")[1].split(".")[0]  # Extracts "00001"
            output_filename = f"incomplete_srs_{file_number}_00003.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(input_path, "r", encoding="utf-8") as f:
                try:
                    srs_data = json.load(f)
                except json.JSONDecodeError:
                    print(f" Error reading {filename}: Invalid JSON format.")
                    continue

            category = file_categories[filename]
            faulty_srs = srs_data.copy()

            print(f"\n🔍 Processing: {filename} (Category: {category})")

            # Collect all existing requirement IDs in this file
            existing_requirements = set()
            for subsec in srs_data.get("[SEC] 2. Requirements", {}).values():
                if isinstance(subsec, dict):
                    existing_requirements.update(subsec.keys())

            # Iterate over requirements
            for sec, content in srs_data.get("[SEC] 2. Requirements", {}).items():
                if isinstance(content, dict):
                    for req_id, req_text in content.items():
                        if isinstance(req_text, str):
                            matches = re.findall(TRACEABILITY_PATTERN, req_text)

                            if matches:
                                original_links = matches[0].split(", ")
                                modified_links = original_links.copy()
                                removed_links = []
                                swapped_links = []

                                # Fault Injection - Remove links
                                if category in ["Mixed", "Missing"]:
                                    num_to_remove = max(1, int(0.3 * len(modified_links)))
                                    removed_links = random.sample(modified_links, min(num_to_remove, len(modified_links)))
                                    modified_links = [link for link in modified_links if link not in removed_links]

                                # Fault Injection - Swap links
                                if category in ["Mixed", "Swapped"]:
                                    num_to_swap = max(1, int(0.3 * len(modified_links)))
                                    for i, link in enumerate(random.sample(modified_links, min(num_to_swap, len(modified_links)))):
                                        prefix = "-".join(link.split("-")[:-1])

                                        if prefix in SWAP_CANDIDATES:
                                            new_trace_link = link.replace(prefix, SWAP_CANDIDATES[prefix])

                                            # Prevent self-linking
                                            if new_trace_link == req_id:
                                                print(f" Swap failed! {req_id} cannot link to itself. Retrying...")
                                                continue

                                            # Inject missing dependencies randomly
                                            if random.random() < 0.5:
                                                fake_trace_id = f"{prefix}-{random.randint(10, 99)}"
                                                if fake_trace_id not in existing_requirements:
                                                    print(f" Injecting missing traceability link: {fake_trace_id} in {req_id}")
                                                    new_trace_link = fake_trace_id

                                            modified_links[i] = new_trace_link
                                            swapped_links.append(modified_links[i])

                                # Clean up text after removing all traceability links
                                if not modified_links:
                                    faulty_srs["[SEC] 2. Requirements"][sec][req_id] = re.sub(r"\s*\(\s*\)", "", req_text).strip()
                                else:
                                    faulty_srs["[SEC] 2. Requirements"][sec][req_id] = req_text.replace(
                                        f"({', '.join(original_links)})",
                                        f"({', '.join(modified_links)})"
                                    )

                                # Log modifications
                                modifications_log.append([
                                    filename, req_id, ", ".join(original_links),
                                    ", ".join(removed_links), ", ".join(swapped_links)
                                ])

            # Save the faulty version
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(faulty_srs, f, indent=2, ensure_ascii=False)

    # Save modification log
    df_log = pd.DataFrame(modifications_log, columns=["Filename", "Requirement", "Original Links", "Removed Links", "Swapped Links"])
    df_log.to_excel(report_file, index=False)

# Function to Process All Folders
def process_all_traceability_folders():
    subfolders = ["Infer", "Train", "Val"]

    for subfolder in subfolders:
        input_folder = os.path.join(BASE_FOLDER, subfolder)
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, subfolder)
        report_file = os.path.join(OUTPUT_BASE_FOLDER, f"{subfolder}_traceability_log.xlsx")

        if os.path.exists(input_folder):
            print(f"\n Processing folder: {subfolder} ...")
            process_traceability_issues(input_folder, output_folder, report_file)

# Run the Program
process_all_traceability_folders()

print("\n Traceability Issue Processing Complete!")