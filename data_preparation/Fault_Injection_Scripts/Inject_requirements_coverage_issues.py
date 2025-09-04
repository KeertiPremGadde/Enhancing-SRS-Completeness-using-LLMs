# Inject functional incompleteness into the input SRS functional requirements with To-be-defined wherever a target phrase is found.
# Set your path appropriately based on your input data directory.

import os
import json
import pandas as pd
import re

# Define Base Paths
BASE_FOLDER = r"D:\Datasets\Thesis_data"
OUTPUT_BASE_FOLDER = r"D:\Datasets\Thesis_data\requirement_coverage"

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# Define Target Phrases for Modification
TARGET_PHRASES = [
    "The system shall allow users to",
    "The system shall support",
    "The system shall provide",
    "The system shall implement",
    "The system shall enable",
    "The system shall process",
    "The system shall store",
    "The system shall retrieve"
]

# Function to Modify Functional Requirements
def modify_functional_requirements(requirements_dict):
    """
    Modifies only one functional requirement per file:
    1. If a target phrase exists, modify only the first occurrence.
    2. If no target phrase exists, modify the first functional requirement while keeping the next word for context.
    3. Always retain traceability links.
    4. Preserve JSON structure so that all requirements remain key-value pairs.
    """
    modified_reqs = requirements_dict.copy()  # Preserve the original structure
    phrase_found = False
    original_line = ""
    modified_line = ""

    # Iterate over functional requirements in order
    for req_id, req_text in modified_reqs.items():
        # Try to find a target phrase and replace only one occurrence
        for phrase in TARGET_PHRASES:
            if phrase in req_text:
                original_line = req_text
                # Preserve traceability links if present
                if "(" in req_text:
                    modified_reqs[req_id] = f"{phrase} TO-BE-DEFINED {req_text[req_text.index('('):]}"
                else:
                    modified_reqs[req_id] = f"{phrase} TO-BE-DEFINED."
                
                modified_line = modified_reqs[req_id]
                phrase_found = True
                break  # Stop modifying after first match

        if phrase_found:
            break  # Stop processing after modifying one requirement

    # If no phrase was found, apply the fallback rule (modify the first "The system shall..." requirement)
    if not phrase_found:
        for req_id, req_text in modified_reqs.items():
            match = re.match(r"(The system shall\s+\S+)", req_text)  # Keep immediate next word
            if match:
                original_line = req_text
                next_word = match.group(1)  # Get first keyword after 'shall'
                # Preserve traceability links if present
                if "(" in req_text:
                    modified_reqs[req_id] = f"{next_word} TO-BE-DEFINED {req_text[req_text.index('('):]}"
                else:
                    modified_reqs[req_id] = f"{next_word} TO-BE-DEFINED."
                
                modified_line = modified_reqs[req_id]
                break  # Apply only to the first functional requirement

    return modified_reqs, original_line, modified_line

# Function to Process All Folders (Infer, Train, Val)
def process_coverage_folders():
    subfolders = ["Infer", "Train", "Val"]

    for subfolder in subfolders:
        input_folder = os.path.join(BASE_FOLDER, subfolder)
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, subfolder)
        excel_log_file = os.path.join(OUTPUT_BASE_FOLDER, f"{subfolder}_coverage_log.xlsx")

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Load Files
        log_data = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".json") and filename.startswith("Complete"):
                file_path = os.path.join(input_folder, filename)

                try:
                    # Load JSON content
                    with open(file_path, "r", encoding="utf-8") as f:
                        srs_data = json.load(f)

                    # Extract functional requirements as a dictionary
                    functional_reqs = srs_data.get("[SEC] 2. Requirements", {}).get("[SUBSEC] 2.2 Functions", {})

                    if isinstance(functional_reqs, dict):  # Ensure it's a dictionary
                        # Apply modification
                        modified_reqs, original_text, modified_text = modify_functional_requirements(functional_reqs)

                        # Save modification if any change happened
                        if original_text and modified_text:
                            srs_data["[SEC] 2. Requirements"]["[SUBSEC] 2.2 Functions"] = modified_reqs

                            # Change filename convention
                            file_number = filename.split("_")[1].split(".")[0]  # Extract "00001"
                            output_filename = f"incomplete_srs_{file_number}_00002.json"
                            output_file_path = os.path.join(output_folder, output_filename)

                            # Save modified JSON file
                            with open(output_file_path, "w", encoding="utf-8") as f:
                                json.dump(srs_data, f, indent=2, ensure_ascii=False)

                            # Log changes
                            log_data.append({
                                "Filename": output_filename,
                                "Original Functional Requirement": original_text,
                                "Modified Functional Requirement": modified_text
                            })

                except Exception as e:
                    print(f" Error processing {filename}: {e}")

        # Convert log data to DataFrame & Save
        if log_data:
            log_df = pd.DataFrame(log_data, columns=["Filename", "Original Functional Requirement", "Modified Functional Requirement"])
            log_df.to_excel(excel_log_file, index=False, engine="openpyxl")

        print(f" {subfolder} processing complete! Modified files are saved in {output_folder}.")
        print(f" Log file saved at {excel_log_file}.")

# Run the Program
process_coverage_folders()
print("\nRequirement Coverage Processing Complete!")
