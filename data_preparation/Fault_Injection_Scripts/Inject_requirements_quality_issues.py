# Inject quality attributes into the input SRS content with percentage-based, time-based, and number-based replacements in SRS requirements especially in performance requirements.
# Set your path appropriately based on your input data directory.

import os
import json
import re
import pandas as pd

# Extract standards/protocols (A-Z words followed by number/decimal)
STANDARD_PATTERN = re.compile(r"\b([A-Z]+[A-Z0-9]*) (\d+(\.\d+)?)\b")

# Extract complete percentage values before replacement (Handles both whole & decimal percentages)
PERCENTAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?%\b")

# Define percentage replacements
PERCENTAGE_DECIMAL_REPLACEMENTS = {
    r"99.99%": "almost always",
    r"99.9%": "nearly always",
    r"99.5%": "highly reliable",
    r"98%": "generally available",
    r"95%": "available most of the time",
    r"90-99%": "in most cases",
    r"80-89%": "in many scenarios",
    r"70-79%": "in several cases",
    r"50-69%": "sometimes",
    r"30-49%": "occasionally",
    r"10-29%": "rarely",
    r"0-9%": "almost never"
}

# Handle time replacements
TIME_REPLACEMENTS = {
    r"\bwithin 1 second\b": "instantly",
    r"\bwithin 2 seconds\b": "without significant delay",
    r"\bwithin 3 seconds\b": "with minimal delay",
    r"\bwithin 5 seconds\b": "promptly",
    r"\bwithin 10 seconds\b": "within a short time",
    r"\bwithin 15 seconds\b": "within a brief moment",
    r"\bwithin 30 seconds\b": "within a short delay",
    r"\bwithin 1 minute\b": "within a short duration",
    r"\bwithin 2 minutes\b": "within a few moments",
    r"\bwithin 5 minutes\b": "within a brief period",
    r"\bwithin 30 minutes\b": "within some time",
    r"\bwithin (\d+)-(\d+) seconds\b": "within a flexible time range"
}

# Handle number-based replacements
NUMBER_REPLACEMENTS = {
    r"\b10,000\b": "many thousands",
    r"\b5,000\b": "several thousand",
    r"\b2,000\b": "a few thousand",
    r"\b1,000\b": "roughly a thousand",
    r"\b500\b": "several hundred",
    r"\b100\b": "about a hundred",
    r"\b50\b": "a few dozen",
    r"\b10\b": "several",
    r"\b5\b": "a few",
    r"\b3\b": "a couple",
    r"\b1\b": "at least one"
}

# Function to extract and replace standards
def extract_standards(text):
    standards = {}
    def replacer(match):
        key = f"SAFE_{len(standards) + 1}"
        standards[key] = match.group(0)
        return key
    
    text = STANDARD_PATTERN.sub(replacer, text)
    return text, standards

def restore_standards(text, standards):
    for key, value in standards.items():
        text = text.replace(key, value)
    return text

# Function to extract, modify, and replace percentages
def extract_and_modify_percentages(text):
    percentages = {}

    def replacer(match):
        original_value = match.group(0)
        modified_value = PERCENTAGE_DECIMAL_REPLACEMENTS.get(original_value, original_value)  # Replace full match
        key = f"PERCENTAGE_{len(percentages) + 1}"
        percentages[key] = modified_value
        return key

    text = PERCENTAGE_PATTERN.sub(replacer, text)  # Replace percentage values
    return text, percentages

def restore_percentages(text, percentages):
    for key, value in percentages.items():
        text = text.replace(key, value)  # Restore all percentages as a unit
    return text

# Function to apply regex replacements
def replace_values(obj, replacements, issue_type, filename, logs):
    if isinstance(obj, dict):
        return {key: replace_values(value, replacements, issue_type, filename, logs) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_values(item, replacements, issue_type, filename, logs) for item in obj]
    elif isinstance(obj, str):
        original_text, standards = extract_standards(obj)
        original_text, percentages = extract_and_modify_percentages(original_text)
        modified_text = original_text

        # Handle exact time replacements
        for pattern, replacement in replacements.items():
            modified_text = re.sub(pattern, replacement, modified_text)

        # Handle time ranges (e.g., "5-30 seconds")
        modified_text = re.sub(r"\bwithin (\d+)-(\d+) seconds\b", "within a flexible time range", modified_text)

        modified_text = restore_percentages(modified_text, percentages)  # Ensure whole % replacement
        modified_text = restore_standards(modified_text, standards)

        if original_text != modified_text:
            logs.append([filename, issue_type, original_text.strip(), modified_text.strip()])
        
        return modified_text
    return obj

# Function to process JSON files
def process_json_files(input_folder, output_folder, report_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logs = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json") and filename.startswith("Complete"):
            input_path = os.path.join(input_folder, filename)
            
            # New Naming Convention
            file_number = filename.split("_")[1].split(".")[0]  # Extracts "00001"
            output_filename = f"incomplete_srs_{file_number}_00004.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(input_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            data = replace_values(data, PERCENTAGE_DECIMAL_REPLACEMENTS, "Percentage-Based", filename, logs)
            data = replace_values(data, TIME_REPLACEMENTS, "Time-Based", filename, logs)
            data = replace_values(data, NUMBER_REPLACEMENTS, "Number-Based", filename, logs)

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)

    df = pd.DataFrame(logs, columns=["Filename", "Issue Type", "Original Sentence", "Updated Sentence"])
    df.to_excel(report_file, index=False)

# Main function to process all folders automatically
def process_all_folders(base_folder, output_base_folder):
    subfolders = ["Infer", "Train", "Val"]

    for subfolder in subfolders:
        input_folder = os.path.join(base_folder, subfolder)
        output_folder = os.path.join(output_base_folder, subfolder)

        report_file = os.path.join(output_base_folder, f"{subfolder}_quality_attributes.xlsx")

        if os.path.exists(input_folder):
            print(f"Processing {subfolder}...")
            process_json_files(input_folder, output_folder, report_file)

# Define paths
BASE_FOLDER = r"D:\Datasets\Thesis_data"
OUTPUT_BASE_FOLDER = r"D:\Datasets\Thesis_data\requirement_quality_attributes"

# Run the program for all folders
process_all_folders(BASE_FOLDER, OUTPUT_BASE_FOLDER)

print("Processing complete! JSON files and reports generated.")