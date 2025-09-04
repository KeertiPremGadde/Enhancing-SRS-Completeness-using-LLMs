"""
This code removes special characters from input json. Set your folder path appropriately in main function.
"""

import json
import os
import re
import sys
import logging

# Set up logging
logging.basicConfig(filename='preprocess.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def clean_raw_backslashes(content):
    """
    Clean raw backslashes while preserving valid escape sequences.
    """
    # First escape all backslashes
    content = content.replace('\\', '\\\\')
    
    # Then fix common escape sequences
    escape_sequences = ['n', 't', 'r', '"', '/', 'b', 'f']
    for seq in escape_sequences:
        content = content.replace(f'\\\\{seq}', f'\\{seq}')
    
    return content

def escape_string(match):
    """
    Enhanced string escaping function that handles all special cases.
    """
    s = match.group(1)
    
    # Handle existing escape sequences first
    s = s.replace('\\', '\\\\')  # Escape backslashes
    s = s.replace('"', '\\"')    # Escape double quotes
    s = s.replace("'", "\\'")    # Escape single quotes
    
    # Only replace unescaped newlines/whitespace
    if '\\n' not in s:
        s = s.replace('\n', '\\n')
    if '\\r' not in s:
        s = s.replace('\r', '\\r')
    if '\\t' not in s:
        s = s.replace('\t', '\\t')
    
    return f'"{s}"'

def preprocess_json_content(content):
    """
    Enhanced preprocessing function combining both approaches.
    """
    try:
        # First try to parse as-is
        json.loads(content)
        logging.info("Content already valid JSON, no preprocessing needed")
        return content
    except json.JSONDecodeError:
        logging.info("Content requires preprocessing")
        
        # Remove BOM if present
        content = content.strip("\ufeff")
        
        # Handle unclosed parentheses
        content = re.sub(r'\(([^\)]*$)', r'(\1)', content)
        
        # Replace whitespace within parentheses
        content = re.sub(r'\(\s+', '(', content)
        content = re.sub(r'\s+\)', ')', content)
        
        # Clean backslashes and preserve valid escape sequences
        content = clean_raw_backslashes(content)
        
        # Use more robust pattern for string matching
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        content = re.sub(pattern, escape_string, content)
        
        # Handle any remaining invalid escape characters
        content = re.sub(r'\\([^"\/bfnrtu\\])', r'\\\1', content)
        
        # Validate the processed content
        try:
            json.loads(content)
            logging.info("Successfully preprocessed and validated JSON content")
            return content
        except json.JSONDecodeError as e:
            error_msg = f"Failed to validate preprocessed JSON: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

def read_and_preprocess_json(file_path):
    """
    Read and preprocess JSON file with enhanced error handling.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"Successfully read file: {file_path}")
        
        preprocessed_content = preprocess_json_content(content)
        result = json.loads(preprocessed_content)
        logging.info(f"Successfully preprocessed file: {file_path}")
        return result
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {str(e)}")
    return None

def write_json_file(file_path, data):
    """
    Write processed JSON to file with error handling.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote output to {file_path}")
        print(f"Successfully wrote output to {file_path}")
    except Exception as e:
        error_msg = f"Error writing JSON to {file_path}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)

if __name__ == "__main__":
    # Set your path appropriately based on your input raw data directory
    BASE_DIR = r"D:\Datasets"
    INPUT_DIR = os.path.join(BASE_DIR, "Pure_79", "Raw")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Pure_79", "With_special_tokens")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of JSON files in input directory
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {INPUT_DIR}")
        sys.exit(1)

    # Print list of files and ask user to choose
    print("Available JSON files:")
    for i, file in enumerate(json_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("Enter the number of the file you want to preprocess (0 to exit): "))
            if choice == 0:
                print("Exiting program.")
                sys.exit(0)
            if 1 <= choice <= len(json_files):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    filename = json_files[choice - 1]
    input_file = os.path.join(INPUT_DIR, filename)
    output_file = os.path.join(OUTPUT_DIR, filename)

    print(f"Preprocessing {filename}...")
    result = read_and_preprocess_json(input_file)
    
    if result:
        write_json_file(output_file, result)
    else:
        print(f"Failed to preprocess {filename}")

    print("Preprocessing complete.")