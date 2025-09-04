"""This code is used to call the OpenAI API to process the input SRS content to get a complete input as per the ISO/IEC/IEEE 29148:2018 Standard for Software Requirements Specification (SRS).
Set your path appropriately based on your input preprocessed data directory"""


import json
import openai
import os
import sys
import time
import random
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'srs_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Print API key status (first 5 chars only for security)
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    logging.info(f"API Key found (starts with: {api_key[:5]}...)")
else:
    logging.error("API Key not found in environment variables")
    sys.exit(1)

# Initialize the OpenAI client
try:
    client = openai.OpenAI(api_key=api_key)
    logging.info("OpenAI client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {str(e)}")
    sys.exit(1)

# Define directories
# Set your path appropriately based on your input preprocessed data directory
BASE_DIR = r"D:\Datasets"
INPUT_DIR = os.path.join(BASE_DIR, "Pure_79", "With_special_tokens")
OUTPUT_DIR = os.path.join(BASE_DIR, "Pure_79", "With_special_tokens")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompt for API
prompt = """
As a Requirements Engineer specializing in the ISO/IEC/IEEE 29148:2018 Standard for Software Requirements Specification (SRS) whose format is mentioned below, your task is to analyze the provided SRS content and enhance it to ensure completeness and adherence to the provided format. Where logical dependencies exist, introduce cross-references implicitly based on the functional relationships in the document. For example, any requirement that displays data or shares functionality with another should reference the relevant ID. Dependencies should be inferred where they logically follow, even if not explicitly present in the input.

### Step 1: Structural Analysis and Hierarchical Alignment
1. **Analyze and restructure** the input SRS content to **match the hierarchical structure** of the ISO/IEC/IEEE 29148:2018 Standard as shown below.
   - **Each section, subsection, and subsubsection should align exactly with the provided structure** (e.g., 1. Introduction, 1.1 Purpose, 1.2 Scope).
   - The **[SEC], [SUBSEC], and [SUBSUBSEC]** tokens are placeholders for this structure and should **not be included in the final output**.
2. **Consolidate content directly under main sections** (e.g., Functions, Usability Requirements, Software System Attributes) without additional subdivision.
   - For similar or minor requirements, **merge them into a single requirement** to reduce redundancy. For example:
     - "The system shall support user account creation via email."  
       "The system shall support user account creation via social media."
       - Should be **merged to**: “The system shall support user account creation via email and social media.”

### Step 2: Requirements Identification
1. For **Section 2 (Requirements)** only:
   - Assign **unique requirement identifiers** in the format **REQ-[Category]-[Sequential Number]** (e.g., REQ-FUNC-001).
   - **Categories should include**: INTF (External Interfaces), FUNC (Functional), PERF (Performance), USAB (Usability), DB (Database), DSGN (Design), COMP (Compliance), ATTR (System Attributes).
   - **Use these IDs consistently for all cross-references**; avoid using requirement IDs in sections like Introduction or Scope.

### Step 3:Token Management
1. **Ensure the entire generated output does not exceed 1024 tokens.**
2. **Limit any subsection or subblock within Section 2 (Requirements) to a maximum of 2-3 requirements. Combine related or minor requirements into concise, consolidated statements while ensuring traceability and logical cross-references are preserved.**

### Step 4: Modal Verb Consistency
1. **Use modal verbs accurately** for requirements clarity:
   - **Mandatory requirements**: "Shall" and "Shall not"
   - **Optional requirements**: "Should," "Should not," "May"
   - **Descriptions**: "Will" for future actions, "Can" for capability description.
   - **Avoid using "must" or "must not"**; instead, use "shall" or "shall not."

### Step 5: **Generate Cross-References and Ensure Traceability**
1. **Identify dependencies** between requirements and **generate cross-references based on logical relationships**:
   - **Analyze the requirements to detect dependencies**, even if they are not explicitly mentioned in the input. **Where one requirement is related to or impacted by another, include a cross-reference to that requirement ID**.
   - **Focus on the following types of relationships**:
      - **Data Flow Dependencies**: If one requirement's output is another’s input (e.g., “The system shall encrypt user data (REQ-FUNC-001) before storing it in the database (REQ-DB-002)”).
      - **Functional Dependencies**: Where one requirement must be met before another can execute (e.g., “The system shall authenticate users (REQ-FUNC-001) before granting access to resources”).
      - **Performance Dependencies**: Where performance requirements impact another requirement (e.g., “The system shall process data within the response time defined in REQ-PERF-001”).
   - **Provide examples directly within requirements where dependencies exist, and integrate cross-references seamlessly** into the requirement text.
     Example:REQ-FUNC-002: The system shall provide real-time data acquisition and processing through scalable microservices (REQ-PERF-003).
   - **Avoid standalone references with format “see REQ-FUNC-001.”** 
   - **Avoid mentioning the references with format “cross-reference REQ-FUNC-001.”**

### Step 6: Dependency Identification
1. **Encourage cross-referencing where dependencies exist**. If missing, add cross-references based on:
   - **Data Flow Relationships**: Where one requirement’s output is another’s input.
   - **Functional Dependencies**: Where sequential requirements exist.
   - **Performance Impacts**: Where a requirement affects system performance.
   - **Technical Dependencies**: Where a specific technology or standard impacts other requirements (e.g., encryption standards impacting data requirements).

### Step 7: Technology Updates
1. **Replace all outdated technologies, software, and standards** (pre-2015) with current equivalents, where applicable. Examples:
   - Replace outdated programming languages (e.g., **COBOL** with **Python** or **Java**).
   - Replace databases like **SQL Server 2008** with **SQL Server 2019** or **PostgreSQL 14**.
   - Update **outdated standards (e.g., FIPS PUB 127-2)** to their latest available versions or remove if irrelevant.
2. **Exclude implicit references, such as the ISO/IEC/IEEE 29148:2018 standard, ISO/IEC/IEEE 29148:2018 - Systems and Software Engineering.**

### Step 8: Definitions and Acronyms
1. **Definitions**: Use **section 1.8** to include detailed descriptions for frequently used terms within the given SRS content(e.g., “User: A person who interacts with the system to achieve a specific goal”). Avoid abbreviations in this section.
2. **Acronyms and Abbreviations**: Use **section 1.9** for abbreviations only, without detailed descriptions (e.g., "API: Application Programming Interface", "EIRENE": "European Integrated Railway Radio Enhanced Network").
3. **Choose the most relevant and frequently used definitions and limit "Definitions" to a maximum of 2-3 entries.**
4. **Choose the most relevant and frequently used acronyms and abbreviations and limit "Acronyms and Abbreviations" to a maximum of 2-3 entries.**

### Step 9: Completeness and Style Consistency
1. **Ensure no placeholders** like "To be defined" are present. Fill in missing sections using context from the SRS and industry-standard practices.
2. **Ensure **consistency in tone and style**. Maintain a professional, relevant tone without slang or informal terms.**
3. **Eliminate any placeholder or incorrect references (e.g., "as specified in sections 5-8") that do not correspond to actual sections in the document. Cross-check references and ensure they map to valid sections or subsections.
4. **If the input contains vague or incorrect placeholders, rephrase to provide clear and relevant information.**

### Output Formatting Rules
1. **Use JSON structure** as shown below:
   - Each section as a key-value pair.
   - Lists as arrays of strings, with each sentence starting on a new line.
   - Proper indentation for JSON hierarchy alignment.

**ISO/IEC/IEEE 29148:2018 Standard for Software Requirements Specification (SRS) Structure**:

```json
{
  "[SEC] 1. Introduction": {
    "[SUBSEC] 1.1 Purpose": "",
    "[SUBSEC] 1.2 Scope": "",
    "[SUBSEC] 1.3 Product Perspective": {
      "[SUBSUBSEC] 1.3.1 System Interfaces": "",
      "[SUBSUBSEC] 1.3.2 User Interfaces": "",
      "[SUBSUBSEC] 1.3.3 Hardware Interfaces": "",
      "[SUBSUBSEC] 1.3.4 Software Interfaces": "",
      "[SUBSUBSEC] 1.3.5 Communications Interfaces": "",
      "[SUBSUBSEC] 1.3.6 Memory Constraints": "",
      "[SUBSUBSEC] 1.3.7 Operations": "",
      "[SUBSUBSEC] 1.3.8 Site Adaptation Requirements": "",
      "[SUBSUBSEC] 1.3.9 Interfaces with Services": ""
    },
    "[SUBSEC] 1.4 Product Functions": "",
    "[SUBSEC] 1.5 User Characteristics": "",
    "[SUBSEC] 1.6 Limitations": "",
    "[SUBSEC] 1.7 Assumptions and Dependencies": "",
    "[SUBSEC] 1.8 Definitions": "",
    "[SUBSEC] 1.9 Acronyms and Abbreviations": ""
  },
  "[SEC] 2. Requirements": {
    "[SUBSEC] 2.1 External Interfaces": "",
    "[SUBSEC] 2.2 Functions": "",
    "[SUBSEC] 2.3 Usability Requirements": "",
    "[SUBSEC] 2.4 Performance Requirements": "",
    "[SUBSEC] 2.5 Logical Database Requirements": "",
    "[SUBSEC] 2.6 Design Constraints": "",
    "[SUBSEC] 2.7 Standards Compliance": "",
    "[SUBSEC] 2.8 Software System Attributes": ""
  },
  "[SEC] 3. Verification": "",
  "[SEC] 4. Supporting Information": "",
  "[SEC] 5. References": ""
}
```

*List each requirement on a new line, and include cross-references, relevant use cases, or workflows.*
{srs_content}
"""

def read_json_file(file_path):
    """
    Read and parse JSON file with enhanced error handling
    """
    logging.info(f"Attempting to read file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try to parse JSON to validate format
            parsed_content = json.loads(content)
            logging.info(f"Successfully read and parsed file: {file_path}")
            return parsed_content
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {file_path}: {str(e)}")
        print(f"Error: Invalid JSON in file {file_path}")
        print(f"JSON error details: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error reading file {file_path}: {str(e)}")
        print(f"Unexpected error reading file {file_path}: {str(e)}")
    return None

def write_json_file(file_path, data):
    """
    Write JSON to file with validation and error handling
    """
    logging.info(f"Attempting to write to file: {file_path}")
    try:
        # Validate JSON before writing
        json.dumps(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote output to {file_path}")
        print(f"Successfully wrote output to {file_path}")
        return True
    except TypeError as e:
        logging.error(f"Data serialization error: {str(e)}")
        print(f"Error: Invalid data structure for JSON serialization: {str(e)}")
    except Exception as e:
        logging.error(f"Error writing JSON to {file_path}: {str(e)}")
        print(f"Error writing JSON to {file_path}: {str(e)}")
    return False

def clean_json_output(json_output):
    """
    Clean and validate JSON output from API response
    """
    try:
        # Remove problematic characters
        json_output = json_output.replace('\r', '').replace('\x00', '')
        
        # Find the JSON part
        json_start = json_output.find('{')
        json_end = json_output.rfind('}') + 1
        
        if json_start == -1 or json_end == -1:
            logging.error("Could not find valid JSON markers in response")
            return None
            
        json_content = json_output[json_start:json_end]
        
        # Validate JSON
        parsed_json = json.loads(json_content)
        return parsed_json
    except Exception as e:
        logging.error(f"Error cleaning JSON output: {str(e)}")
        return None

def post_process_srs(content):
    """
    Post-process SRS content with improved error handling
    """
    logging.info("Starting post-processing of SRS content")
    
    def process_value(value):
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], list):
                value = value[0]
            return [item.strip() if isinstance(item, str) else process_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, str):
            return value.strip()
        return value

    def remove_empty_items(data):
        if isinstance(data, dict):
            return {k: remove_empty_items(v) for k, v in data.items() if v not in (None, "", [], {})}
        elif isinstance(data, list):
            return [remove_empty_items(item) for item in data if item not in (None, "", [], {})]
        return data

    try:
        processed_content = process_value(content)
        cleaned_content = remove_empty_items(processed_content)
        logging.info("Successfully completed post-processing")
        return cleaned_content
    except Exception as e:
        logging.error(f"Error in post-processing: {str(e)}")
        return None
    
def process_srs(input_file, max_retries=3, initial_delay=1):
    """
    Main SRS processing function with enhanced error handling and retry logic
    """
    logging.info(f"Starting processing of file: {input_file}")
    
    # Read input JSON
    srs_content = read_json_file(input_file)
    if not srs_content:
        logging.error(f"Failed to read input file: {input_file}")
        return None
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Prepare the prompt with the SRS content
            formatted_prompt = prompt.replace("{srs_content}", json.dumps(srs_content))
            
            # Make the API call
            logging.info("Making API call to OpenAI")
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=3000,
                temperature=0.1,
            )

            # Extract and process the content
            logging.info("Processing API response")
            json_output = response['choices'][0]['message']['content']
            
            # Handle different response types
            if isinstance(json_output, list):
                json_output = ''.join(str(block.text) for block in json_output if hasattr(block, 'text'))
            elif hasattr(json_output, 'text'):
                json_output = json_output.text
            else:
                json_output = str(json_output)

            # Clean and parse JSON
            parsed_json = clean_json_output(json_output)
            if not parsed_json:
                raise ValueError("Failed to extract valid JSON from response")

            # Post-process the content
            processed_json = post_process_srs(parsed_json)
            if not processed_json:
                raise ValueError("Failed to post-process JSON content")

            logging.info("Successfully processed SRS content")
            return processed_json

        except openai.error.OpenAIError as e:
            if e.status_code == 529:  # Overloaded error
                retry_count += 1
                if retry_count < max_retries:
                    delay = initial_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                    logging.warning(f"API overloaded. Retrying in {delay:.2f} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    print(f"API overloaded. Retrying in {delay:.2f} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached. API is still overloaded.")
                    print("Max retries reached. API is still overloaded.")
                    return None
            else:
                logging.error(f"API error: {str(e)}")
                print(f"API error: {str(e)}")
                return None
        except Exception as e:
            logging.error(f"Error processing {input_file}: {str(e)}")
            print(f"Error processing {input_file}: {str(e)}")
            return None
    
    logging.error("Failed to process file after multiple attempts")
    return None

def main():
    """
    Main execution function with improved error handling and user interaction
    """
    # Get list of JSON files
    try:
        json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    except Exception as e:
        logging.error(f"Error accessing input directory: {str(e)}")
        print(f"Error accessing input directory: {str(e)}")
        return

    if not json_files:
        logging.error(f"No JSON files found in {INPUT_DIR}")
        print(f"No JSON files found in {INPUT_DIR}")
        return

    # Display available files
    print("\nAvailable JSON files:")
    for i, file in enumerate(json_files, 1):
        print(f"{i}. {file}")

    # Get user selection
    while True:
        try:
            choice = input("\nEnter the number of the file you want to process (0 to exit): ")
            if not choice.strip():
                continue
                
            choice = int(choice)
            if choice == 0:
                print("Exiting program.")
                logging.info("Program terminated by user")
                return
                
            if 1 <= choice <= len(json_files):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            logging.error(f"Error in file selection: {str(e)}")
            print(f"Error in file selection: {str(e)}")
            return

    # Process selected file
    filename = json_files[choice - 1]
    input_file = os.path.join(INPUT_DIR, filename)
    output_file = os.path.join(OUTPUT_DIR, filename)

    print(f"\nProcessing {filename}...")
    logging.info(f"Starting processing of {filename}")

    result = process_srs(input_file)
    
    if result:
        if write_json_file(output_file, result):
            logging.info(f"Successfully processed and saved {filename}")
            print(f"\nSuccessfully processed {filename}")
        else:
            logging.error(f"Failed to save processed file {filename}")
            print(f"\nFailed to save processed file {filename}")
    else:
        logging.error(f"Failed to process {filename}")
        print(f"\nFailed to process {filename}")

    print("\nProcessing complete.")
    logging.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}")
        print(f"Critical error: {str(e)}")