import json
import os
import logging
import argparse

# Configure standard commercial logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_total_object_count(data):
    count = 0
    if isinstance(data, list):
        count += len(data)
        for item in data:
            count += get_total_object_count(item)
    elif isinstance(data, dict):
        count += len(data)
        for value in data.values():
            count += get_total_object_count(value)
    return count

def check_json_count(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: '{file_path}'")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Analyzing file: {os.path.basename(file_path)}")
        
        if isinstance(data, list):
            logger.info("Data type: JSON Array (List)")
            logger.info(f"Top-level object count: {len(data)}")
        elif isinstance(data, dict):
            logger.info("Data type: JSON Object (Dictionary)")
            logger.info(f"Top-level key count: {len(data.keys())}")
        else:
            logger.info(f"Data type: {type(data).__name__}")
            logger.warning("Cannot calculate top-level count (root element is not a List or Dictionary)")
            
        if isinstance(data, (list, dict)):
            logger.info(f"Total object count (including nested): {get_total_object_count(data)}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file '{file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing '{file_path}': {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the total object count in a JSON file.")
    parser.add_argument("file_path", help="Path to the JSON file to be analyzed")
    
    args = parser.parse_args()
    check_json_count(args.file_path)