import os
import json
import time
import uuid
from google import genai
from google.cloud import storage
from dotenv import load_dotenv

# Constants
LOCAL_INPUT_FILE = "batch_input.jsonl"
LOCAL_OUTPUT_DIR = "./output_data"
BATCH_OUTPUT_PREFIX = f"gemini_batch_output_{uuid.uuid4().hex[:8]}"

SYSTEM_PROMPT = """
You are an expert data analysis system. Your task is to analyze the provided text chunk and return a structured JSON object.
The JSON output must contain:
- "ai_summary": An ultra-concise 5-10 word summary of the content.
- "content_category": The category of the text (e.g., 'technical_doc', 'creative_writing', 'business_report').
- "style_tags": A list of 3-5 relevant topic or style keywords.
"""

TEST_DATA = [
    "The Gemini 1.5 Pro model is a powerful, multimodal large language model developed by Google AI. It excels at long-context understanding.",
    "The sun dipped below the horizon, painting the sky in hues of orange and purple. A gentle breeze rustled the leaves.",
    "Q3 sales increased by 15% year-over-year, driven by strong performance in the APAC region. Key initiatives for Q4 include market expansion."
]

def create_batch_input_file(file_path: str):
    print(f"Creating local batch input file: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for text in TEST_DATA:
            # Format specifically for Vertex AI Batch Prediction
            request_obj = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": text}]}],
                    "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                    "generationConfig": {"responseMimeType": "application/json"}
                }
            }
            f.write(json.dumps(request_obj) + "\n")

def gcs_upload(bucket_name: str, source_file: str, dest_blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(source_file)
    gcs_uri = f"gs://{bucket_name}/{dest_blob_name}"
    print(f"Uploaded input file to GCS: {gcs_uri}")
    return gcs_uri

def gcs_download_and_cleanup(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    print("\nDownloading results from GCS...")
    downloaded_files = []
    for blob in blobs:
        if blob.name.endswith("/"): continue
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        print(f" - Downloaded: {local_path}")
        downloaded_files.append(local_path)
        
    print("Cleaning up GCS output files...")
    for blob in blobs:
        blob.delete()
    print("GCS output cleanup completed.")
    
    return downloaded_files

def calculate_tokens(output_files: list):
    total_input = 0
    total_output = 0
    
    for file_path in output_files:
        if not file_path.endswith(".jsonl"): continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "response" in data and "usageMetadata" in data["response"]:
                        usage = data["response"]["usageMetadata"]
                        total_input += usage.get("promptTokenCount", 0)
                        total_output += usage.get("candidatesTokenCount", 0)
                except Exception as e:
                    continue
                    
    print("\n--- Token Usage Summary ---")
    print(f"Total Input Tokens:  {total_input}")
    print(f"Total Output Tokens: {total_output}")
    print(f"Overall Total:       {total_input + total_output}")
    print("---------------------------\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dotenv_path = os.path.join(project_root, 'mock.env')
    load_dotenv(dotenv_path)

    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash-002")

    if not all([PROJECT_ID, BUCKET_NAME]):
        print("Error: Missing GOOGLE_CLOUD_PROJECT or GCS_BUCKET_NAME in environment.")
        return

    print(f"Initializing Vertex AI GenAI Client (Project: {PROJECT_ID}, Location: {LOCATION})...")
    # Declaring vertexai=True routes the SDK automatically to the Vertex AI enterprise endpoint
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    input_blob_name = f"batch_input_{uuid.uuid4().hex[:8]}.jsonl"
    gcs_dest_prefix = f"gs://{BUCKET_NAME}/{BATCH_OUTPUT_PREFIX}/"

    try:
        # 1. Prepare and upload input
        create_batch_input_file(LOCAL_INPUT_FILE)
        gcs_input_uri = gcs_upload(BUCKET_NAME, LOCAL_INPUT_FILE, input_blob_name)
        
        # 2. Create Batch Job
        print(f"\nSubmitting Batch Job for model: {MODEL_NAME}...")
        job = client.batches.create(model=MODEL_NAME, src=gcs_input_uri, config={"dest": gcs_dest_prefix})
        print(f"Batch Job created successfully. Job ID: {job.name}")
        
        # 3. Monitor Status
        print("\nMonitoring job status (Updates every 30 seconds). This may take 10-30 minutes...")
        while True:
            job = client.batches.get(name=job.name)
            state_str = str(job.state).upper()
            print(f"[{time.strftime('%X')}] Current State: {state_str}")
            
            if any(end_state in state_str for end_state in ["SUCCEEDED", "FAILED", "CANCELLED"]):
                if "PARTIALLY" not in state_str and "SUCCEEDED" not in state_str:
                    print(f"Job ended with non-success state: {state_str}")
                break
            time.sleep(30)
            
        # 4. Download output and clean up GCS
        output_files = gcs_download_and_cleanup(BUCKET_NAME, BATCH_OUTPUT_PREFIX, LOCAL_OUTPUT_DIR)
        
        # 5. Calculate total input/output tokens
        calculate_tokens(output_files)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Ensure uploaded local and cloud input files are deleted
        print("Cleaning up GCS input file...")
        try:
            storage.Client().bucket(BUCKET_NAME).blob(input_blob_name).delete()
        except Exception:
            pass
        if os.path.exists(LOCAL_INPUT_FILE):
            os.remove(LOCAL_INPUT_FILE)
        print("Execution finished.")

if __name__ == "__main__":
    main()