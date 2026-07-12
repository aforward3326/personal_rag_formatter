# RAG and Anonymization Pipeline

## 1. Introduction

This project provides a comprehensive Python-based solution for advanced document processing and analysis. It integrates a Retrieval-Augmented Generation (RAG) pipeline for intelligent information retrieval, robust data anonymization capabilities to ensure privacy, and a CI/CD workflow for streamlined development and deployment.

The primary functionalities are divided into two main components:
1.  **RAG Ingestion Pipeline:** Processes source documents, enriches them using Large Language Models (LLMs), and ingests them into a vector database for efficient retrieval.
2.  **Anonymization Pipeline:** Scans and redacts Personally Identifiable Information (PII) from structured JSON files.

## 2. Core Features

*   **Retrieval-Augmented Generation (RAG):** Leverages state-of-the-art Large Language Models (LLMs), including OpenAI's GPT series and Google's Gemini, in conjunction with vector databases such as ChromaDB and pgvector. This architecture enables the system to deliver contextually relevant and accurate responses from a large knowledge base.
*   **Data Anonymization:** Incorporates the Presidio framework to automatically detect and redact Personally Identifiable Information (PII) from text, ensuring compliance with privacy standards.
*   **Multi-Format Document Processing:** The RAG pipeline offers robust support for a variety of document formats, including PDF, DOCX, PPTX, and plain text, enabling versatile data ingestion.
*   **Batch and Real-time Processing:** The RAG pipeline supports both real-time ingestion for smaller workloads and a batch processing mode for handling large volumes of data efficiently.
*   **CI/CD Integration:** Features a pre-configured `Jenkinsfile` to facilitate continuous integration and deployment, automating the testing and release lifecycle.
*   **Modular Architecture:** The codebase is organized into distinct, high-cohesion modules for key functionalities, including anonymization, the RAG pipeline, CI/CD processes, and text formatting utilities.

## 3. Project Structure

The repository is structured as follows:

```
.
├── src
│   ├── anonymize         # Logic for PII anonymization
│   ├── cicd_pipeline     # Scripts for the CI/CD pipeline
│   ├── rag_pipeline      # Core implementation of the RAG pipeline
│   └── text_formatting   # Text formatting and normalization utilities
├── data                  # Data sources for the RAG pipeline
├── test                  # Automated tests and test scripts
├── .gitignore            # Specifies files and directories to be ignored by version control
├── requirements.txt      # A list of Python package dependencies
└── pipeline.env.example  # An example environment configuration file
```

## 4. Installation and Configuration

### 4.1. Prerequisites

*   Python 3.11 or higher
*   `pip` and `venv`

### 4.2. Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file by copying the provided example. Populate this file with the required API keys and other service configurations.
    ```bash
    cp pipeline.env.example .env
    ```

## 5. Usage

This project contains two main executable pipelines.

### 5.1. RAG Ingestion Pipeline

This pipeline processes documents from the `data/` directory and ingests them into the configured vector database. It can be run in either **Standard (real-time)** or **Batch API** mode, controlled by the `USE_BATCH_API` variable in the configuration.

**To execute the pipeline, run the `main.py` module within the `rag_pipeline` package:**

```bash
python -m src.rag_pipeline.main [ARGUMENTS]
```

**Execution Modes & Arguments:**

*   **Standard Mode (`USE_BATCH_API=False`):**
    *   `--mode ingest`: (Default) Processes and ingests files in a single, real-time workflow.
    ```bash
    python -m src.rag_pipeline.main --mode ingest
    ```

*   **Batch API Mode (`USE_BATCH_API=True`):**
    This mode breaks the process into multiple steps.
    1.  `--mode submit`: Prepares and submits document chunks to the batch processing API.
        ```bash
        python -m src.rag_pipeline.main --mode submit
        ```
    2.  `--mode import_results`: Retrieves the processed results from a completed batch job.
        ```bash
        # The job name is automatically read from a local job file
        python -m src.rag_pipeline.main --mode import_results
        # Or specify the job name manually
        python -m src.rag_pipeline.main --mode import_results --job-name "your_batch_job_name"
        ```
    3.  `--mode ingest_db`: Ingests the cleaned, processed files into the vector database.
        ```bash
        python -m src.rag_pipeline.main --mode ingest_db
        ```
    4.  `--mode recover`: A utility to recover and ingest results from orphaned batch jobs.
        ```bash
        python -m src.rag_pipeline.main --mode recover
        ```

### 5.2. Anonymization Pipeline

This pipeline scans for `.json` files in the configured input directory, anonymizes their content, and saves the masked files to an output directory.

**To execute the pipeline, run the `main.py` module within the `anonymize` package:**

```bash
python -m src.anonymize.main
```
The script will automatically detect files and, if a previous run was interrupted, will prompt to resume from where it left off.

## 6. CI/CD Pipeline

This project is configured for automation using Jenkins. The `Jenkinsfile` orchestrates the CI/CD pipeline, which is composed of the following stages:

*   **Checkout:** Clones the source code from the version control repository.
*   **Install Dependencies:** Installs all required Python packages as specified in `requirements.txt`.
*   **Test:** Executes the automated test suite to validate code integrity.
*   **Deploy:** Deploys the application to a specified target environment (if applicable).

## 7. License

This project is licensed under the **GNU General Public License v3.0**. For complete details, please refer to the [LICENSE](LICENSE) file.
