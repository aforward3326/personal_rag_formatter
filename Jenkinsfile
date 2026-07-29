pipeline {
    agent any

    parameters {
        booleanParam(name: 'USE_BATCH_API', defaultValue: false, description: 'Check this to use the batch processing API.')
        booleanParam(name: 'RESUME_MODE', defaultValue: false, description: 'Check this to load a checkpoint and resume execution from the last failed step.')
        string(name: 'RESUME_BATCH_ID', defaultValue: '', description: 'Enter a Batch ID (e.g., cicd_xxx) to skip LLM batch submission and directly fetch and process the results.')
    }

    environment {
        // SSH connection info
        REMOTE_HOST = '192.168.x.x'
        REMOTE_USER = 'admin'
        LOCAL_PC_CRED_ID = 'CRED_ID'
        NAS_GIT_CRED_ID  = 'CRED_ID'

        // Project paths
        RAG_PROJECT_DIR = 'your-path'
        LOCAL_REPO_PATH = 'your-path'
        BASE_WORKSPACE_DIR = '/tmp/rag_workspace'
        LOG_DIR = '/tmp/rag_workspace/logs'

        // Core project settings
        PROJECT_NAME = 'your-project'
        PROGRAM_TYPE = 'python'
        GIT_URL      = 'your-git'
        BRANCH       = 'main'
        LOG_LEVEL = 'INFO'

        // --- Dual-track LLM Provider Settings ---
        ROUTING_STRATEGY = "auto" // auto, always_standard, always_thinking

        // --- Standard Track Settings ---
        STANDARD_AI_PROVIDER = "lm_studio"
        STANDARD_MODEL_NAME = "local-model/gemma-2b-it-q8_0.gguf"
        STANDARD_BASE_URL = "http://192.168.x.x:1234/v1"
        // STANDARD_API_KEY is set from credentials

        // --- Thinking Track Settings ---
        THINKING_AI_PROVIDER = "vertex"
        THINKING_MODEL_NAME = "gemini-1.5-pro"
        // THINKING_API_KEY is set from credentials
        THINKING_BASE_URL = "" // Not needed for Vertex

        // --- Embedding Settings ---
        EMBEDDING_PROVIDER = 'lm_studio'
        EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5-GGUF'
        EMBEDDING_DIM = 768
        EMBEDDING_BASE_URL = 'http://192.168.x.x:1234/v1'
        // EMBEDDING_API_KEY is set from credentials

        // Batch Processing & Cloud Settings
        GCS_BUCKET_NAME = 'your_process'
        GOOGLE_CLOUD_LOCATION = 'us-central1'

        // Database settings
        DB_HOST      = '192.168.x.x'
        DB_PORT      = '5432'
        DB_USER      = 'postgres'
        DB_NAME      = 'db'
        DB_TABLE_NAME   = 'table'
    }

    stages {
        stage('Deploy & Execute RAG Pipeline') {
            steps {
                sshagent(credentials: [LOCAL_PC_CRED_ID, NAS_GIT_CRED_ID]) {
                    script {
                        withCredentials([
                            string(credentialsId: 'your-db-password-id', variable: 'DB_PASSWORD'),
                            string(credentialsId: 'your-standard-api-key-id', variable: 'STANDARD_API_KEY'), // e.g., LM Studio key
                            string(credentialsId: 'your-thinking-api-key-id', variable: 'THINKING_API_KEY'), // e.g., Gemini/Vertex key
                            string(credentialsId: 'your-embedding-api-key-id', variable: 'EMBEDDING_API_KEY') // e.g., LM Studio key
                        ]) {

                            def resumeFlag = params.RESUME_MODE ? '--resume' : ''
                            def resumeBatchFlag = params.RESUME_BATCH_ID ? "--resume-batch-id ${params.RESUME_BATCH_ID}" : ''
                            def useBatchApi = params.USE_BATCH_API ? 'true' : 'false'

                            def remoteCmd = """
                                # 1. Update code repositories
                                cd ${LOCAL_REPO_PATH} && git fetch origin && git checkout ${BRANCH} && git pull origin ${BRANCH}
                                cd ${RAG_PROJECT_DIR}

                                # 2. Export environment variables
                                export PROJECT_NAME='${PROJECT_NAME}'
                                export PROGRAM_TYPE='${PROGRAM_TYPE}'
                                export GIT_URL='${GIT_URL}'
                                export BRANCH='${BRANCH}'
                                export LOCAL_REPO_PATH='${LOCAL_REPO_PATH}'
                                export DB_TABLE_NAME='${DB_TABLE_NAME}'
                                export DB_HOST='${DB_HOST}'
                                export DB_PORT='${DB_PORT}'
                                export DB_USER='${DB_USER}'
                                export DB_PASSWORD='\${DB_PASSWORD}'
                                export DB_NAME='${DB_NAME}'
                                export BASE_WORKSPACE_DIR='${BASE_WORKSPACE_DIR}'
                                export LOG_LEVEL='${LOG_LEVEL}'
                                export LOG_DIR='${LOG_DIR}'

                                # Export Batch and Cloud Settings
                                export USE_BATCH_API='${useBatchApi}'
                                export GOOGLE_CLOUD_PROJECT='${PROJECT_NAME}'
                                export GCS_BUCKET_NAME='${GCS_BUCKET_NAME}'
                                export GOOGLE_CLOUD_LOCATION='${GOOGLE_CLOUD_LOCATION}'
                                export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/gcp-service-account-key.json'

                                # Export LLM Provider Settings
                                export ROUTING_STRATEGY='${ROUTING_STRATEGY}'

                                export STANDARD_AI_PROVIDER='${STANDARD_AI_PROVIDER}'
                                export STANDARD_MODEL_NAME='${STANDARD_MODEL_NAME}'
                                export STANDARD_API_KEY='\${STANDARD_API_KEY}'
                                export STANDARD_BASE_URL='${STANDARD_BASE_URL}'

                                export THINKING_AI_PROVIDER='${THINKING_AI_PROVIDER}'
                                export THINKING_MODEL_NAME='${THINKING_MODEL_NAME}'
                                export THINKING_API_KEY='\${THINKING_API_KEY}'
                                export THINKING_BASE_URL='${THINKING_BASE_URL}'

                                # Export Embedding Settings
                                export EMBEDDING_PROVIDER='${EMBEDDING_PROVIDER}'
                                export EMBEDDING_MODEL_NAME='${EMBEDDING_MODEL_NAME}'
                                export EMBEDDING_DIM='${EMBEDDING_DIM}'
                                export EMBEDDING_API_KEY='\${EMBEDDING_API_KEY}'
                                export EMBEDDING_BASE_URL='${EMBEDDING_BASE_URL}'

                                # 3. Execute the RAG Pipeline
                                source venv/bin/activate
                                cd src
                                python3 -m cicd_pipeline.main --pipeline code_rag ${resumeFlag} ${resumeBatchFlag}
                            """

                            echo "Connecting to remote host to execute Git Pull and RAG Pipeline..."
                            sh "ssh -A -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} \"${remoteCmd}\""
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'CI/CD pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline execution failed. Check logs for details.'
        }
    }
}