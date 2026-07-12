pipeline {
    agent any

    parameters {
        booleanParam(name: 'RESUME_MODE', defaultValue: false, description: '勾選此項以載入 Checkpoint，從上次失敗的 Step 繼續執行')
        string(name: 'RESUME_BATCH_ID', defaultValue: '', description: '輸入 Batch ID (如 cicd_xxx) 以跳過 LLM 批次提交，直接抓取並處理結果')
    }

    environment {
        // SSH connection info for the remote host
        REMOTE_HOST = '192.168.x.x'
        REMOTE_USER = 'admin'
        LOCAL_PC_CRED_ID = 'CRED_ID'
        NAS_GIT_CRED_ID  = 'CRED_ID'

        // Absolute path to the RAG project on the remote host (directory containing Python code)
        RAG_PROJECT_DIR = 'your-path'
        LOCAL_REPO_PATH = 'your-path'

        // Required environment variables for the RAG program
        LLM_PROVIDER = 'gemini_vertex'
        USE_BATCH_API = true
        GCS_BUCKET_NAME = 'your_process'
        PROJECT_NAME = 'your-project'
        PROGRAM_TYPE = 'python'
        GIT_URL      = 'your-git'
        BRANCH       = 'main'
        BATCH_MODEL_NAME = 'gemini-2.5-flash'
        EMBEDDING_PROVIDER = 'lm_studio'
        EMBEDDING_MODEL_NAME = 'text-embedding-nomic-embed-code'

        // Database connection settings
        DB_HOST      = '192.168.x.x'
        DB_PORT      = '5432'
        DB_USER      = 'postgres'
        DB_NAME      = 'db'
        DB_TABLE_NAME   = 'table'

        // Local LLM service settings
        LM_STUDIO_BASE_URL = 'http://localhost:1234/v1'
    }

    stages {
        stage('Deploy & Execute on Local Computer') {
            steps {
                sshagent(credentials: [LOCAL_PC_CRED_ID, NAS_GIT_CRED_ID]) {
                    script {
                        // Bind credentials as environment variables for this block
                        withCredentials([
                            string(credentialsId: 'your-id', variable: 'DB_PASSWORD'),
                            string(credentialsId: 'gemini-api-key-id', variable: 'GEMINI_API_KEY'),
                            string(credentialsId: 'lm_studio_key', variable: 'LM_STUDIO_API_KEY')
                        ]) {

                            def resumeFlag = params.RESUME_MODE ? '--resume' : ''
                            def resumeBatchFlag = params.RESUME_BATCH_ID ? "--resume-batch-id ${params.RESUME_BATCH_ID}" : ''

                            def remoteCmd = """
                                # 1. Enter the target code repository and update via Git
                                cd ${LOCAL_REPO_PATH} && \\
                                git fetch origin && \\
                                git checkout ${BRANCH} && \\
                                git pull origin ${BRANCH} && \\

                                # 2. Switch to the RAG program directory to prepare for execution
                                cd ${RAG_PROJECT_DIR} && \\

                                # 3. Export all required environment variables for the RAG program (no spaces around equals, ends with && \\)
                                export PROJECT_NAME='${PROJECT_NAME}' && \\
                                export PROGRAM_TYPE='${PROGRAM_TYPE}' && \\
                                export GIT_URL='${GIT_URL}' && \\
                                export BRANCH='${BRANCH}' && \\
                                export LOCAL_REPO_PATH='${LOCAL_REPO_PATH}' && \\
                                export DB_TABLE_NAME='${DB_TABLE_NAME}' && \\
                                export DB_HOST='${DB_HOST}' && \\
                                export DB_PORT='${DB_PORT}' && \\
                                export DB_USER='${DB_USER}' && \\
                                export DB_PASSWORD='\${DB_PASSWORD}' && \\
                                export DB_NAME='${DB_NAME}' && \\
                                export LM_STUDIO_BASE_URL='${LM_STUDIO_BASE_URL}' && \\
                                export LLM_PROVIDER='${LLM_PROVIDER}' && \\
                                export GEMINI_API_KEY='\${GEMINI_API_KEY}' && \\
                                export LM_STUDIO_API_KEY='\${LM_STUDIO_API_KEY}' && \\
                                export USE_BATCH_API='true' && \\
                                export BATCH_MODEL_NAME='${BATCH_MODEL_NAME}' && \\
                                export GOOGLE_CLOUD_PROJECT='${PROJECT_NAME}' && \\
                                export GCS_BUCKET_NAME='${GCS_BUCKET_NAME}' && \\
                                export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/gcp-service-account-key.json' && \
                                export EMBEDDING_PROVIDER='${EMBEDDING_PROVIDER}' && \\
                                export EMBEDDING_MODEL_NAME='${EMBEDDING_MODEL_NAME}' && \\

                                # 4. Execute the RAG Pipeline
                                source venv/bin/activate && \\
                                cd src && \\
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
            echo 'CI/CD pipeline executed successfully. RAG data updated and imported into the database!'
        }
        failure {
            echo 'Pipeline execution failed. Please check the network connection to the remote host or the Python error logs.'
        }
    }
}