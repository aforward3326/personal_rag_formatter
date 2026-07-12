import os
import re
from typing import List, Dict, Any
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from anonymize.factory import AnonymizerFactory

class CodeChunkerAndSanitizer:
    """Splits code into chunks and redacts sensitive information."""
    SUPPORTED_EXTS = [
        '.py', '.ts', '.tsx', '.js', '.jsx', '.java', '.dart',
        '.json', '.yaml', '.yml', '.xml', '.html', '.css',
        '.properties', '.gradle', '.sql', '.sh', '.md'
    ]
    EXT_TO_LANG = {
        '.py': Language.PYTHON, '.ts': Language.TS, '.tsx': Language.TS,
        '.js': Language.JS, '.jsx': Language.JS, '.java': Language.JAVA,
        '.html': Language.HTML, '.md': Language.MARKDOWN,
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            # Import the centralized anonymization processor (including Regex, Identity, Sensitive Words, Presidio strategies)
            self.anonymizer = AnonymizerFactory.create_processor()
            self.logger.info("Successfully loaded centralized AnonymizationProcessor.")
        except Exception as e:
            self.logger.warning(f"Failed to load centralized AnonymizerFactory: {e}. Falling back to basic regex.")
            self.anonymizer = None
            
        # Retain the code-specific API Key / Secret anonymization regex
        self.secret_regex = re.compile(
            r'(?i)(api[_\-]?key|secret|token|password)[\s=:]+[\'"][a-zA-Z0-9_\-\.]{12,}[\'"]'
        )

    def sanitize(self, text: str) -> str:
        """Redacts secrets and uses centralized anonymization for sensitive info."""
        # 1. Code-specific secret filtering
        text = self.secret_regex.sub(r'\1 = "[REDACTED_SECRET]"', text)
        
        # 2. Unified anonymization filtering (including identity, sensitive words, general regex like IP/Email/Phone, and Presidio NLP)
        if self.anonymizer:
            # Wrap as a dict and pass to process_document to trigger recursive checking and strategy application
            text = self.anonymizer.process_document({"content": text}).get("content", text)
        else:
            # If centralized component loading fails, keep basic IP anonymization as a fallback
            text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[REDACTED_IP]', text)
            
        return text

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads, sanitizes, and chunks a single file."""
        _, ext = os.path.splitext(file_path)
        if ext not in self.SUPPORTED_EXTS:
            return []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_text = f.read()
        except Exception as e:
            self.logger.warning(f"Could not read file {file_path}: {e}")
            return []

        sanitized_text = self.sanitize(original_text)
        lang = self.EXT_TO_LANG.get(ext)
        
        splitter = (
            RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=800, chunk_overlap=50)
            if lang
            else RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        )

        chunks = splitter.split_text(sanitized_text)
        results = []
        for chunk_text in chunks:
            start_idx = sanitized_text.find(chunk_text)
            start_line = sanitized_text.count('\n', 0, start_idx) + 1 if start_idx != -1 else 1
            end_line = start_line + chunk_text.count('\n')
            
            results.append({
                "content": chunk_text,
                "start_line": start_line,
                "end_line": end_line,
                "language": ext.strip('.')
            })
        return results
