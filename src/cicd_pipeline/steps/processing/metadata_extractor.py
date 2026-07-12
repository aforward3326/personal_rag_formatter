import re
from typing import Dict, Any
import logging

class CodeMetadataExtractor:
    """Extracts basic code metadata using regular expressions."""
    PATTERNS = {
        'python': {
            'function': re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            'class': re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            'imports': re.compile(r'^(?:from\s+([^\s\.]+)|import\s+([^\s\.]+))', re.MULTILINE)
        },
        'javascript': {
            'function': re.compile(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            'class': re.compile(r'class\s+([A-Z][a-zA-Z0-9_]*)'),
            'imports': re.compile(r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE)
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, code_chunk: str, language: str) -> Dict[str, Any]:
        """Extracts node_type, node_name, and dependencies."""
        lang_patterns = self.PATTERNS.get(language)
        if not lang_patterns:
            return {"node_type": "unknown", "node_name": "unknown", "dependencies": []}

        class_match = lang_patterns.get('class', re.compile('(?!)')).search(code_chunk)
        if class_match:
            node_type, node_name = "class", class_match.group(1)
        else:
            func_match = lang_patterns.get('function', re.compile('(?!)')).search(code_chunk)
            node_type, node_name = ("function", func_match.group(1)) if func_match else ("unknown", "unknown")
        
        imports = lang_patterns.get('imports', re.compile('(?!)')).findall(code_chunk)
        dependencies = list(filter(None, {item for t in imports for item in t if item}))

        return {"node_type": node_type, "node_name": node_name, "dependencies": dependencies}
