from .base import AnonymizeStrategy
import re
from typing import List, Tuple

class RegexAnonymizer(AnonymizeStrategy):
    """
    Anonymizes text by applying a series of regular expression patterns
    to find and replace sensitive information like IPs, URLs, emails, etc.
    """
    
    # Pre-defined, compiled regex rules for efficiency
    DEFAULT_REGEX_RULES: List[Tuple[re.Pattern, str]] = [
        # IP Address
        (re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)'), '[REDACTED_IP]'),
        # URL
        (re.compile(r'https?://[^\s<>"\']+|(?:www\.)[^\s<>"\']+'), '[REDACTED_URL]'),
        # Email
        (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'), '[REDACTED_EMAIL]'),
        # Taiwan Phone Numbers (Mobile and Landline)
        (re.compile(r'(?:\+886\s*|0)9\d{2}[\s\-]?\d{3}[\s\-]?\d{3}'), '[REDACTED_PHONE]'),
        (re.compile(r'(?:\+886\s*|0)[2-8][\s\-]?\d{3,4}[\s\-]?\d{4}'), '[REDACTED_PHONE]'),
        # Taiwan National ID
        (re.compile(r'(?<![A-Za-z0-9])[A-Za-z][1289]\d{8}(?![0-9])'), '[REDACTED_ID]'),
        # Simple 8-digit number that could be an ID
        (re.compile(r'(?<!\d)\d{8}(?!\d)'), '[REDACTED_ID]')
    ]

    def __init__(self, rules: List[Tuple[re.Pattern, str]] = None):
        """
        Initializes the strategy with a list of regex rules.
        If no rules are provided, a default set is used.

        :param rules: A list of tuples, where each tuple contains a
                      compiled regex pattern and its replacement string.
        """
        self.rules = rules if rules is not None else self.DEFAULT_REGEX_RULES

    def anonymize(self, text: str, **kwargs) -> str:
        if not isinstance(text, str):
            return text
            
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
            
        return text
