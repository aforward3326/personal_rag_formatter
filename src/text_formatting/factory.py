from typing import Optional

from . import config
from .strategies.base import TextFormatterStrategy
from .strategies.gmail_formatter import GmailFormatter
from .strategies.chat_html_formatter import ChatHtmlFormatter
from .strategies.meta_formatter import MetaFormatter
from .strategies.corpus_builder import CorpusBuilder

class FormatterFactory:
    """
    Factory to create the appropriate text formatting strategy based on a key.
    """

    @staticmethod
    def create_formatter(formatter_name: str) -> Optional[TextFormatterStrategy]:
        """
        Selects and instantiates a formatter strategy.

        :param formatter_name: The name of the formatter to create
                               (e.g., 'gmail', 'chat', 'meta', 'corpus').
        :return: An instance of a TextFormatterStrategy, or None if the
                 name is not recognized.
        """
        formatter_name = formatter_name.lower()

        # Common progress file for all formatters running under this process
        progress_file = config.PROC_DIR / f"text_formatting_{formatter_name}_progress.json"

        if formatter_name == 'gmail':
            return GmailFormatter(
                input_dir=config.GMAIL_INPUT_MBOX_DIR,
                output_file=config.GMAIL_OUTPUT_FILE,
                my_email=config.GMAIL_MY_EMAIL,
                progress_file=progress_file
            )
        elif formatter_name == 'chat':
            return ChatHtmlFormatter(
                input_dir=config.CHAT_INPUT_HTML_DIR,
                output_file=config.CHAT_OUTPUT_FILE,
                progress_file=progress_file
            )
        elif formatter_name == 'meta':
            return MetaFormatter(
                input_dir=config.META_INPUT_DIR,
                output_file=config.META_OUTPUT_FILE,
                progress_file=progress_file
            )
        elif formatter_name == 'corpus':
            return CorpusBuilder(
                input_dir=config.CORPUS_INPUT_DIR,
                output_file=config.CORPUS_OUTPUT_FILE,
                progress_file=progress_file
            )
        else:
            print(f"Error: Unknown formatter name '{formatter_name}'")
            return None
