import os
import json
import uuid
import re
from pathlib import Path
from bs4 import BeautifulSoup

from .base import TextFormatterStrategy
from ..utils.file_utils import save_json_batch, load_progress, save_progress, clear_progress, prompt_resume

class ChatHtmlFormatter(TextFormatterStrategy):
    """
    Processes exported HTML chat logs from various platforms (LINE, WhatsApp, etc.),
    extracts conversations, and formats them for RAG ingestion.
    """

    def process(self):
        self.logger.info("Process Started: Chat HTML Formatter")
        if not self.input_dir.exists():
            self.logger.error(f"Directory {self.input_dir} not found.")
            return

        progress_data = load_progress(self.progress_file)
        if not prompt_resume(progress_data, self.logger):
            self.logger.info("Starting fresh, clearing previous progress.")
            clear_progress(self.progress_file)
            progress_data = {"processed_files": [], "total_html_files": 0}
        else:
            self.logger.info(f"Resuming process...")

        processed_files = set(progress_data.get("processed_files", []))
        batch_results = []

        try:
            all_files = list(self.input_dir.rglob('*.html'))
            self.logger.info(f"Found {len(all_files)} HTML files to process.")

            for file_path in all_files:
                file_rel_path = str(file_path.relative_to(self.input_dir))
                if file_rel_path in processed_files:
                    continue

                self.logger.info(f"Processing: {file_rel_path}")
                thread_data = self._parse_html_chat(file_path)
                if thread_data and thread_data["conversation"]:
                    batch_results.append(thread_data)
                
                processed_files.add(file_rel_path)
                progress_data["processed_files"] = list(processed_files)
                save_progress(self.progress_file, progress_data)

                if len(batch_results) >= 100: # Using a fixed batch size for now
                    save_json_batch(batch_results, self.output_dir, self.output_file.name, "messages")
                    batch_results = []

            if batch_results:
                save_json_batch(batch_results, self.output_dir, self.output_file.name, "messages")

            self.logger.info("Process Finished Successfully.")
            clear_progress(self.progress_file)

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user. Progress saved.")
            save_progress(self.progress_file, progress_data)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            save_progress(self.progress_file, progress_data)

    def _parse_html_chat(self, file_path: Path) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

        # Basic cleanup
        for tag in soup(["nav", "aside", "header", "footer", "script", "style"]):
            tag.decompose()

        source, thread_id = self._determine_source_and_thread(soup, file_path)
        
        conversation = []
        if source == "line":
            conversation = self._parse_line_chat(soup)
        elif source == "whatsapp":
            conversation = self._parse_whatsapp_chat(soup)
        else: # Fallback for generic structures
            conversation = self._parse_generic_chat(soup)

        return {
            "thread_id": thread_id,
            "metadata": {"source": source},
            "conversation": conversation
        } if conversation else None

    def _determine_source_and_thread(self, soup: BeautifulSoup, file_path: Path) -> tuple:
        source = "unknown"
        thread_id = file_path.stem
        title = (soup.title.string or "").strip().lower()
        path_str = str(file_path).lower()

        if "line" in title or "line" in path_str:
            source = "line"
            to_element = soup.find(string=re.compile(r"^TO:\s*(.*)"))
            if to_element and (match := re.search(r"^TO:\s*(.*)", to_element)):
                thread_id = match.group(1).strip()
        elif "whatsapp" in title or "whatsapp" in path_str:
            source = "whatsapp"
            if title and title != "whatsapp messages":
                thread_id = soup.title.string.strip()
        
        return source, thread_id

    def _extract_content(self, element: BeautifulSoup) -> tuple:
        media_attachments = [img.get("src") for img in element.find_all("img") if img.get("src")]
        text_content = element.get_text(separator=' ', strip=True)
        content_type = "image" if media_attachments else "text"
        # Basic content type detection based on keywords
        if content_type == "text":
            lower_text = text_content.lower()
            if any(k in lower_text for k in ["[相片]", "[圖片]", "[貼圖]", "image omitted"]): content_type = "image"
            elif any(k in lower_text for k in ["[語音訊息]", "[語音]", "audio omitted"]): content_type = "voice"
            elif "[影片]" in lower_text or "video omitted" in lower_text: content_type = "video"
        return content_type, media_attachments, text_content

    def _parse_line_chat(self, soup: BeautifulSoup) -> list:
        conversation = []
        current_time, current_sender = "", "Unknown"
        container = soup.find(class_='content') or soup
        for element in container.children:
            if not hasattr(element, 'name'): continue
            if element.name == 'p':
                if element.get('align') == 'center' and re.search(r'\d{4}/\d{2}/\d{2}', element.text):
                    current_time = element.text.strip()
                elif element.get('align') == 'left' and not element.get('class'):
                    current_sender = element.text.strip()
                elif any('triangle' in c for c in element.get('class', [])):
                    content_type, _, text = self._extract_content(element)
                    if text and content_type == "text":
                        conversation.append(self._create_message(text, current_sender, current_time, content_type))
        return conversation

    def _parse_whatsapp_chat(self, soup: BeautifulSoup) -> list:
        # Simplified logic for WhatsApp, can be expanded
        return self._parse_generic_chat(soup)

    def _parse_generic_chat(self, soup: BeautifulSoup) -> list:
        conversation = []
        msg_blocks = soup.find_all(attrs={"class": re.compile(r"message|msg|chat|bubble", re.I)})
        if not msg_blocks: msg_blocks = soup.find_all(["div", "p", "li"])

        for block in msg_blocks:
            sender = (block.find(attrs={"class": re.compile(r"name|sender|author", re.I)}) or "Unknown")
            if hasattr(sender, 'get_text'): sender = sender.get_text(strip=True)
            
            time_el = block.find(attrs={"class": re.compile(r"time|date", re.I)})
            time_str = time_el.get_text(strip=True) if time_el else ""
            
            content_el = block.find(attrs={"class": re.compile(r"text|content", re.I)}) or block
            content_type, _, text = self._extract_content(content_el)

            if text and content_type == "text":
                # Clean up text that includes sender/time
                if sender != "Unknown" and text.startswith(sender): text = text[len(sender):].strip()
                if time_str and text.endswith(time_str): text = text[:-len(time_str)].strip()
                if text: conversation.append(self._create_message(text, sender, time_str, content_type))
        return conversation

    def _create_message(self, text: str, sender: str, timestamp: str, content_type: str) -> dict:
        return {
            "message_id": str(uuid.uuid4()),
            "page_content": text,
            "metadata": {
                "sender_name": sender,
                "timestamp": timestamp,
                "content_type": content_type
            }
        }
