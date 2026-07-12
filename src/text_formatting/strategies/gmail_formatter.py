import mailbox
import email.utils
import re
import json
import hashlib
import warnings
from email.header import decode_header
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .base import TextFormatterStrategy
from ..utils.file_utils import save_json_batch, load_progress, save_progress, clear_progress, prompt_resume

# Ignore BeautifulSoup XML parsing warnings which can be noisy
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class GmailFormatter(TextFormatterStrategy):
    """
    Processes .mbox files from Google Takeout, extracts email conversations,
    filters for two-way interactions, and formats them for RAG ingestion.
    """
    
    def __init__(self, input_dir: Path, output_file: Path, my_email: str, **kwargs):
        super().__init__(input_dir, output_file, **kwargs)
        self.my_email = my_email.lower()

    def process(self):
        self.logger.info("Process Started: Gmail Mbox Formatter")
        if not self.input_dir.exists():
            self.logger.error(f"Cannot find MBOX directory: {self.input_dir}")
            return

        progress_data = load_progress(self.progress_file)
        if not prompt_resume(progress_data, self.logger):
            self.logger.info("Starting fresh, clearing previous progress.")
            clear_progress(self.progress_file)
            progress_data = self._get_initial_progress()
        else:
            self.logger.info(f"Resuming from file {progress_data.get('current_file')}...")

        grouped_threads = progress_data["grouped_threads"]
        all_mbox_files = list(self.input_dir.glob("*.mbox"))

        try:
            for mbox_file_path in all_mbox_files:
                if self._is_file_processed(mbox_file_path, progress_data):
                    continue
                
                self._process_single_mbox(mbox_file_path, progress_data, grouped_threads)

            self.logger.info(f"Email reading completed. Found {len(grouped_threads)} threads. Filtering...")
            self._finalize_and_save(grouped_threads)
            clear_progress(self.progress_file)
            self.logger.info("Process Finished Successfully.")

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user. Progress saved.")
            save_progress(self.progress_file, progress_data)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            save_progress(self.progress_file, progress_data)

    def _get_initial_progress(self):
        return {"processed_files": [], "grouped_threads": {}, "total_messages_parsed": 0, "current_file": None, "current_msg_idx": 0}

    def _is_file_processed(self, file_path: Path, progress: dict) -> bool:
        file_name_str = str(file_path.name)
        return file_name_str in progress["processed_files"] and progress["current_file"] != file_name_str

    def _process_single_mbox(self, mbox_file_path: Path, progress: dict, threads: dict):
        file_name_str = str(mbox_file_path.name)
        if progress["current_file"] != file_name_str:
            progress["current_file"] = file_name_str
            progress["current_msg_idx"] = 0

        self.logger.info(f"Parsing MBOX file: {mbox_file_path.name}")
        mbox = mailbox.mbox(mbox_file_path)
        
        for idx, msg in enumerate(mbox):
            if idx < progress["current_msg_idx"]:
                continue

            progress["total_messages_parsed"] += 1
            if progress["total_messages_parsed"] % 1000 == 0:
                self.logger.info(f"Scanned {progress['total_messages_parsed']} emails...")
                save_progress(self.progress_file, progress)

            message_data = self._parse_message(msg, mbox_file_path.name)
            if message_data:
                subject = message_data.pop("cleaned_subject")
                if subject not in threads:
                    threads[subject] = []
                threads[subject].append(message_data)
        
        progress["processed_files"].append(file_name_str)
        progress["current_file"] = None
        progress["current_msg_idx"] = 0
        save_progress(self.progress_file, progress)

    def _parse_message(self, msg, source_file: str) -> dict:
        raw_subject = msg.get('Subject', '')
        cleaned_subject = self._clean_subject(raw_subject)
        
        page_content = self._clean_content(self._extract_body(msg))
        if not page_content:
            return None

        sender_email = self._extract_email(msg.get('From', ''))
        
        return {
            "cleaned_subject": cleaned_subject,
            "message_id": msg.get('Message-ID', f"msg_{hashlib.md5(f'{cleaned_subject}{msg.get_content_type()}'.encode()).hexdigest()}").strip('<>'),
            "page_content": page_content,
            "metadata": {
                "sender": sender_email,
                "receiver": self._extract_email(msg.get('To', '')),
                "timestamp": self._parse_date_to_iso(msg.get('Date', '')),
                "is_me": self.my_email == sender_email,
                "source_file": source_file
            }
        }

    def _finalize_and_save(self, threads: dict):
        final_threads = []
        for subject, conversation in threads.items():
            if any(msg['metadata']['is_me'] for msg in conversation):
                conversation.sort(key=lambda x: x['metadata']['timestamp'] or "")
                final_threads.append({
                    "thread_id": hashlib.md5(subject.encode('utf-8')).hexdigest(),
                    "subject": subject,
                    "metadata": {"source": "gmail", "total_messages": len(conversation)},
                    "conversation": conversation
                })
        
        self.logger.info(f"Retained {len(final_threads)} threads with two-way interaction.")
        save_json_batch(final_threads, self.output_dir, self.output_file.name, batch_key="email_threads")

    def _decode_mime_header(self, h_val: str) -> str:
        if not h_val: return ""
        try:
            parts = decode_header(h_val)
            return "".join(p.decode(c or 'utf-8', 'replace') if isinstance(p, bytes) else str(p) for p, c in parts).strip()
        except Exception: return str(h_val)

    def _clean_subject(self, subject_raw: str) -> str:
        subject = self._decode_mime_header(subject_raw)
        if not subject: return "No Subject"
        pattern = re.compile(r'^(re|fwd|fw|回覆|轉寄)\s*[:：]\s*', re.IGNORECASE)
        while pattern.search(subject): subject = pattern.sub('', subject).strip()
        return subject or "No Subject"

    def _parse_date_to_iso(self, date_str: str) -> str:
        if not date_str: return ""
        try:
            parsed_tuple = email.utils.parsedate_tz(date_str)
            if parsed_tuple: return datetime.fromtimestamp(email.utils.mktime_tz(parsed_tuple), tz=timezone.utc).isoformat()
        except Exception: pass
        return ""

    def _extract_email(self, addr_str: str) -> str:
        if not addr_str: return ""
        return email.utils.parseaddr(self._decode_mime_header(addr_str))[1].lower()

    def _extract_body(self, msg) -> str:
        body_text, body_html = "", ""
        if msg.is_multipart():
            for part in msg.walk():
                if 'attachment' in str(part.get('Content-Disposition')) and part.get_content_type() not in ['text/plain', 'text/html']:
                    continue
                if part.get_content_type() == 'text/plain':
                    try: body_text += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'replace')
                    except Exception: pass
                elif part.get_content_type() == 'text/html':
                    try: body_html += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'replace')
                    except Exception: pass
        else:
            try:
                payload = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', 'replace')
                if msg.get_content_type() == 'text/plain': body_text = payload
                elif msg.get_content_type() == 'text/html': body_html = payload
            except Exception: pass
        return body_text.strip() or (BeautifulSoup(body_html, 'html.parser').get_text(separator='\n') if body_html.strip() else "")

    def _clean_content(self, text: str) -> str:
        if not text: return ""
        lines = BeautifulSoup(text, 'html.parser').get_text(separator='\n').split('\n')
        filtered_lines = []
        quote_patterns = [re.compile(r'^On\s+.+?wrote:$', re.I), re.compile(r'^-{3,}\s*Original Message\s*-{3,}', re.I), re.compile(r'^_{3,}$'), re.compile(r'^From:\s+.*<.+@.+>', re.I)]
        sig_patterns = [re.compile(r'^Sent from my (iPhone|iPad|Android)', re.I), re.compile(r'^從我的\s*(iPhone|iPad|Android)\s*傳送', re.I)]
        for line in lines:
            s_line = line.strip()
            if any(p.match(s_line) for p in quote_patterns) or s_line.startswith('>') or any(p.match(s_line) for p in sig_patterns):
                break
            filtered_lines.append(line)
        return re.sub(r'\n{3,}', '\n\n', '\n'.join(filtered_lines).strip())
