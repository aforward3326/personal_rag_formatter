import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

from .base import TextFormatterStrategy
from ..utils.file_utils import save_json_batch, load_progress, save_progress, clear_progress, prompt_resume

class MetaFormatter(TextFormatterStrategy):
    """
    Processes exported data from Meta platforms (Facebook, Instagram),
    parsing posts, comments, and messages into a unified RAG-ready format.
    """

    def process(self):
        self.logger.info("Process Started: Meta Data Formatter")
        if not self.input_dir.exists():
            self.logger.error(f"Directory {self.input_dir} not found.")
            return

        progress_data = load_progress(self.progress_file)
        if not prompt_resume(progress_data, self.logger):
            self.logger.info("Starting fresh, clearing previous progress.")
            clear_progress(self.progress_file)
            progress_data = {"processed_files": set()}
        else:
            self.logger.info("Resuming process...")

        processed_files = set(progress_data.get("processed_files", []))
        all_posts, all_messages, all_comments = [], [], []

        try:
            all_files = [p for p in self.input_dir.rglob('*.json')]
            self.logger.info(f"Found {len(all_files)} JSON files to process.")

            for file_path in all_files:
                file_rel_path = str(file_path.relative_to(self.input_dir))
                if file_rel_path in processed_files:
                    continue

                self.logger.info(f"Processing: {file_rel_path}")
                source = self._get_source_from_path(file_path)
                
                if "message" in file_path.name or "inbox" in str(file_path).lower():
                    thread = self._parse_messages(file_path, source)
                    if thread: all_messages.append(thread)
                elif "comment" in file_path.name:
                    all_comments.extend(self._parse_comments(file_path))
                elif "post" in file_path.name or "profile" in file_path.name:
                    all_posts.extend(self._parse_posts(file_path, source))

                processed_files.add(file_rel_path)
                progress_data["processed_files"] = list(processed_files)
                save_progress(self.progress_file, progress_data)

            self._link_comments_to_posts(all_posts, all_comments)
            
            # Save combined data in a single batch file
            combined_data = {"posts": all_posts, "messages": all_messages}
            save_json_batch(combined_data, self.output_dir, self.output_file.name, "meta_data")

            self.logger.info("Process Finished Successfully.")
            clear_progress(self.progress_file)

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user. Progress saved.")
            save_progress(self.progress_file, progress_data)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            save_progress(self.progress_file, progress_data)

    def _get_source_from_path(self, path: Path) -> str:
        path_str = str(path).lower()
        if "facebook" in path_str: return "facebook"
        if "instagram" in path_str: return "instagram"
        if "threads" in path_str: return "threads"
        return "unknown"

    def _fix_encoding(self, text: str) -> str:
        try: return text.encode('latin1').decode('utf-8')
        except: return text

    def _to_iso8601(self, ts) -> str:
        try:
            ts = float(ts)
            if ts > 1e11: ts /= 1000 # Convert ms to s
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except: return ""

    def _parse_posts(self, file_path: Path, source: str) -> list:
        posts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            raw_posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
            for rp in raw_posts:
                if not isinstance(rp, dict): continue
                content = self._fix_encoding(next((d.get("post", "") for d in rp.get("data", []) if "post" in d), rp.get("title", "")))
                if content:
                    posts.append({
                        "post_id": str(uuid.uuid4()),
                        "page_content": content,
                        "metadata": {"source": source, "timestamp": self._to_iso8601(rp.get("timestamp"))},
                        "comments": []
                    })
        except Exception as e: self.logger.error(f"Error parsing post {file_path.name}: {e}")
        return posts

    def _parse_comments(self, file_path: Path) -> list:
        comments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            raw_comments = data.get("comments_v2", data.get("comments", data if isinstance(data, list) else []))
            for rc in raw_comments:
                if not isinstance(rc, dict): continue
                comment_data = next((d.get("comment", {}) for d in rc.get("data", []) if "comment" in d), {})
                content = self._fix_encoding(comment_data.get("comment", ""))
                if content:
                    comments.append({
                        "comment_id": str(uuid.uuid4()),
                        "page_content": content,
                        "metadata": {"author": self._fix_encoding(comment_data.get("author", "")), "timestamp": self._to_iso8601(rc.get("timestamp"))},
                        "target_title": self._fix_encoding(rc.get("title", ""))
                    })
        except Exception as e: self.logger.error(f"Error parsing comments {file_path.name}: {e}")
        return comments

    def _parse_messages(self, file_path: Path, source: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if not isinstance(data, dict): return None
            
            conversation = []
            for msg in data.get("messages", []):
                content = self._fix_encoding(msg.get("content", ""))
                if content:
                    conversation.append({
                        "message_id": str(uuid.uuid4()),
                        "page_content": content,
                        "metadata": {"sender_name": self._fix_encoding(msg.get("sender_name", "")), "timestamp": self._to_iso8601(msg.get("timestamp_ms"))}
                    })
            
            if conversation:
                return {
                    "thread_id": str(uuid.uuid4()),
                    "metadata": {"source": f"{source}_message", "participants": [self._fix_encoding(p.get("name", "")) for p in data.get("participants", [])]},
                    "conversation": conversation[::-1] # Reverse for chronological order
                }
        except Exception as e: self.logger.error(f"Error parsing messages {file_path.name}: {e}")
        return None

    def _link_comments_to_posts(self, posts: list, comments: list):
        self.logger.info(f"Attempting to link {len(comments)} comments to {len(posts)} posts.")
        for comment in comments:
            target_title = comment.pop("target_title", "")
            if not target_title: continue
            for post in posts:
                if post.get("page_content", "").startswith(target_title[:50]):
                    post["comments"].append(comment)
                    break
