import os
from typing import List

def build_repo_context(repo_path: str) -> str:
    """
    Builds a high-level context string for the repository by reading the README
    and listing the root directory structure.

    :param repo_path: The absolute path to the repository.
    :return: A formatted string containing the project overview and structure, or an empty string.
    """
    context_parts: List[str] = []

    # 1. Read README.md if it exists
    readme_content = ""
    readme_path = os.path.join(repo_path, "README.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read(2000) # Truncate to first 2000 chars
        if readme_content:
            context_parts.append(f"Project Overview (from README.md):\n{readme_content}")
    except FileNotFoundError:
        pass # README.md is optional

    # 2. Build the first-level directory tree
    dir_structure = ""
    try:
        root_items = os.listdir(repo_path)
        tree_lines = []
        for item in sorted(root_items):
            # Ignore hidden files/dirs and common noise
            if item.startswith('.') or item in ['node_modules', '__pycache__', 'venv', 'env']:
                continue
            
            item_path = os.path.join(repo_path, item)
            if os.path.isdir(item_path):
                tree_lines.append(f"  - {item}/")
            else:
                tree_lines.append(f"  - {item}")
        
        if tree_lines:
            dir_structure = "\n".join(tree_lines)
            context_parts.append(f"Top-Level Directory Structure:\n{dir_structure}")

    except OSError:
        pass # Failed to list directory

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)
