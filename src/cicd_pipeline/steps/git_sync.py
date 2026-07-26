import os
import git
from typing import Dict, Any, List
from .base import PipelineStep

class GitSyncStep(PipelineStep):
    """
    A pipeline step that clones or pulls a Git repository and identifies changed files
    for incremental processing.
    """

    def _get_changed_files(self, repo: git.Repo, is_new_clone: bool) -> List[str]:
        """
        Identifies changed files based on the git diff between the current and previous commits.
        If it's a new clone or the history is shallow, it returns all files.
        """
        if is_new_clone:
            self.logger.info("New clone detected. Processing all files in the repository.")
            return [os.path.join(repo.working_dir, f) for f in repo.git.ls_files().splitlines()]

        try:
            previous_commit = repo.commit("HEAD~1")
            current_commit = repo.head.commit
            
            diff_index = previous_commit.diff(current_commit)
            
            changed_files = []
            # A: Added, M: Modified. We ignore D: Deleted files.
            for diff_item in diff_index.iter_change_type('A'):
                changed_files.append(os.path.join(repo.working_dir, diff_item.b_path))
            for diff_item in diff_index.iter_change_type('M'):
                changed_files.append(os.path.join(repo.working_dir, diff_item.b_path))
            
            if not changed_files:
                self.logger.info("No new or modified files detected since the last commit.")
            else:
                self.logger.info(f"Detected {len(changed_files)} changed files for incremental processing.")
                for f in changed_files[:5]: # Log a sample of changed files
                    self.logger.debug(f"  - Changed (sample): {f}")

            return changed_files
        except git.exc.BadName:
            self.logger.warning("Could not find previous commit (HEAD~1). This might be the first commit or a shallow clone. Processing all files.")
            return [os.path.join(repo.working_dir, f) for f in repo.git.ls_files().splitlines()]
        except Exception as e:
            self.logger.error(f"An error occurred while calculating git diff: {e}", exc_info=True)
            self.logger.warning("Falling back to processing all files due to an unexpected diff error.")
            return [os.path.join(repo.working_dir, f) for f in repo.git.ls_files().splitlines()]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the repository is up-to-date, identifies changed files, and updates the context.

        :param context: The pipeline context.
        :return: Updated context with 'repo_local_path', 'commit_hash', 'changed_files', and 'merged_branches'.
        """
        repo_url = self.config.git_url
        project_name = self.config.project_name
        
        local_dir = self.config.local_repo_path or os.path.join(self.config.base_workspace_dir, project_name)
        
        is_new_clone = False
        try:
            if os.path.exists(os.path.join(local_dir, '.git')):
                self.logger.info(f"Repository exists at {local_dir}.")
                repo = git.Repo(local_dir)
                
                if not self.config.local_repo_path:
                    self.logger.info(f"Fetching and checking out branch '{self.config.branch}'...")
                    repo.remotes.origin.fetch()
                    repo.git.checkout(self.config.branch)
                    repo.remotes.origin.pull(self.config.branch)
            else:
                self.logger.info(f"Cloning repository from {repo_url} into {local_dir}...")
                os.makedirs(local_dir, exist_ok=True)
                # Clone with a depth of 2 to ensure we have HEAD and HEAD~1 for diffing
                repo = git.Repo.clone_from(repo_url, local_dir, branch=self.config.branch, depth=2)
                is_new_clone = True

            commit_hash = repo.head.commit.hexsha
            self.logger.info(f"Repository synced on branch '{self.config.branch}'. Current commit hash: {commit_hash}")
            
            # Identify changed files for incremental processing
            changed_files = self._get_changed_files(repo, is_new_clone)
            
            merged_branches = []
            try:
                # Identify branches merged into main/master for potential DB cleanup
                main_ref = 'origin/main' if 'origin/main' in [r.name for r in repo.remotes.origin.refs] else 'origin/master'
                if main_ref:
                    merged_refs = repo.git.branch('-r', '--merged', main_ref).split('\n')
                    for ref in merged_refs:
                        ref = ref.strip()
                        if not ref or '->' in ref: continue
                        branch_name = ref.replace('origin/', '', 1)
                        if branch_name not in ['main', 'master']:
                            merged_branches.append(branch_name)
                    self.logger.info(f"Identified merged branches for potential cleanup: {merged_branches}")
            except Exception as e:
                self.logger.warning(f"Could not identify merged branches: {e}")
            
            # Update the context for subsequent steps
            context['repo_local_path'] = local_dir
            context['commit_hash'] = commit_hash
            context['changed_files'] = changed_files
            context['merged_branches'] = merged_branches
            
            return context
            
        except git.exc.GitCommandError as e:
            self.logger.error(f"A Git command failed: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during the Git sync process: {e}", exc_info=True)
            raise
