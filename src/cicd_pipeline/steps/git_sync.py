import os
import git
from typing import Dict, Any
from .base import PipelineStep

class GitSyncStep(PipelineStep):
    """
    A pipeline step that clones or pulls a Git repository.
    """
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the repository is up-to-date and adds its path and
        commit hash to the context.

        :param context: The pipeline context.
        :return: Updated context with 'repo_local_path' and 'commit_hash'.
        """
        repo_url = self.config.git_url
        project_name = self.config.project_name
        
        # Prioritize the local Git root directory injected externally (e.g., via Jenkins)
        local_dir = self.config.local_repo_path or os.path.join(self.config.base_workspace_dir, project_name)
        
        try:
            if os.path.exists(os.path.join(local_dir, '.git')):
                self.logger.info(f"Repository exists at {local_dir}.")
                repo = git.Repo(local_dir)
                
                # If the workspace is provided by a CI system (like Jenkins), a manual pull might not be necessary,
                # but if it's a directory managed by the script itself, perform synchronization.
                if not self.config.local_repo_path:
                    self.logger.info(f"Fetching and checking out {self.config.branch}...")
                    repo.remotes.origin.fetch()
                    repo.git.checkout(self.config.branch)
                    repo.remotes.origin.pull(self.config.branch)
            else:
                self.logger.info(f"Cloning repository from {repo_url} to {local_dir}...")
                os.makedirs(local_dir, exist_ok=True)
                repo = git.Repo.clone_from(repo_url, local_dir)
                repo.git.checkout(self.config.branch)

            commit_hash = repo.head.commit.hexsha
            self.logger.info(f"Repository synced on branch {self.config.branch}. Current commit hash: {commit_hash}")
            
            merged_branches = []
            try:
                # Identify branches merged into main or master for DB cleanup
                target = None
                refs = [r.name for r in repo.remotes.origin.refs]
                if 'origin/main' in refs: target = 'origin/main'
                elif 'origin/master' in refs: target = 'origin/master'
                
                if target:
                    merged_refs = repo.git.branch('-r', '--merged', target).split('\n')
                    for ref in merged_refs:
                        ref = ref.strip()
                        if not ref or '->' in ref: continue
                        b_name = ref.replace('origin/', '', 1)
                        if b_name not in ['main', 'master']:
                            merged_branches.append(b_name)
                self.logger.info(f"Identified merged branches for cleanup: {merged_branches}")
            except Exception as e:
                self.logger.warning(f"Failed to identify merged branches: {e}")
            
            # Update the context for subsequent steps
            context['repo_local_path'] = local_dir
            context['commit_hash'] = commit_hash
            context['merged_branches'] = merged_branches
            
            return context
            
        except git.exc.GitCommandError as e:
            self.logger.error(f"Git operation failed: {e}")
            raise  # Re-raise the exception to halt the pipeline
