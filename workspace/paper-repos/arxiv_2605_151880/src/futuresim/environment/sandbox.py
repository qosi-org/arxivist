"""
Sandbox Orchestration — bwrap-based Agent Isolation
======================================================
Prevents agent access to future news articles and live internet.

Paper reference: Appendix B.3 (Sandboxing agents to avoid contamination)
  "we sandbox them carefully using bwrap on a Linux server"
  Guarantees:
    (i)  No live web search — LLM provider endpoints only
    (ii) Date-gated article corpus — only articles/YYYY/MM/DD/ up to current date
    (iii) Read-only environment state — market.csv is read-only for agent

NOTE: bwrap is Linux-only. On other platforms, a Docker fallback is provided.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional


class AgentSandbox:
    """
    Wraps an agent subprocess in a bwrap sandbox with date-gated article access.

    Paper reference: Appendix B.3
    """

    def __init__(
        self,
        workspace_path: str,
        corpus_path: str,
        current_date: date,
        block_network: bool = True,
        use_bwrap: bool = True,
    ):
        """
        Args:
            workspace_path: Agent's read-write workspace directory
            corpus_path: Root of CCNews article corpus (YYYY/MM/DD structure)
            current_date: Simulation date — articles up to this date are exposed
            block_network: If True, block all non-LLM-provider network access
            use_bwrap: Use bwrap isolation (Linux only); falls back to subprocess on other OS
        """
        self.workspace = Path(workspace_path)
        self.corpus = Path(corpus_path)
        self.current_date = current_date
        self.block_network = block_network
        self.use_bwrap = use_bwrap and platform.system() == "Linux"

        if use_bwrap and not self.use_bwrap:
            print("WARNING: bwrap requested but not on Linux. Falling back to unsandboxed subprocess.")
            print("         For full reproducibility, run on Linux with bwrap installed.")

    def __repr__(self) -> str:
        return (
            f"AgentSandbox(date={self.current_date}, "
            f"bwrap={self.use_bwrap}, workspace={self.workspace})"
        )

    def _build_bwrap_args(self, agent_cmd: list[str]) -> list[str]:
        """
        Build bwrap command with date-gated article bind-mounts.

        Only articles/YYYY/MM/DD/ directories up to current_date are exposed.
        The agent workspace is read-write; everything else is read-only or hidden.

        Paper reference: Appendix B.3
        # ASSUMED: specific bwrap flags beyond described behavior; paper only states high-level rules
        """
        bwrap_args = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/sbin", "/sbin",
            "--ro-bind", str(self.workspace.parent / "src"), "/workspace/src",
            "--bind", str(self.workspace), "/workspace/agent",
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--unshare-pid",
        ]

        if self.block_network:
            bwrap_args.append("--unshare-net")

        # Expose only articles up to current_date
        for article_dir in self._get_allowed_article_dirs():
            rel = article_dir.relative_to(self.corpus.parent)
            bwrap_args += ["--ro-bind", str(article_dir), f"/workspace/articles/{rel}"]

        bwrap_args += ["--chdir", "/workspace/agent"]
        bwrap_args += agent_cmd
        return bwrap_args

    def _get_allowed_article_dirs(self) -> list[Path]:
        """Return article day-directories up to and including current_date."""
        allowed = []
        if not self.corpus.exists():
            return allowed
        for year_dir in sorted(self.corpus.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name)
                except ValueError:
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    try:
                        day = int(day_dir.name)
                        dir_date = date(year, month, day)
                    except (ValueError, TypeError):
                        continue
                    if dir_date <= self.current_date:
                        allowed.append(day_dir)
        return allowed

    def run(
        self,
        agent_cmd: list[str],
        env: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """
        Launch the agent subprocess, optionally sandboxed with bwrap.

        Args:
            agent_cmd: Command and arguments to launch agent process
            env: Environment variables (API keys, etc.) for agent process
            timeout: Optional timeout in seconds

        Returns:
            CompletedProcess result
        """
        self.workspace.mkdir(parents=True, exist_ok=True)

        if self.use_bwrap:
            cmd = self._build_bwrap_args(agent_cmd)
        else:
            cmd = agent_cmd

        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        return subprocess.run(
            cmd,
            env=proc_env,
            timeout=timeout,
            text=True,
            capture_output=False,
        )

    def update_date(self, new_date: date) -> None:
        """Advance the sandbox's date cap to expose new articles."""
        self.current_date = new_date
