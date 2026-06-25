"""
SSH & SFTP Utilities for CEDIA HPC
===================================
Shared connection management, remote command execution, and file transfer.
"""
import os
import posixpath
import sys
from pathlib import Path

import paramiko

from cedia_config import (
    HOSTNAME, USERNAME, KEY_FILE,
    SKIP_DIR_NAMES, UPLOAD_EXTENSIONS,
)


def force_utf8_stdout() -> None:
    """Force UTF-8 encoding on Windows stdout."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def get_ssh_client() -> paramiko.SSHClient:
    """Create and return an authenticated SSH client to CEDIA."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, key_filename=KEY_FILE, timeout=30)
    print(f"✅ Conexión SSH exitosa → {USERNAME}@{HOSTNAME}")
    return ssh


def run_remote(ssh: paramiko.SSHClient, cmd: str, *, check: bool = True) -> tuple:
    """Execute a command on the remote server. Returns (exit_code, stdout, stderr)."""
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if check and code != 0:
        print(f"⚠️  Comando remoto falló (exit {code}): {cmd}", file=sys.stderr)
        if err.strip():
            print(f"   STDERR: {err.strip()}", file=sys.stderr)
    return code, out, err


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    """Recursively create remote directories (like mkdir -p)."""
    parts = [p for p in remote_dir.split("/") if p]
    current = "/" if remote_dir.startswith("/") else "."
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    """Upload a single file, creating remote directories as needed."""
    mkdir_p(sftp, posixpath.dirname(remote_path))
    sftp.put(str(local_path), remote_path)


def upload_directory(
    sftp: paramiko.SFTPClient,
    local_dir: Path,
    remote_dir: str,
    *,
    extensions: set = None,
    verbose: bool = True,
) -> int:
    """
    Recursively upload a local directory to a remote path.
    Returns the number of files uploaded.
    """
    if extensions is None:
        extensions = UPLOAD_EXTENSIONS

    uploaded = 0
    for root, dirs, files in os.walk(local_dir):
        # Filter out unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]

        for fname in sorted(files):
            if extensions and Path(fname).suffix not in extensions:
                continue

            local_path = Path(root) / fname
            rel = local_path.relative_to(local_dir).as_posix()
            remote_path = posixpath.join(remote_dir, rel)

            if verbose:
                print(f"  📤 {rel}")

            upload_file(sftp, local_path, remote_path)
            uploaded += 1

    return uploaded


def quote_remote(value: str) -> str:
    """Shell-quote a string for safe remote command execution."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def progress_bar(done: int, total: int, label: str = "") -> None:
    """Print a simple progress bar."""
    width = 30
    filled = int(width * done / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * done / max(total, 1)
    print(f"\r  [{bar}] {pct:5.1f}% ({done}/{total}) {label}", end="", flush=True)
    if done == total:
        print()
