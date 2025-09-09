import argparse
# import subprocess
import sys
# import time
from slurm_util.utils import get_cluster, get_job_nodes
import subprocess
import shutil
import os

def _expand_first_hostname(nodelist_expr: str) -> str | None:
    try:
        result = subprocess.run([
            "scontrol", "show", "hostnames", nodelist_expr
        ], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    # Fallback: naive parse like foo[01-02] -> foo
    return nodelist_expr.split(',')[0].split('[')[0]

def _detect_editor_cli() -> str | None:
    # Prefer Cursor if available, else VS Code
    if shutil.which("cursor") is not None:
        return "cursor"
    if shutil.which("code") is not None:
        return "code"
    return None

def _launch_remote_editor(editor_cli: str, remote_authority: str, working_dir: str) -> None:
    # Try to reuse an existing window/session if possible
    # cmd = [editor_cli, "--remote", f"ssh-remote+{remote_authority}", working_dir]
    cmd = [editor_cli, "--folder-uri", f"vscode-remote://ssh-remote+{remote_authority}/{working_dir}"]
    
    # Run detached; don't block the CLI
    try:
        subprocess.Popen(cmd)
    except FileNotFoundError:
        pass

def attach(job_id, cluster):
    print(f"Waiting for job {job_id} to be allocated nodes...")
    nodes = get_job_nodes(job_id)
    if not nodes:
        print(f"Job {job_id} submitted, but node information not yet available.")
        print(f"Check job status with: squeue -j {job_id}")
        return 0
    first_host = _expand_first_hostname(nodes)
    # Determine destination port per cluster
    ssh_port = cluster.get_ssh_port(job_id)
    user = os.environ.get("USER", "")
    editor_cli = _detect_editor_cli()
    jump = cluster.name
    if not editor_cli:
        print("Neither 'cursor' nor 'code' CLI found in PATH. Please install one to open a remote session.")
        print(f"Manual SSH (via jump host): ssh -J {jump} -t -p {ssh_port} {user}@{first_host} tmux attach -t {job_id}")
        return 0
    remote_authority = f"{user}@{cluster.name}-{first_host}-p{ssh_port}"
    working_dir = os.getcwd()
    print(f"Opening {editor_cli} remote: ssh-remote+{remote_authority} in {working_dir}")
    _launch_remote_editor(editor_cli, remote_authority, working_dir)
    print("If the remote editor fails to connect, try this locally:")
    print(f"ssh -J {jump} -t -p {ssh_port} {user}@{first_host} tmux attach -t {job_id}")


def main():
    parser = argparse.ArgumentParser(description="Wait for a SLURM job to complete")
    parser.add_argument("--job", "-j", type=str, help="SLURM job ID")
    args = parser.parse_args()
    cluster = get_cluster()
    attach(args.job, cluster)

if __name__ == "__main__":
    sys.exit(main())