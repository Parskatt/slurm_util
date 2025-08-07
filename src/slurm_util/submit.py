from dataclasses import dataclass
import tyro
import subprocess
import sys
import time
import re
import os
from abc import ABC, abstractmethod

class Cluster(ABC):
    @abstractmethod
    def resource_alloc(self, *, gpus_per_node, cpus_per_node, nodes) -> str:
        pass

    @abstractmethod
    def ssh_setup(self, *, no_ssh, custom_ssh_port) -> str:
        pass

def trim_whitespace(s):
    return "\n".join([line.strip() for line in s.split("\n") if line.strip()])

class Alvis(Cluster):
    """https://www.nsc.liu.se/support/systems/alvis/#21-resource-allocation-guidelines"""
    def resource_alloc(self, *, gpus_per_node, gpu_model, cpus_per_node, nodes) -> str:
        gpu_alloc = f"--gpus-per-node {gpus_per_node}" if gpu_model != "NOGPU" else "-C NOGPU"
        cpu_alloc = f"--cpus-per-task {cpus_per_node*nodes}"
        node_alloc = f"--nodes {nodes}"
        return trim_whitespace(f"""
            #SBATCH {gpu_alloc}
            #SBATCH {cpu_alloc}
            #SBATCH {node_alloc}
                """)

    def ssh_setup(self, *, no_ssh, custom_ssh_port) -> str:
        # alvis supports ssh by default
        return ""

    def get_ssh_port(self, _):
        return 22

class Berzelius(Cluster):
    """https://www.nsc.liu.se/support/systems/berzelius-gpu/#21-resource-allocation-guidelines"""
    def resource_alloc(self, *, gpus_per_node, gpu_model, cpus_per_node, nodes) -> str:
        gpu_alloc = f"#SBATCH --gpus-per-node {gpus_per_node}" if gpu_model != "NOGPU" else "--partition=berzelius-cpu"
        if "80GB" in gpu_model:
            gpu_alloc += "\n#SBATCH -C fat"
        elif "40GB" in gpu_model:
            gpu_alloc += "\n#SBATCH -C thin"
        elif "1g.10gb" in gpu_model:
            gpu_alloc += "\n#SBATCH --reservation=1g.10gb"
        cpu_alloc = f"#SBATCH --cpus-per-gpu {cpus_per_node // gpus_per_node}" if gpu_model != "NOGPU" else ""
        node_alloc = f"#SBATCH --nodes {nodes}"
        alloc_str = trim_whitespace(f"""
            {gpu_alloc}
            {cpu_alloc}
            {node_alloc}
                """)
        return alloc_str
    
    def ssh_setup(self, *, no_ssh, custom_ssh_port) -> str:
        if no_ssh:
            return ""
        
        # Use job-specific directory and port calculation
        return trim_whitespace("""
            # Calculate unique SSH port based on job ID (base port 10000 + job_id % 55000)
            SSH_PORT=$((10000 + $SLURM_JOB_ID % 55000))
            JOB_SSH_DIR="/tmp/slurm_ssh_$SLURM_JOB_ID"
            
            # Create job-specific SSH directory
            mkdir -p "$JOB_SSH_DIR"
            
            # Generate SSH host key if it doesn't exist
            if [ ! -f "$JOB_SSH_DIR/ssh_host_key" ]; then
                ssh-keygen -t rsa -f "$JOB_SSH_DIR/ssh_host_key" -N '' -q
            fi
            
            # Create sshd_config for this job
            cat > "$JOB_SSH_DIR/sshd_config" << EOF
Port $SSH_PORT
PidFile $JOB_SSH_DIR/sshd.pid
HostKey $JOB_SSH_DIR/ssh_host_key
AuthorizedKeysFile ~/.ssh/authorized_keys
PasswordAuthentication no
PubkeyAuthentication yes
ChallengeResponseAuthentication no
Subsystem sftp internal-sftp
EOF
            
            # Start SSH daemon in background
            /usr/sbin/sshd -f "$JOB_SSH_DIR/sshd_config" -D &
            
            # Store the SSH port for later use
            echo "$SSH_PORT" > "$JOB_SSH_DIR/ssh_port"
            """).strip()

    def get_ssh_port(self, job_id):
        return 10000 + (int(job_id) % 55000)

def format_in_box(text, line_width=76):
    """Format text in a box with specified line width."""
    box_width = line_width + 2  # +2 for the border characters
    lines = []
    lines.append("┌" + "─" * box_width + "┐")
    
    for line in text.strip().split('\n'):
        if len(line) <= line_width:
            lines.append(f"│ {line:<{line_width}} │")
        else:
            # Wrap long lines
            while len(line) > line_width:
                lines.append(f"│ {line[:line_width]:<{line_width}} │")
                line = line[line_width:]
            if line:  # Print remaining part if any
                lines.append(f"│ {line:<{line_width}} │")
    
    lines.append("└" + "─" * box_width + "┘")
    return '\n'.join(lines)


def wrap_in_sbatch(
    *,
    command,
    account,
    gpus_per_node,
    gpu_model,
    cpus_per_node,
    no_ssh,
    nodes,
    time_alloc,
    num_tasks,
    shell_env,
    interactive,
    stdout_path,
    cluster,
):
    stdout_file = stdout_path + "/%A.out"
    os.makedirs(stdout_path, exist_ok=True)
    stdout_str = f"#SBATCH -o {stdout_file}"
    ssh_setup_str = cluster.ssh_setup(no_ssh = no_ssh, custom_ssh_port = '$SLURM_JOB_ID')
    resource_alloc_str = cluster.resource_alloc(gpus_per_node = gpus_per_node, gpu_model = gpu_model, cpus_per_node = cpus_per_node, nodes = nodes)
    
    command = f"{shell_env} {command}" if shell_env else command
    
    if interactive:
        command = "script -qec \"tmux new-session -s '$SLURM_JOB_ID'\" /dev/null"
    else:
        # Use $SLURM_JOB_ID to create the actual output file path at runtime
        actual_stdout_file = f"{stdout_path}/$SLURM_JOB_ID.out"
        command = f"script -qec \"tmux new-session -s '$SLURM_JOB_ID' 'uv run {command} 2>&1 | tee {actual_stdout_file}'\" /dev/null"
    
    sbatch_command = f"""#!/bin/bash
#SBATCH -A {account}
#SBATCH -t {time_alloc}
{resource_alloc_str}
{stdout_str}
{ssh_setup_str}
{command}
"""
    return sbatch_command


def get_default_slurm_acc():
    import os
    with os.popen("sacctmgr show association where user=$USER format=Account --noheader --parsable | cut -d'|' -f1") as f:
        return f.read().split("\n")[0]


def get_cluster():
    with os.popen("sacctmgr show association where user=$USER format=Cluster --noheader") as f:
        clustr_str = trim_whitespace(f.read().split("\n")[0])
    if clustr_str == "alvis":
        return Alvis()
    elif clustr_str == "berzelius":
        return Berzelius()
    else:
        raise ValueError(f"Cluster {clustr_str} not supported")

def get_job_nodes(job_id):
    """Get the nodes allocated to a SLURM job."""
    max_attempts = 30  # Wait up to 5 minutes for job to start
    attempt = 0
    
    while attempt < max_attempts:
        # Try to get node information from squeue
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%N"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            nodes = result.stdout.strip()
            if nodes and nodes != "(null)" and "(" not in nodes:  # Job has started
                return nodes
        
        # If not found in squeue, try scontrol
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            # Look for NodeList in the output
            for line in result.stdout.split('\n'):
                if 'NodeList=' in line:
                    node_match = re.search(r'NodeList=(\S+)', line)
                    if node_match:
                        nodes = node_match.group(1)
                        if nodes and nodes != "(null)" and nodes != "N/A":
                            return nodes
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting...", flush=True)
            return None
        attempt += 1
        print(f"Waiting for job {job_id} to be allocated nodes... (attempt {attempt}/{max_attempts})", flush=True)
    
    return None

def print_ssh_info(job_id, nodes, no_ssh, cluster):
    """Print SSH connection information for the job."""
    if no_ssh:
        print(f"Job {job_id} submitted successfully!")
        if nodes:
            print(f"Allocated nodes: {nodes}")
        print(f"Note: tmux session 'slurm_job_{job_id}' will be available once job starts")
        return
    
    if nodes:
        # extract first node if multiple
        first_node = nodes.split(',')[0].split('[')[0]  # Handle node ranges
        
        # Calculate SSH port using same formula as in ssh_setup
        ssh_port = cluster.get_ssh_port(job_id)
        
        ssh_info = trim_whitespace(f"""
            SSH Connection Information with tmux:
            Job ID: {job_id}
            Node(s): {nodes}
            SSH Port: {ssh_port}
            tmux session: {job_id}

            To connect and monitor real-time output:
            ssh -t -p {ssh_port} $USER@{first_node} tmux attach-session -t {job_id}
                        
            To detach from tmux (leave job running): Ctrl-b d
            To list tmux sessions: tmux list-sessions
            
            You can check job status with: squeue -j {job_id}
            
            Note: SSH daemon files are stored in /tmp/slurm_ssh_{job_id} on the compute node
            """)

        
        print(format_in_box(ssh_info.strip()))
    else:
        job_info = f"Job {job_id} submitted, but node information not yet available."
        status_info = f"Check job status with: squeue -j {job_id}"
        tmux_info = f"tmux session 'slurm_job_{job_id}' will be available once job starts on a node"
        print(f"{job_info}\n{status_info}\n{tmux_info}")

def wait_for_job(job_id):
    """Wait for a SLURM job to complete."""
    print(f"Waiting for job {job_id} to complete...", flush = True)
    
    while True:
        # Check if job is still in queue
        status_result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%T"],
            capture_output=True, text=True
        )
        
        if status_result.returncode != 0 or not status_result.stdout.strip():
            # Job is no longer in queue, check final status
            sacct_result = subprocess.run(
                ["sacct", "-j", job_id, "--noheader", "--format=State"],
                capture_output=True, text=True
            )
            
            if "COMPLETED" in sacct_result.stdout:
                print(f"Job {job_id} completed successfully", flush = True)
                return True
            elif "FAILED" in sacct_result.stdout or "CANCELLED" in sacct_result.stdout:
                print(f"Job {job_id} failed or was cancelled", flush = True)
                return False
            else:
                print(f"Job {job_id} finished", flush = True)
                return True
        
        time.sleep(10)  # Wait 10 seconds before checking again

@dataclass
class Config:
    """Configuration for slurm job submission"""
    no_ssh: bool = False
    """Do not setup ssh server on berzelius"""
    dry_run: bool = False
    """Whether to submit the job or not"""
    blocking: bool = False
    """Whether to block until job completes before returning"""
    g: int = 1
    """Number of gpus per node"""
    m: str = "A100"
    """GPU model"""
    a: str = get_default_slurm_acc()
    """SLURM account number to use"""
    time: str = "0-00:30:00"
    """Time allocation in SLURM format"""
    shell_env: str = ""
    """Shell environment"""
    cpus_per_node: int = 16
    """Number of cpu cores per node"""
    num_tasks: int = 1
    """Number of tasks to run"""
    nodes: int = 1
    """Number of nodes to use"""
    interactive: bool = False
    """Whether to run in interactive mode"""
    stdout_path: str = os.path.expanduser("~/.cache/slurm")
    """Path to stdout folder"""
    tmux: bool = False
    """Whether to run in tmux session"""
    command: str = ""
    """Command to run"""

def main():
    config = tyro.cli(Config)
    # default_stdout = os.path.expanduser("~/.cache/slurm")
    # default_nodes = 1
    # default_cpus_per_node = 16
    # default_time = "0-00:30:00"
    # parser = argparse.ArgumentParser(description="Run experiment using SLURM")
    # parser.add_argument("--no_ssh", required=False, action="store_true", help="Do not setup ssh server on berzelius")
    # parser.add_argument("--dry_run", help="Whether to submit the job or not", action="store_true")
    # parser.add_argument("--blocking", help="Block until job completes before returning", action="store_true")
    # parser.add_argument("--gpus_per_node", "-g", required=False, help="Num gpus per node. (default: 1)", default=1, type = int)
    # parser.add_argument("--gpu_model", "-m", required=False, help="GPU model. (default: A100)", default="A100")
    # parser.add_argument("--account", "-a", required=False, help="SLURM account number to use", default=get_default_slurm_acc())
    # parser.add_argument(
    #     "--time",
    #     "-t",
    #     default=default_time,
    #     help=f"Time allocation in SLURM format (default: {default_time})",
    # )
    # parser.add_argument(
    #     "--shell_env",
    #     default="",
    #     help="shell env (default: )",
    # )
    # parser.add_argument(
    #     "--cpus_per_node",
    #     default=default_cpus_per_node,
    #     help=f"number of cpu cores per node (default: {default_cpus_per_node})",
    # )
    # parser.add_argument("--num_tasks", "-n", default=1, type=int, help="number of tasks to run (default: 1)")
    # parser.add_argument(
    #     "--nodes",
    #     "-N",
    #     default=default_nodes,
    #     type=int,
    #     help=f"number of nodes to use (default: {default_nodes})",
    # )
    # parser.add_argument(
    #     "--interactive",
    #     action="store_true",
    #     help="Runs a basic sleep command instead of the provided command",
    # )
    # parser.add_argument(
    #     "--stdout_path",
    #     default=default_stdout,
    #     required=False,
    #     type=str,
    #     help=f"Path to stdout folder (default: {default_stdout})",
    # )
    # parser.add_argument(
    #     "--tmux",
    #     action="store_true",
    #     help="Run command in a tmux session for real-time monitoring (requires SSH access to compute node)",
    # )
    # parser.add_argument(
    #     "command",
    #     nargs=argparse.REMAINDER,
    #     help="The command to run, along with its arguments.",
    # )

    # args = parser.parse_args()
    cluster = get_cluster()
    sbatch_command = wrap_in_sbatch(
        command=" ".join(config.command),
        account=config.a,
        gpus_per_node=config.g,
        cpus_per_node=config.cpus_per_node,
        nodes=config.nodes,
        no_ssh=config.no_ssh,
        time_alloc=config.time,
        num_tasks=config.num_tasks,
        shell_env=config.shell_env,
        interactive=config.interactive,
        stdout_path=config.stdout_path,
        gpu_model=config.m,
        cluster=cluster,
    )
    
    if not config.dry_run:
        print("Running the following sbatch script:")
        print(format_in_box(sbatch_command))        
        result = subprocess.run(["sbatch"], input=sbatch_command, text=True, capture_output=True)
        
        if result.returncode != 0:
            print(f"Failed to submit job: {result.stderr}")
            return 1
            
        # Extract job ID
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
        else:
            print(f"Could not extract job ID from: {result.stdout}")
            return 1
        
        print(result.stdout.strip())
        
        # Get node information and print SSH details
        nodes = get_job_nodes(job_id)
        print_ssh_info(job_id, nodes, config.no_ssh, cluster)
        
        if config.blocking:
            success = wait_for_job(job_id)
            return 0 if success else 1
        else:
            return 0
    else:
        print("If dry run was disabled, the following sbatch command would have been run:")
        print(format_in_box(sbatch_command))
        
    return 0

if __name__ == "__main__":
    sys.exit(main())