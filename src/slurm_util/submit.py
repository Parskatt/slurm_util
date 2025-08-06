import argparse
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
    def resource_alloc(self, *, gpus_per_node, cpus_per_node, nodes) -> str:
        gpu, num = gpus_per_node.split(":")
        gpu_alloc = f"--gpus-per-node {gpus_per_node}" if gpu != "NOGPU" else "-C NOGPU"
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

class Berzelius(Cluster):
    """https://www.nsc.liu.se/support/systems/berzelius-gpu/#21-resource-allocation-guidelines"""
    def resource_alloc(self, *, gpus_per_node, cpus_per_node, nodes) -> str:
        gpu, num = gpus_per_node.split(":")
        gpu_alloc = f"#SBATCH --gpus-per-node {num}" if gpu != "NOGPU" else "--partition=berzelius-cpu"
        if "80GB" in gpu:
            gpu_alloc += "\n#SBATCH -C fat"
        elif "40GB" in gpu:
            gpu_alloc += "\n#SBATCH -C thin"
        elif "1g.10gb" in gpu:
            gpu_alloc += "\n#SBATCH --reservation=1g.10gb"
        cpu_alloc = f"#SBATCH --cpus-per-gpu {cpus_per_node // int(num)}" if gpu != "NOGPU" else ""
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
        ssh_host_key_path = os.path.expanduser("~/.ssh/custom_sshd/ssh_host_key")
        custom_sshd_dir = os.path.expanduser("~/.ssh/custom_sshd")
        if not os.path.exists(ssh_host_key_path):
            os.makedirs(custom_sshd_dir, exist_ok=True)
            print("Setting up ssh server setup on berzelius...")
            print("NOTE: make sure that your ssh key is in ~/.ssh/authorized_keys")
            print(f"NOTE: Make sure to add {custom_ssh_port=} to your ~/.ssh/config")
            os.popen(trim_whitespace(f"""cat > sshd_config << 'EOF'
            Port {custom_ssh_port}
            PidFile ~/.ssh/custom_sshd/sshd.pid
            HostKey ~/.ssh/custom_sshd/ssh_host_key
            AuthorizedKeysFile ~/.ssh/authorized_keys
            PasswordAuthentication no
            PubkeyAuthentication yes
            ChallengeResponseAuthentication no
            Subsystem sftp internal-sftp
            EOF
            ssh-keygen -t rsa -f ~/.ssh/custom_sshd/ssh_host_key -N ''
            """))
        return "/usr/sbin/sshd -f ~/.ssh/custom_sshd/sshd_config -D &"



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
    cpus_per_node,
    no_ssh,
    nodes,
    time_alloc,
    num_tasks,
    max_parallel,
    custom_ssh_port,
    shell_env,
    interactive,
    stdout_path,
):
    cluster = get_cluster()
    stdout_file = stdout_path + "/%A.out"
    os.makedirs(stdout_path, exist_ok=True)
    stdout_str = f"#SBATCH -o {stdout_file}"
    ssh_setup_str = cluster.ssh_setup(no_ssh = no_ssh, custom_ssh_port = custom_ssh_port)
    resource_alloc_str = cluster.resource_alloc(gpus_per_node = gpus_per_node, cpus_per_node = cpus_per_node, nodes = nodes)
    
    if interactive:
        command = "script -qec \"tmux new-session -s '$SLURM_JOB_ID'\" /dev/null"
    else:
        command = f"script -qec \"tmux new-session -d -s '$SLURM_JOB_ID' '{command} 2>&1 | tee {stdout_file}' && tmux wait $SLURM_JOB_ID\" /dev/null"
    
    shell_env_wrapped_command = f"{shell_env} {command}" if shell_env else command
    sbatch_command = f"""#!/bin/bash
#SBATCH -A {account}
#SBATCH -t {time_alloc}
{resource_alloc_str}
{stdout_str}
{ssh_setup_str}
{shell_env_wrapped_command}
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

def print_ssh_info(job_id, nodes, no_ssh, custom_ssh_port):
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
        
        ssh_info = trim_whitespace(f"""
            SSH Connection Information with tmux:
            Job ID: {job_id}
            Node(s): {nodes}
            tmux session: {job_id}

            To connect and monitor real-time output:
            1. SSH to compute node: ssh -p {custom_ssh_port} $USER@{first_node}
            2. Attach to tmux session: tmux attach-session -t {job_id}
            
            To detach from tmux (leave job running): Ctrl-b d
            To list tmux sessions: tmux list-sessions
            
            You can check job status with: squeue -j {job_id}
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


def main():
    default_stdout = os.path.expanduser("~/.cache/slurm")
    default_nodes = 1
    default_max_parallel = 4
    default_cpus_per_node = 16
    default_time = "0-00:30:00"
    parser = argparse.ArgumentParser(description="Run experiment using SLURM")
    parser.add_argument("--no_ssh", required=False, action="store_true", help="Do not setup ssh server on berzelius")
    parser.add_argument("--dry_run", help="Whether to submit the job or not", action="store_true")
    parser.add_argument("--blocking", help="Block until job completes before returning", action="store_true")
    parser.add_argument("--gpus_per_node", "-g", required=False, help="Num gpus per node. (default: NOGPU:0)", default="NOGPU:0")
    parser.add_argument("--account", "-a", required=False, help="SLURM account number to use", default=get_default_slurm_acc())
    parser.add_argument("--custom_ssh_port", required=False, help="Port to use for custom ssh server on berzelius", default=2222)
    parser.add_argument(
        "--time",
        "-t",
        default=default_time,
        help=f"Time allocation in SLURM format (default: {default_time})",
    )
    parser.add_argument(
        "--shell_env",
        default="",
        help="shell env (default: )",
    )
    parser.add_argument(
        "--cpus_per_node",
        default=default_cpus_per_node,
        help=f"number of cpu cores per node (default: {default_cpus_per_node})",
    )
    parser.add_argument("--num_tasks", "-n", default=1, type=int, help="number of tasks to run (default: 1)")
    parser.add_argument(
        "--max_parallel",
        "-m",
        default=default_max_parallel,
        type=int,
        help=f"max number of tasks to run in parallel (default: {default_max_parallel})",
    )
    parser.add_argument(
        "--nodes",
        "-N",
        default=default_nodes,
        type=int,
        help=f"number of nodes to use (default: {default_nodes})",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Runs a basic sleep command instead of the provided command",
    )
    parser.add_argument(
        "--stdout_path",
        default=default_stdout,
        required=False,
        type=str,
        help=f"Path to stdout folder (default: {default_stdout})",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Run command in a tmux session for real-time monitoring (requires SSH access to compute node)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run, along with its arguments.",
    )

    args = parser.parse_args()
    sbatch_command = wrap_in_sbatch(
        command=" ".join(args.command),
        account=args.account,
        gpus_per_node=args.gpus_per_node,
        cpus_per_node=args.cpus_per_node,
        nodes=args.nodes,
        no_ssh=args.no_ssh,
        custom_ssh_port=args.custom_ssh_port,
        time_alloc=args.time,
        num_tasks=args.num_tasks,
        max_parallel=args.max_parallel,
        shell_env=args.shell_env,
        interactive=args.interactive,
        stdout_path=args.stdout_path,
    )
    
    if not args.dry_run:
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
        print_ssh_info(job_id, nodes, args.no_ssh, args.custom_ssh_port)
        
        if args.blocking:
            success = wait_for_job(job_id)
            return 0 if success else 1
        else:
            return 0
    else:
        print("If dryrun was disabled, the following sbatch command would have been run:")
        print(format_in_box(sbatch_command))
        
    return 0


if __name__ == "__main__":
    sys.exit(main())