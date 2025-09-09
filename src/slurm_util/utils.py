from abc import ABC, abstractmethod
import subprocess
import time
import re
import os
from typing import Literal, Union

def trim_whitespace(s):
    return "\n".join([line.strip() for line in s.split("\n") if line.strip()])

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

class Cluster(ABC):
    name: str
    @abstractmethod
    def resource_alloc(self, *, gpus_per_node, cpus_per_gpu, nodes) -> str:
        pass

    @abstractmethod
    def ssh_setup(self, *, no_ssh, custom_ssh_port) -> str:
        pass




class Alvis(Cluster):
    name = "alvis"
    DeviceType = Literal["A100:40GB", "A100:80GB", "A40", "V100", "T4", "cpu"]
    DefaultDeviceType: DeviceType = "A100:40GB"
    """https://www.nsc.liu.se/support/systems/alvis/#21-resource-allocation-guidelines"""
    _map_to_alvis_naming: dict[DeviceType, str] = {
        "A100:40GB": "A100",
        "A100:80GB": "A100fat",
        "A40": "A40",
        "V100": "V100",
        "T4": "T4",
        "cpu": "",
    }
    def resource_alloc(self, *, gpus_per_node: int, device_type: DeviceType, cpus_per_gpu: int, nodes: int) -> str:
        device_type_alvis = self._map_to_alvis_naming[device_type]
        gpu_alloc = f"--gpus-per-node {device_type_alvis}:{gpus_per_node}" if device_type != "cpu" else "-C NOGPU"
        cpu_alloc = f"--cpus-per-task {cpus_per_gpu*gpus_per_node}"  # one task per node, CPUs = GPUs * CPUs per GPU
        task_alloc = "--ntasks-per-node 1"
        node_alloc = f"--nodes {nodes}"
        return trim_whitespace(f"""
            #SBATCH {gpu_alloc}
            #SBATCH {cpu_alloc}
            #SBATCH {task_alloc}
            #SBATCH {node_alloc}
                """)

    def ssh_setup(self, *, no_ssh, custom_ssh_port) -> str:
        # alvis supports ssh by default
        return ""

    def get_ssh_port(self, _):
        return 22

class Berzelius(Cluster):
    name = "berzelius"
    DeviceType = Literal["A100","A100:80GB", "100:40GB", "A100:10GB", "cpu"]
    DefaultDeviceType: DeviceType = "A100"
    """https://www.nsc.liu.se/support/systems/berzelius-gpu/#21-resource-allocation-guidelines"""
    def resource_alloc(self, *, gpus_per_node: int, device_type: DeviceType, cpus_per_gpu: int, nodes: int) -> str:
        gpu_alloc = f"#SBATCH --gpus-per-node {gpus_per_node}" if device_type != "cpu" else "--partition=berzelius-cpu"    
        if device_type == "100:80GB":
            gpu_alloc += "\n#SBATCH -C fat"
        elif device_type == "100:40GB":
            gpu_alloc += "\n#SBATCH -C thin"
        elif device_type == "A100:10GB":
            gpu_alloc += "\n#SBATCH --reservation=1g.10gb"
        # Use one Slurm task per node when launching with torchrun under srun
        task_alloc = "#SBATCH --ntasks-per-node 1"
        cpu_alloc = f"#SBATCH --cpus-per-task {cpus_per_gpu*gpus_per_node}" if device_type != "cpu" else ""
        node_alloc = f"#SBATCH --nodes {nodes}"
        alloc_str = trim_whitespace(f"""
            {gpu_alloc}
            {task_alloc}
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


DeviceType = Union[Alvis.DeviceType, Berzelius.DeviceType]



def print_ssh_info(job_id, cluster):
    """Print SSH connection information for the job."""
    nodes = get_job_nodes(job_id)
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

