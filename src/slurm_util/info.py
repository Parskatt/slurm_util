import argparse
# import subprocess
import sys
# import time
from slurm_util.utils import format_in_box, get_cluster, get_job_nodes, trim_whitespace


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

# def wait_for_job(job_id):
#     """Wait for a SLURM job to complete."""
    
#     while True:
#         # Check if job is still in queue
#         status_result = subprocess.run(
#             ["squeue", "-j", job_id, "--noheader", "--format=%T"],
#             capture_output=True, text=True
#         )
        
#         if status_result.returncode != 0 or not status_result.stdout.strip():
#             # Job is no longer in queue, check final status
#             sacct_result = subprocess.run(
#                 ["sacct", "-j", job_id, "--noheader", "--format=State"],
#                 capture_output=True, text=True
#             )
#             print(sacct_result.stdout)
#             if "COMPLETED" in sacct_result.stdout:
#                 print(f"Job {job_id} completed successfully", flush = True)
#                 return True
#             elif "FAILED" in sacct_result.stdout or "CANCELLED" in sacct_result.stdout:
#                 print(f"Job {job_id} failed or was cancelled", flush = True)
#                 return False
#             else:
#                 print(f"Job {job_id} either finished or was not started", flush = True)
#                 return True
#         print(f"Waiting for job {job_id} to start...", flush = True)
#         time.sleep(10)  # Wait 10 seconds before checking again

def main():
    parser = argparse.ArgumentParser(description="Wait for a SLURM job to complete")
    parser.add_argument("--job", "-j", type=str, help="SLURM job ID")
    args = parser.parse_args()
    cluster = get_cluster()
    print_ssh_info(args.job, cluster)
    return 

if __name__ == "__main__":
    sys.exit(main())