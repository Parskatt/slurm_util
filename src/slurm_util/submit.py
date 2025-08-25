from slurm_util.utils import DeviceType
import argparse
import subprocess
import sys
# import time
# import re
import os
from slurm_util.utils import (
    format_in_box,
    get_default_slurm_acc,
    get_cluster,
)


def wrap_in_sbatch(
    *,
    command,
    account,
    gpus_per_node,
    device_type,
    cpus_per_gpu,
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
    ssh_setup_str = cluster.ssh_setup(no_ssh=no_ssh, custom_ssh_port="$SLURM_JOB_ID")
    resource_alloc_str = cluster.resource_alloc(
        gpus_per_node=gpus_per_node,
        device_type=device_type,
        cpus_per_gpu=cpus_per_gpu,
        nodes=nodes,
    )

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

def validate_args(args):
    if not args.command and not args.interactive:
        raise ValueError("Command is required when not running interactively.")

def main():
    cluster = get_cluster()
    default_stdout = os.path.expanduser("~/.cache/slurm")
    default_nodes = 1
    default_cpus_per_gpu = 16
    default_time = "0-00:30:00"
    parser = argparse.ArgumentParser(description="Run experiment using SLURM")
    parser.add_argument(
        "--no_ssh",
        required=False,
        action="store_true",
        help="Do not setup ssh server on berzelius",
    )
    parser.add_argument(
        "--dry_run", help="Whether to submit the job or not", action="store_true"
    )
    parser.add_argument(
        "--blocking",
        help="Block until job completes before returning",
        action="store_true",
    )
    parser.add_argument(
        "--gpus_per_node",
        "-g",
        required=False,
        help="Num gpus per node. (default: 1)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--device_type",
        "-d",
        required=False,
        help=f"Device type. (default: {cluster.DefaultDeviceType})",
        choices=cluster.DeviceType.__args__,
        default=cluster.DefaultDeviceType,
    )
    parser.add_argument(
        "--account",
        "-a",
        required=False,
        help="SLURM account number to use",
        default=get_default_slurm_acc(),
    )
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
        "--cpus_per_gpu",
        default=default_cpus_per_gpu,
        help=f"number of cpu cores per gpu (default: {default_cpus_per_gpu})",
    )
    parser.add_argument(
        "--num_tasks",
        "-n",
        default=1,
        type=int,
        help="number of tasks to run (default: 1)",
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
        "-i",
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
        cpus_per_gpu=args.cpus_per_gpu,
        nodes=args.nodes,
        no_ssh=args.no_ssh,
        time_alloc=args.time,
        num_tasks=args.num_tasks,
        shell_env=args.shell_env,
        interactive=args.interactive,
        stdout_path=args.stdout_path,
        device_type=args.device_type,
        cluster=cluster,
    )

    if not args.dry_run:
        print("Running the following sbatch script:")
        print(format_in_box(sbatch_command))
        result = subprocess.run(
            ["sbatch"], input=sbatch_command, text=True, capture_output=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Failed to submit job: {result.stderr}")
            return 1
    else:
        print(
            "If dry run was disabled, the following sbatch command would have been run:"
        )
        print(format_in_box(sbatch_command))

    return 0


if __name__ == "__main__":
    sys.exit(main())
