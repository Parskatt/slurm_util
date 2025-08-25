import subprocess

def test_submit():
    subprocess.run(["submit", "-g", "1", "-t", "0-00:01:00", "tests/test_job.py"], check=True)
    subprocess.run(["submit", "-g", "1", "-t", "0-00:01:00", "tests/test_job.py"], check=True)
    subprocess.run(["submit", "-g", "1", "-t", "0-00:01:00", "-i"], check=True)


if __name__ == "__main__":
    test_submit()