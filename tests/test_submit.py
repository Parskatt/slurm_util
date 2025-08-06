import subprocess

def test_submit():
    subprocess.run(["submit", "-g", ":1", "--dry_run"], check=True)
    subprocess.run(["submit", "-g", ":1", "--dry_run", "--interactive"], check=True)


if __name__ == "__main__":
    test_submit()