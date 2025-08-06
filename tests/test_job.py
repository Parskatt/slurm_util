import time
import sys


T = 100
print(f"Let's count to {T}...")
sys.stdout.flush()  # Force output to be displayed immediately
for t in range(T):
    print(f"Hello, world! {t}")
    sys.stdout.flush()
    time.sleep(1)
print(f"Job finished! We counted to {T}!")
sys.stdout.flush()