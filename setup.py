# Setup for torch, torchvision

import subprocess
import time


print("Setting up environment...")
start_time = time.time()

all_process = [
    ['pip', 'install', 'torch==1.12.1+cu116', 'torchvision==0.13.1+cu116', '--extra-index-url', 'https://download.pytorch.org/whl/cu116'],
]

for process in all_process:
    running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')

end_time = time.time()
print(f"Environment set up in {end_time-start_time:.0f} seconds")