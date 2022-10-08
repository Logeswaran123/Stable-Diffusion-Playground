# Setup for animate mode

import subprocess
import time


print("Setting up environment...")
start_time = time.time()

all_process = [
    ['git', 'clone', 'https://github.com/deforum/stable-diffusion'],
    ['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git'],
    ['git', 'clone', 'https://github.com/isl-org/MiDaS.git'],
    ['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git'],
    ['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'],
    ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
]

for process in all_process:
    running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')

print(subprocess.run(['git', 'clone', 'https://github.com/deforum/k-diffusion/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
with open('k-diffusion/k_diffusion/__init__.py', 'w') as f:
    f.write('')

end_time = time.time()
print(f"Environment set up in {end_time-start_time:.0f} seconds")