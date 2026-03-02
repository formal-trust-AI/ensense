"""
Run single_change.py for all trees followed by a run of rounding sat for that encoding
The times are stored in solving_times.txt
"""

from subprocess import check_output
import subprocess
import re
import random
import time

from pathlib import Path
Path('./outputs/').mkdir(parents=True, exist_ok=True)
with open('solving_times.txt', 'w') as f:
    f.write(f'i z3_time rounding_time\n')
for j in range(5):
    for i in range(13):
        featurenum = random.randint(0, 10)
        start = time.perf_counter()
        args = ['python', 'single_change.py', f'{i}', 'z3', '--max_trees' ,'30', '--features', f'{featurenum}']
        z3 = subprocess.Popen(args, stdout=subprocess.PIPE)
        ret_code = z3.wait()
        z3_time = time.perf_counter() - start
        args[3] = 'rounding'
        start = time.perf_counter()
        roundingsat = subprocess.Popen(args, stdout=subprocess.PIPE)
        ret_code = roundingsat.wait()
        rounding_time = time.perf_counter() - start
        with open('solving_times.txt', 'a') as f:
            f.write(f'{i} {z3_time} {rounding_time}\n')
