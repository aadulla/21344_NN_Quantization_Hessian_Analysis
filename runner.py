#!/usr/bin/env python3

import os
import subprocess

CPP_EXEC = "/code/build/src/21344"
EXP_DIR = "/code/exps"

for curr_dir in os.listdir(EXP_DIR):
    exp_dir_path = os.path.join(EXP_DIR, curr_dir)
    config_json_path = os.path.join(exp_dir_path, "config.json")
    results_json_path = os.path.join(exp_dir_path, "results.json")
    
    print("Running Experiment", curr_dir, "...")
    print("*"*50)
    print()
    subprocess.run([CPP_EXEC, config_json_path, results_json_path], stderr=subprocess.STDOUT)
    print("Finished Experiment", curr_dir, "!!!")
    print("*"*50)
    print("*"*50)
    print()
