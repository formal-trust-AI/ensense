#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import sys
import pickle
# from src.utils import model_files
import json
import sys
import joblib
import xgboost as xgb
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from subprocess import check_output
from pb import pb_solver
import milp
from pathlib import Path
import glitch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))




# import subprocess

# from utils import open_model, open_model_xgb, open_model_sklearn, sigmoid_inv, model_details_file
from options import *


# pd.set_option("display.max_rows", 500)

def main(args,options):
    if options.spec == 'sens':
        if options.solver == "milp":
            milp.milp_solver(options)
        elif options.solver in [ "pb", "naive_smt", "rounding", "roundingsoplex"]:
            # --------------------------------------
            # Calling various pseudo boolean solvers 
            # --------------------------------------
            pb_solver(options)
        elif options.solver == "veritas":
                from solve_veritas import main as veritas_solver
                veritas_solver(args)
    elif options.spec == "monitor":
        try:
            from monitor.monitor_lib import monitor
        except ModuleNotFoundError:
            raise SystemExit("monitor solver is not available")
        monitor(args,options)
    
    elif options.spec == "glitch":
        glitch.glitch_solver(options)
    else:
        print("Unrecognized spec: ", options.spec)



if __name__ == "__main__":
        
    # --------------------------------
    # Argument to options
    # --------------------------------
    args,options = process_arguments()
    if args.warnings:
        warnings.resetwarnings()
    main(args,options)
    
