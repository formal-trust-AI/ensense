#!/usr/bin/python3
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
from rangedbooster import ExtendedBooster
from converttoopb import roundingSolve
from subprocess import check_output
from xyplot import Curve
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np

# import subprocess

from milp import main as milp_solver

# dummy milp_solver
# def milp_solver(args,options):
#     pass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from utils import open_model, open_model_xgb, open_model_sklearn, sigmoid_inv, model_details_file
from options import *
from pb import pb_solver


# pd.set_option("display.max_rows", 500)

def main(args,options):
    if options.solver == "milp":
         milp_solver(args,options)
    elif options.solver in [ "pb", "naive_smt", "rounding", "roundingsoplex"]:
        # --------------------------------------
        # Calling various pseudo boolean solvers 
        # --------------------------------------
        pb_solver(options)
    elif options.solver == "monitor":
        try:
            from monitor.monitor_lib import monitor
        except ModuleNotFoundError:
            raise SystemExit("monitor solver is not available")
        monitor(args,options)
    elif options.solver == "veritas":
        from solve_veritas import main as veritas_solver
        veritas_solver(args)
    else:
        print("Unrecognized solver: ", options.solver)



if __name__ == "__main__":

    # -----------------------------------------
    # GUI is launched if no options are passed
    # -----------------------------------------
    if len(sys.argv) == 1:
        ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ROOT))
        
        import gui
        gui.main()
        sys.exit(0)

        
    # --------------------------------
    # Argument to options
    # --------------------------------
    args,options = process_arguments()
    main(args,options)
    
