#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
import itertools
import subprocess
import json
import re

modelTrees = { 
              'breast_cancer' : '0004',
                'covtype'       : '0080',
                'diabetes'      : '0020',
                'fashion'       : '0200',
                'ijcnn'         : '0060',
                'ori_mnist'     : '0200',
                'webspam'       : '0100',
}

modelFeature = {
    'adult' : 15 ,
    'churn' : 21 ,
    'pimadiabetes': 9,
    'winequality_red':11,
    'iris':4,
    'german_credit':20
}

multimodel = ['covtype','fashion','ori_mnist','iris','winequality_red']


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
BENCHMARKPATH = ROOT_DIR / 'models'

benchmarks = [ 
    ("breast_cancer","robust"),
    ("breast_cancer","unrobust"),
    # ("diabetes","robust"),
    ("diabetes","unrobust"),
    # ("ijcnn","robust"),
    ("ijcnn","unrobust"),
    # ("adult",'t200_d5'),
    ("adult",'t300_d5'),
    # ("adult",'t500_d5'),
    # ("adult",'t200_d6'),
    # ("adult",'t300_d6'),
    # ("adult",'t500_d6'),
    # ("adult",'t200_d5'),
    # ("churn",'t300_d5'),
    ("churn",'t500_d5'),
    # ("churn",'t200_d6'),
    # ("churn",'t300_d6'),
    # ("churn",'t500_d6'),
    # ("pimadiabetes",'t200_d5'),
    # ("pimadiabetes",'t300_d5'),
    ("pimadiabetes",'t500_d5'),
    # ("pimadiabetes",'t200_d6'),
    # ("pimadiabetes",'t300_d6'),
    # ("pimadiabetes",'t500_d6'),
    ("german_credit",'t500_d5'),
    # ("german_credit",'t800_d5'),
    # ("german_credit",'t500_d6'),
    # ("german_credit",'t800_d6'),
    # ("covtype","robust"),
    # ("covtype","unrobust"),
    # ("fashion","robust"),
    # ("fashion","unrobust"),
    # ("ori_mnist","robust"),
    ("ori_mnist","unrobust"),
    ("iris","t100_d5"),
    ("winequality_red","t100_d5")
    ]



def model_files(modelname, modeltype):
    trees = None
    if modelname in modelTrees:
        trees = modelTrees[modelname]
    no_feature = 0
    featlist = []
    if modelname in modelTrees:
        if modelname in multimodel:
            with open(f'{BENCHMARKPATH}/tree_verification_models/{modelname}_{modeltype}/feat_imp.json', 'r') as f:
                feat_file = json.load(f)
            featlist = [ t[1:] for t in list(feat_file.keys())[:100]]
        else:
            details_path = BENCHMARKPATH / f"dataset/{modelname}/{modelname}_details.csv"
            no_feature = pd.read_csv(details_path, index_col=0).shape[0]
            featlist = [f for f in range(no_feature)]
    elif modelname in modelFeature.keys():
            no_feature = modelFeature[modelname]
            featlist = [f for f in range(no_feature)]
    else:
        print(f"feat info missing")
        exit()

    if modelname in modelTrees:
        model_file = f"models/tree_verification_models/{modelname}_{modeltype}/{trees}.resaved.json"
    else:
        model_file = f"models/{modelname}/{modelname}_{modeltype}.json"
    
    detail_file = f"models/dataset/{modelname}/{modelname}_details.csv"
    if modelname in modelFeature:
        detail_file = f"models/dataset/{modelname}/details.csv"
    
    data_file = f"models/dataset/{modelname}/{modelname}_train.csv"
    if modelname in modelFeature:
        data_file = f"models/dataset/{modelname}/train.csv"

    clause_file = f"outputs/output/learned-clauses_{modelname}_{modeltype}.txt"
    if modelname in modelFeature:
        clause_file = f"outputs/output/learned-clauses_{modelname}_{modeltype}.txt"
    multi = False
    
    if modelname in multimodel: multi = True
    return model_file, featlist, detail_file, data_file, clause_file, multi


def simple_test(model_file, featlist, detail_file, data_file, clause_file,multi=False):
    feature_choices = [str(f) for f in featlist]
    gap_choices = [("0.4", "0.6"), ("0.3", "0.7")]
    solver_choices = ["pb", "milp"]

    commands = []
    for feature, (lgap, ugap), solver in itertools.product(feature_choices, gap_choices, solver_choices):
        base = (
            f"python ./src/sensitive.py {model_file} "
            f"--features {feature} --output_gap {lgap} {ugap} "
        )
        if solver == "pb" and not multi:
            commands.append(base + f" --solver pb --details {detail_file}")

        if solver == "milp" and not multi:
            commands.append(base + f" --details {detail_file}")
            commands.append(base + f" --details {detail_file} --all_opt")
            commands.append(base + f" --details {detail_file} --all_opt --prob --data_file {data_file}")
            commands.append(base + f" --details {detail_file} --all_opt --compute_data_distance --data_file {data_file} --in_distro_clauses {clause_file}")
            commands.append(base + f" --details {detail_file} --all_opt --prob --compute_data_distance --data_file {data_file} --in_distro_clauses {clause_file}")
        if solver == 'milp' and multi:
                commands.append(base + f" --multiclass --truelabel 1 --otherlabel 0")
                commands.append(base + f" --all_opt --multiclass --truelabel 1 --otherlabel 0")
    return commands

def arguments():
    parser = argparse.ArgumentParser(description='Run experiments with specified model and type')
    parser.add_argument('--modelname',default="",help='modelname')
    parser.add_argument('--modeltype',default="",help='mdeltype')
    parser.add_argument('--no_run', action='store_true', help='do not run generated commands')
    parser.add_argument('--limit', type=int, default=None, help='Run/print only first N commands')
    parser.add_argument('--show-output',action='store_true')
    args = parser.parse_args()
    modelname = args.modelname
    modeltype = args.modeltype

    return modelname, modeltype, args.no_run, args.limit, args.show_output


def parse_output(output):
    dist = re.search(r"Distance from data distype L2:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
    distance = float(dist.group(1)) if dist else None
    time = re.search(r"# Time:\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
    timetaken = float(time.group(1)) if time else None
    return distance, timetaken

def runoutput(rc,stdout,stderr):
    if rc != 0:
        return { "modelname": modelname,
                "modeltype": modeltype,
                "options": c,
                "distance": None,
                "Time": None,
                "output":stderr,
        }
    else:
        distance, timetaken = parse_output(stdout)
        return { "modelname": modelname,
                "modeltype": modeltype,
                "options": c,
                "distance": distance,
                "Time": timetaken,
                "output":stdout
        }


if __name__ == "__main__":
    results = []
    modelname, modeltype, no_run, limit, show_output = arguments()
    if modelname == "" and modeltype == "":
        print(f"Runing full benchmark")
    elif modelname != "" and modeltype != "":
        benchmarks = [(modelname,modeltype)]
    else:
        print(f"Error: missing modelname or modeltype")
        exit()
    for (modelname,modeltype) in benchmarks:
        # print(modelname,modeltype)
        model_file, featlist, detail_file, data_file, clause_file, multi = model_files(modelname, modeltype)
        commands = simple_test(model_file, featlist, detail_file, data_file, clause_file, multi)
        if limit is not None:
            commands = commands[:limit]
        passed = 0
        failed = 0
        if no_run:
            for c in commands:
                print(c)
        else:
            for c in commands:
                if show_output: print(f"[RUN]: {c}")
                res = subprocess.run(c, shell=True, cwd=str(ROOT_DIR),capture_output=True,text = True)
                rc = res.returncode
                stdout = res.stdout
                stderr = res.stderr
                if rc != 0:
                    if show_output: print(stderr)
                    failed += 1
                    row = runoutput(rc,stdout,stderr)
                else:
                    if show_output: print(stdout)
                    passed += 1
                    row = runoutput(rc,stdout,stderr)
                results.append(row)
        print(f"{modelname}_{modeltype}: PASSED:{passed} FAILED: {failed}")
    df = pd.DataFrame(results)
    outputfile = CURRENT_DIR / 'iclr2026results.csv'
    df.to_csv(outputfile)
    print(f'file saved to {outputfile}')
