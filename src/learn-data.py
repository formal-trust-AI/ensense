#!/usr/bin/python3

import re
import os
import shutil
import sys
import subprocess
import pandas as pd
# import z3
from z3 import *
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
from itertools import combinations
import random
from options import *
import argparse
import numpy as np
import json
# import itertools

import utils
# from pyfinite import ffield
pd.set_option('display.max_rows', 80000)


def rev(s):
    return "".join(reversed(s))

sys.path.append(os.path.join(os.path.dirname(__file__), "computed"))

def clean_up_data( df ):
    observed_map = {}
    observed_map['SAL_BIN'] = { 'NO SALARY'  : 0,
                                'SAL_15_25K' : 1,
                                'SAL_25_35K' : 2,
                                'SAL_35_45K' : 3,
                                'BAL_45K_55K': 4,
                                'BAL_55K_75K': 5,
                                'BAL_75K_1L' : 6,
                                'BAL_1L_2L'  : 7,
                               }

    observed_map['FEATURE_35_BIN_NEW'] =  { 'AVG_BAL_LESS_THAN_0' : 0,
                                            'BAL_0_5K'            : 1,
                                            'BAL_5_15K'           : 2,
                                            'BAL_15K_30K'         : 3,
                                            'BAL_30K_50K'         : 4,
                                            'BAL_50K_1L'          : 5,
                                            'GRT_1L'              : 6 }

    observed_map['CRIF_RISK'] = {'NO CREDIT HISTORY': 0,
                                 'LOW RISK'         : 1,
                                 'MEDIUM RISK'      : 2,
                                 'VERY HIGH RISK'   : 3} #[1.0, 2.0, 3.0]
    for f in observed_map:
        df[f] = df[f].map(observed_map[f])
    return df


def process_data( options ):
    model, trees, n_trees, n_features, n_classes, base_val, feature_names, op_range_list = utils.open_model(
        options.model_file, details_file=options.details_file
    )
    
    feature_list = []
    features_range = {}
    for j in list(range(n_features)):
        name = feature_names[j]
        features_range[name] = ( list(set(trees[trees['Feature'] == f'f{j}']['Split'])) )
        features_range[name].sort()
        if len(features_range[name]) > 0:
            feature_list.append(name)
            num_guard = len(features_range[name])
            stride = num_guard // options.max_splits
            if stride > 1:
                new_guards = []
                for i in range(0,num_guard,stride):
                    new_guards.append(features_range[name][i])
                features_range[name] = new_guards

    # print(f"loading data from {options.data_file} with limit {options.data_limit}")
    # df = pd.read_csv(options.data_file, nrows=options.data_limit)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_df = pd.read_csv(options.data_file)   
    print(f"Data loaded from {options.data_file} with {len(full_df)} rows") 
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)    
    df = full_df.head(options.data_limit)
    
    # df = clean_up_data(df)
    # df.to_csv(data_file+'-cleaned.csv')
    
    features_in_both = []
    observed_ranges = {}
    # print(f"col: {df.columns}")
    for index, f in enumerate(feature_list):
        if not f in df.columns:  continue
        observed_ranges[f] = set(df[f])
        # -------------------------------------
        # Drop features that have no data range
        # -------------------------------------
        if len(observed_ranges[f]) > 0:
            features_in_both.append(f)
    feature_list = features_in_both
    df = df[feature_list]

    #-------------------------------------
    # Update data according to the guards
    #-------------------------------------
    # print('Updating data  (optimized)')
    for f in feature_list:
        thresholds = features_range[f]
        thresholds = np.array(thresholds)
        df[f] = np.searchsorted(thresholds, df[f].values)
    df = df.astype(int)
    # for index, row in df.iterrows():
    #     for pos,f in enumerate(feature_list):
    #         val = 0
    #         for ival, v in enumerate(features_range[f]):
    #             if row[f] < v: break
    #             val = ival+1
    #         df.at[ index, f ] = val
    # df = df.astype(int)
    return df, features_range



class Clause:
    def __init__(self, size, samples, features_gurads, output_file=None):
        self.samples = samples
        self.size = size
        self.n_feature = len(samples.columns)
        self.guard_v = features_gurads
        self.range_v = []
        # print(f"featre's guards: {features_gurads}")
        # print(f"sample: {samples.columns}")
        # -------------------------------------
        for f in samples.columns:
            self.range_v.append( len(features_gurads[f]) + 1 )
        self.considered_samples = []
        self.is_pos = [Bool("is_pos_{}".format(i)) for i in range(size)]
        self.var = [[Bool("var_{}_{}".format(i,j)) for j in range(self.n_feature)] for i in range(size)]
        self.bnd = [[Bool("bnd_{}_{}".format(i,j)) for j in range(max(self.range_v))] for i in range(size)]  # can be optimized?
        if output_file:
            self.ofile = output_file
        else:
            self.ofile = sys.stdout

    #------------------------------
    # Can we we make it incremental
    #------------------------------
    def add_consistency_cons( self, s ):
        cons = []
        # -------------------------------------
        # each literal has one variable
        # -------------------------------------
        print(f"adding consistency constraints!")
        for i in range(self.size):
            bits = [(1,v) for v in self.var[i]]
            cons.append(z3.PbEq(bits, 1))
            
        # -------------------------------------
        # sorted clause
        # -------------------------------------
        # If literal i selects feature v, then no earlier literal (ip) can select any smaller feature (vp).
        for i in range(self.size):
            for ip in range(i):
                for v in range(self.n_feature):
                    for vp in range(v):
                        cons.append(Implies(self.var[i][v], Not(self.var[ip][vp]))) 
                                
        # -------------------------------------
        # constraints for distinct variables in the clause
        # -------------------------------------
        for v in range(self.n_feature):
            bits = [(1,self.var[i][v]) for i in range(self.size)]
            cons.append(z3.PbLe(bits, 1))

        # -------------------------------------
        # constraints for the guard consistancy
        # -------------------------------------
        for i in range(self.size):
            for j in range(len(self.bnd[i])-1):
                cons.append( Implies( self.bnd[i][j], self.bnd[i][j+1] ) )

        # -------------------------------------
        # Maximum guard always holds 
        # -------------------------------------
        for i in range(self.size):
            for v in range(self.n_feature):
                max_guard = self.range_v[v]-1
                cons.append( Implies( self.var[i][v], self.bnd[i][max_guard] ) )
                
        s.add(And(cons))

    def evaluate_solution(self,clause):
        # for index, row in self.samples.iterrows():
        #     sample = row.to_list()
        #     clause_val = False
        #     for i,(sign,var,g) in enumerate(clause):
        #         # print(sample[var],var,sign,g)
        #         clause_val = clause_val or (sign == (sample[var] < g)) 
        #     if clause_val == False:
        #         end_time = time.time()
        #         return index
        # time_taken = time.time() - start_time
        # print(f"Clause {clause} covers all samples, Time taken: {time_taken:.5f} seconds")
        
        # return None
        # optimized
        X = self.samples.values  # shape: (n_samples, n_features)
    
        satisfied = np.zeros(X.shape[0], dtype=bool)

        for (sign, var, g) in clause:
            condition = X[:, var] < g
            if not sign:
                condition = ~condition
            satisfied |= condition  

        unsatisfied_indices = np.where(~satisfied)[0]
        if len(unsatisfied_indices) > 0:
            return int(unsatisfied_indices[0]), len(unsatisfied_indices)
        return None,0

    
    
    
    def stochastic_clause(self, clause, df):
        total_samples = len(df)
        covered_samples = df[df.apply(
            lambda row: any(
                sign == (row.iloc[var] < g)
                for (sign, var, g) in clause
            ), axis=1)
        ]
        
        coverage_prob = len(covered_samples) / total_samples
        return coverage_prob 
    
    def optimize_solution(self, clause):
        # Random order minimize the clause
        permutation = list(range(len(clause)))
        random.shuffle(permutation)
        cols = self.samples.columns
        # Optimization: Can I remove or simplify literal i without losing any samples?
        for i in permutation:
            (sign, var, g) = clause[i]
            fltr = self.samples.apply(lambda row: True, axis=1)
            for j in range(len(clause)):
                if i == j : continue
                if clause[j] == None: continue
                (sign1,var1,g1) = clause[j]
                cnd = ~(self.samples[ cols[var1] ] < g1)
                if not sign1: cnd = ~cnd
                fltr = fltr & cnd
            if sign:
                new_g = self.samples[fltr][ cols[var] ].max()+1
                assert( pd.isna(new_g) or new_g <= g)
            else:
                new_g = self.samples[fltr][ cols[var] ].min()
                assert( pd.isna(new_g) or new_g >= g)
            if pd.isna(new_g):
                clause[i] = None
            elif new_g != g:
                clause[i] = (sign, var, new_g) 
        clause = [lit for lit in clause if lit != None]
        return clause
            

    def encode_lit(self,pos,lit,cons):
        (sign,var,g) = lit
        
            
        sign_lit = self.is_pos[pos] if sign else Not(self.is_pos[pos])
        var_bit = self.var[pos][var]
        cons.append(sign_lit)
        cons.append(var_bit)
        if sign:
            if g > 1:
                var_bnd = self.bnd[pos][g-1]
                cons.append( Not(var_bnd) )
        else:
            if g < len(self.bnd[pos])-1:
                var_bnd = self.bnd[pos][g+1]
                cons.append(var_bnd)
        
    def encode_solution(self, clause):
        if len(clause) == self.size:
            cons = []
            for i,lit in enumerate(clause):
                self.encode_lit(i,lit,cons)
            return And(cons)
        else:
            lst     = list(range(self.size))
            subsets = list(combinations(lst, len(clause)))
            all_cons = []
            for subset in subsets:
                cons = []
                j = 0
                for i in range(self.size):
                    if i in subset:
                        self.encode_lit(i,clause[j],cons)
                        j += 1
                all_cons.append( And(cons) )
            return Or(all_cons)
            
        
    def reject_solution( self, solver, clause):
        cons = self.encode_solution(clause)
        solver.add( Not(cons) ) # Rejection clause
        
    def force_solution( self, solver, clause):
        cons = self.encode_solution(clause)
        solver.add( (cons) )
        
    def read_clause( self, m ):
        clause = []
        for i in range(self.size):
            is_p = z3.is_true(m[self.is_pos[i]])
            for v in range(self.n_feature):
                if z3.is_true(m[self.var[i][v]]):
                    break
            for idx,b in enumerate(self.bnd[i]):
                if idx == 0 : continue # zero bit does not contribute #! why?
                if self.range_v[v] <= idx: break
                if z3.is_true(m[b]): break
            clause.append( (is_p,v,idx) )
        return clause

    def print_clause( self, clause ):
        l = len(clause)-1
        for i,(is_p,v,g) in enumerate(clause):
            # if False:
            #     sample_vals = []
            #     for s in self.considered_samples:
            #         sample = self.samples.iloc[s].to_list()
            #         sample_vals.append( sample[v] )
            # else:
            #     sample_vals = ""
            end = " OR " if i != l else "\n"
            is_p = "" if is_p else "not"
            print(f"{is_p}({self.samples.columns[v]} < {g})", end=end)
        
    def dump_clause( self, clause ):
        d_clause = []
        for (is_p,v,g) in clause:
            if g > 0: # g == 0 imiplies that the iteral is always false, no need to save!
                vname = self.samples.columns[v]
                guard = self.guard_v[vname][g-1]
                d_clause.append((is_p,vname,guard))
        print(d_clause, file=self.ofile)
        
    def eval_literal(self, i, sample):
        cons = []
        for v in range(self.n_feature):
            guard = []
            sign = self.is_pos[i]
            # --------------------
            # for ordered features 
            #---------------------
            for r in range(1,self.range_v[v]): # range of values for v
                b = self.bnd[i][r]
                b = Implies(b,sign) if sample[v] < r else Implies( b,Not(sign) )
                guard.append(b)
            cons.append( Implies( self.var[i][v], And(guard) ))
        return And(cons)
    
    def cons_clause(self, sample_pos):
        sample = self.samples.iloc[sample_pos].to_list()
        # print(f"cons_clause: {sample_pos}, \n sample: {sample}")
        cons = [ self.eval_literal( i, sample) for i in range(self.size) ]
        # print(f"cons: {cons}")
        return Or(cons)

    def add_sample_cons( self, solver, sample):
        self.considered_samples.append(sample) # sample idx
        cons = self.cons_clause( sample )
        solver.add( cons )

# size = 1
# num_samples = 3000


def learn_clauses( options ):
    df,features_range = process_data( options )
    # with open(f'./models/tree_verification_models/ori_mnist_unrobust/feat_imp.json', 'r') as f:
    #     feat_imp = json.load(f)
    # feat_imp = list(feat_imp.keys())[:150]
    # feature_range = {f: features_range[f] for f in feat_imp if f in features_range}
    # label = [col for col in df.columns if col not in feature_range]
    # df = df[feat_imp+label]
    # print(df.columns)
    # input()
    coverage_threshold = options.coverage_threshold
    print(f"Data size: {len(df)}")
    # print(f"Features: {list(df.columns)}")
    ofile = open(options.output_file, "w")
    learned_clauses = []
    used_samples = list(range(1))
    # for size in range(1,options.max_learned_clause_size+1):
    stop = 0
    if options.max_clauses == 0:
        stop = -1
    
    for size in range(1,4):
        if stop == 1: break
        start_time = time.time()
        print(f'=== Solving for size {size} =====')
        # print(df.head())
        c = Clause( size, df, features_range, output_file=ofile)
        s = Solver()
        c.add_consistency_cons( s )
        for i in used_samples:
            c.add_sample_cons( s, i)
        for clause in learned_clauses: 
            c.reject_solution(s, clause)
        while True:
            if stop !=-1: 
                if len(learned_clauses) > options.max_clauses: 
                    stop=1
                    break
            
            if s.check() == sat:
                m = s.model()
                clause = c.read_clause(m)
                violating_sample, unsat = c.evaluate_solution(clause) # TODO: Only if there are too many violation
                if violating_sample:
                    c.add_sample_cons( s, violating_sample )
                    used_samples.append(violating_sample)

                    # coverage_prob = c.stochastic_clause(clause,df)
                    coverage_prob = (len(df)-unsat)/ len(df)
                    if coverage_prob > coverage_threshold:
                        c.reject_solution( s, clause) #negate the learned clause and add it to the solver
                        learned_clauses.append(clause)
                        print( f'=== Solving {len(learned_clauses)} =====' )
                        # c.print_clause( clause )
                        c.dump_clause( clause )
                else:
                    
                    # input()
                    # TODO: optimize the clause
                    # (f0 < g0) \/ (f1 < g1)
                    clause = c.optimize_solution(clause)
                    c.reject_solution( s, clause) #negate the learned clause and add it to the solver
                    learned_clauses.append(clause)
                    # print( f'=== Solving {len(learned_clauses)} =====' )
                    # c.print_clause( clause )
                    c.dump_clause( clause )
                
                if stop !=-1: 
                    if len(learned_clauses) > options.max_clauses: 
                        stop=1
                        break
                
            else:
                print(f'No solution! Considered samples: {len(used_samples)}')
                break
            
        # print(used_samples)
        end_time = time.time()
        print(f'Time taken for size {size}: {end_time - start_time:.2f} seconds')

if __name__ == "__main__":

    options = Options()


    parser = argparse.ArgumentParser(
        description="Find sensitivity on any single feature"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="File containing data")

    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="File containing model")

    parser.add_argument(
        "--details",
        type=str,
        default="",
        help="File containing extra details of the model. Sometimes models have missing details!")

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="File for dumping clauses")

    parser.add_argument(
        "--max_size",
        type=int,
        default=3,
        help="maximum size of a clause")

    parser.add_argument(
        "--max_splits",
        type=int,
        default=1000000,
        help="maximum number of splits per feature")
    
    parser.add_argument(
        "--data_limit",
        type=int,
        default=10000,
        help="maximum size of the data")
    
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=1,
        help="coverage threshold for a clause to be accepted")
    
    parser.add_argument(
        "--max_clauses",
        type=int,
        default=0,
        help="maximum number of clauses to learn before stopping"
    )

    args = parser.parse_args()
    options.data_file = args.data
    options.model_file = args.model
    options.output_file = f'{args.output}.txt'
    options.details_file = args.details
    options.max_learned_clause_size = args.max_size
    options.data_limit = args.data_limit
    options.max_splits = args.max_splits
    options.coverage_threshold = args.coverage_threshold
    options.max_clauses = args.max_clauses
    starttime = time.time()
    learn_clauses( options )
    print(f"totaltimetaken is ({time.time() - starttime})")

    # options.data_file = "./models/sbi/sbi-fraud-sample-data-3000.csv"
    # options.model_file = './models/sbi/sbi-fraud.json'
    # options.output_file = './outputs/learned-clauses.txt'
