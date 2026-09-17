import argparse
import utils
import pandas as pd
import os
import time
class Options:
    def __init__(self):
        self.in_distro_clauses_file = ""
        self.data_file = ""
        self.model_file = ""
        self.modelName = ""
        self.modeltype = ""
        self.details_file = None
        self.output_file = ""
        self.output_csv_file = ""
        self.max_learned_clause_size = 3 # default value
        self.solver = "smt"
        self.data_limit = 10000
        self.max_splits = 100000
        self.gap = 0.2
        self.output_gap = None # [0.3,0.7]
        self.sureofcounter = False
        self.encoding = ""
        self.truelabel = -2
        self.otherlabel = -2
        self.multiclass = False
        self.verbosity = 0
        self.local_check_file = None
        self.local_check_samples = None
        self.prune        = False
        self.numeric = False
        self.categorical = False
        self.timeout = 3600
        self.max_clauses = None
        self.objective = False
        self.unaffected_cons = False
        self.affected_cons = False
        self.ancestor_cons = False   
        self.all_features = False   
        self.compute_data_distance = False
        self.plot = False
        self.plot_file = None
        self.features = [0]
        self.precision = 1000
        self.max_trees = None
        self.perturb = 0.1
        self.metric = None
        self.pca = False
        self.pca_d = 0
        self.limit_sens = -1
        # self.random = False
        self.random_samples = 0
        self.seed = 7
        self.n_jobs = 8
        self.gowal = False
        self.current_sample = None
        self.binary_perturb = 0
        self.log_stream = None
        self.log_folder = None
        self.no_remove = False
        self.anchor = False
        self.all_single = False
        self.alpha = 0
        self.spec = 'sens'
        
def arguments_to_options(args):
    options=Options()

    options.solver = args.solver
    options.spec = args.spec
    if options.spec=='glitch' and options.solver != 'milp':
        print("glitch only supports milp")
        exit(1)
    
    if args.solver == "naive_smt":
        options.encoding = "allsum"
    else:
        options.encoding = "pb"

    if args.sure_counterexamples: options.sureofcounter = True
    options.verbosity              = args.verbosity
    options.in_distro_clauses_file = args.in_distro_clauses
    # utils.file_check(options.in_distro_clauses_file)
    options.data_file              = args.data_file
    utils.file_check(options.data_file)
    options.model_library          = args.model_library 
    options.output_gap             = args.output_gap
    options.local_check_file       = args.local_check_file
    utils.file_check(options.local_check_file)
    options.timeout                = args.timeout
    options.max_trees              = args.max_trees
    
    options.objective       = args.objective
    options.unaffected_cons = args.unaffected_cons
    options.affected_cons   = args.affected_cons
    options.ancestor_cons   = args.ancestor_cons
    options.all_features    = args.all_features
    options.small_change    = args.small_change
    options.compute_data_distance = args.compute_data_distance
    options.plot         = args.plot
    options.plot_file    = args.plot_file
    if options.plot_file: options.plot = True
    options.all_single   = args.all_single
    options.limit_sens   = args.limit_sens
    options.strong_multi = args.strong_multi
    
    options.model_file   = args.filenum
    utils.file_check(options.model_file)
    options.details_file = args.details
    utils.file_check(options.details_file)

    options.log_file = args.log
    options.log_folder = args.log_folder
    options.no_remove = args.no_remove
    if options.log_file:
        open(options.log_file, 'w').close() 
        log_dir = options.log_folder + 'senstivity_' + time.strftime("%Y%m%d_%H%M%S")  
        os.mkdir(log_dir)
        options.log_folder = log_dir
        
        # utils.file_check( os.path.dirname(options.log_file) ) # todo:
        # options.log_stream = open(options.log_file, "w")
    # utils.open_log_file( options )
    # if options.log_file and not os.path.exists( Path(options.log_file).parent ):
    #     print( f"The path to the log file {options.details_file} is missing!" )
    
    if args.features != None:
        options.features = args.features
    elif args.solver != "glitch":
        options.features = [0]
        
    if args.features != None:
        options.features = args.features
    options.precision = args.precision
    options.alpha = args.alpha
    options.debug = args.debug
    options.prob = args.prob
    options.perturb = args.perturb

    options.metric = args.metric
    options.pca = args.pca
    options.pca_d = args.pca_d
    # options.random = args.random
    options.random_samples = args.random_samples
    options.seed = args.seed
    options.n_jobs = args.n_jobs
    options.gowal = args.gowal
    options.binary_perturb = args.binary_perturb
    options.numeric = args.numeric
    options.categorical = args.categorical
    options.prune        = args.prune
    if args.local_check_sample:
        options.local_check_samples    = [args.local_check_sample]
    options.anchor = args.anchor
    
    utils.local_senstivity_check(options.random_samples,
                        options.local_check_samples,
                        options.local_check_file
                        )
    
    if options.local_check_file:            
        local_samples = pd.read_csv( args.local_check_file )
        samples = []
        for index, row in local_samples.iterrows():
            samples.append( row.tolist() )
        options.local_check_samples = samples[:1000]
    
    if options.output_gap != None:
        options.lgap = options.output_gap[0]
        options.ugap = options.output_gap[1]
    else:
        options.lgap = 0.5-0.2 # args.gap
        options.ugap = 0.5+0.2 # args.gap
    utils.dump_info( options, 5, f"Effective gap {options.lgap} and {options.ugap}")

    if args.multiclass:
        options.truelabel = (args.truelabel)  # -1 indicates take or else we have a represantitive class
        options.otherlabel = args.otherlabel
        options.multiclass = True
    else:
        options.truelabel = -2  # Binary
        options.otherlabel = -2
        options.multiclass = False
    
    if options.prune:                                                                                                                                                                                            
        if options.in_distro_clauses_file:                                                                                                                                                                       
            utils.print_error("Options", "--prune is be combined with --in_distro_clauses")                                                                                                                  
        if options.gowal:                                                                                                                                                                                        
            utils.print_error("Options", "--prune is be combined with --gowal") 
   
    return options

def process_arguments():
    parser = argparse.ArgumentParser(
        description="Find sensitivity on any single feature"
    )
    parser.add_argument(
        "filenum",
        help="An integer file number. (Look in utils.py for list of files) or a filename",
    )
    parser.add_argument(
        "--model_library", help="0:xgboost  1:lgbm 2:sklearn", type=str, default="xgboost"
    )

    parser.add_argument(
        "--truelabel", help="Label of true class, required", type=int, default=-1
    )
    parser.add_argument(
        "--otherlabel", help="Label of other class, required", type=int, default=-1
    )

    # Add the 'solver' argument with choices
    parser.add_argument(
        "--solver",
        choices=["pb", "naive_smt", "rounding", "roundingsoplex", "milp"],
        help="The solver to use. Choose either 'smt' or 'rounding'.",
    )
    
    parser.add_argument(
            "--spec",
            choices=["sens","monitor","glitch"],
            default='sens',
            help=" which spec you want to check.sens stands for senstivity",
        )

    # Add the 'close' argument which is a boolean (true/false)
    parser.add_argument(
        "--close",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Close option, either 'true' or 'false'. Default is 'false'. (deprecated)",
    )
    parser.add_argument(
        "--max_trees",
        type=int,
        default=None,
        help="Maximum number of trees to consider",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=100,
        help="Maximum number of classes to consider (deprecated)",
    )
    # parser.add_argument(
    #     "--stop",
    #     action="store_true",
    #     help="whether to stop when the range of a node becomes less than a threshold",
    # )
    parser.add_argument(
        "--debug", action="store_true", help="Run serially and stop on pdb statements"
    )
    parser.add_argument(
        "--strong_multi", action="store_true", help="Strong multiclass checking"
    )
    parser.add_argument(
        "--no_strong_multi", action="store_true", help="Weak multiclass checking"
    )
    parser.add_argument(
        "--stop_param",
        type=float,
        default=0.1,
        help="Tunes how aggresssively we fold nodes",
    )
    parser.add_argument(
        "--all_single", action="store_true", help="run on all singular feature sets"
    )
    
    parser.add_argument(
            "--limit_sens",
            type=int,
            default=-1,
            help="put the limit on all_single, it will select the --limit_sens randomly from features",
        )
    
    parser.add_argument(
        "--prob", action="store_true", help="Activate probability objective"
    )
    
    parser.add_argument(
          "--prune",
          action="store_true",
          help="local search only: skip tree branches the perturbation box makes unreachable",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for each senstivity task",
    )
    
    parser.add_argument(
        "--lambda",
        type=int,
        default=100,
        help="Lambda for the objective function (deprecated)",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=-1,
        help="dataset index",
    )
    # parser.add_argument(
    #     "--random", action="store_true", help="to local senstivity on random data"
    # )
    parser.add_argument(
        "--random_samples",
        type=int,
        default=0,
        help="n0 of  random saamples for local senstivity",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="parrallel queries for local senstivity",
    )

    for (feature,help_text) in [
            ("all_features","Allow all features to change"),
            ("small_change","Only allow small change in the inputs"),
            ("ancestor_cons","Add additional ancestor constraints"),
            ("affected_cons","Enable affected constraints optimization"),
            ("unaffected_cons","Enable unaffected constraints optimization"),
            ("precise","Compute with any approximation on leaf values"),
            ("multiclass","Enable support for multi-class models"),
            ("objective","Add objective function while solving"),
            ("all_opt","Enable all optimizations"),
            ("compute_data_distance","Compute mimimum distance from the data"),
    ]:
        parser.add_argument(f"--{feature}", action="store_true", default=False, help=help_text)
        parser.add_argument(f"--no-{feature}", dest=f"{feature}", action="store_false")

    parser.set_defaults(all_opt=True)
    parser.add_argument("--plot", action="store_true", help="plot the results on display")

    parser.add_argument("--plot_file",type=str, default=None, help="save the plot in a file" )
    
    parser.add_argument(
        "--sure_counterexamples",
        action="store_true",
        help="Be sure about counterexamples and unsure about fairness",
    )
    parser.add_argument(
        "--gap", type=float, default=0.2, help="Gap for checking sensitivity (deprecated)"
    )

    
    parser.add_argument(
        "--output_gap",
        type=float,
        nargs=2,
        default=None,
        help="Give the expected gap in the probability of the model"
    )    
    
    parser.add_argument(
        "--precision", type=float, default=0, help="Scale for checking sensitivity"
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help="Indexes of the features for which to do sensitivity analysis",
    )

    #local sensitivity argument added
    parser.add_argument(
        "--local_check_sample",
        type=float,
        nargs='+',
        default=None,
        help="input vector to check sensitivity in the vicinity"
    )

    parser.add_argument(
          "--anchor",
          action="store_true",
          help="Anchored local sensitivity: the --local_check_sample point IS the second "
               "witness; search for one counterfactual relative to it (use with --perturb 0)",
      )
    
    parser.add_argument(
        "--numeric",
        action="store_true",
        help="includes only numeric features in the sensitive features "
    )
    
    parser.add_argument(
            "--categorical",
            action="store_true",
            help="includes only categorical features in the sensitive features "
    )

    #local sensitivity argument added
    parser.add_argument(
        "--local_check_file",
        type=str,
        default=None,
        help="File containing samples for which we need to run the tool",
    )

    #perturbation argument added
    parser.add_argument(
        "--perturb",
        type=float,
        default=0.1,
        help="maximum perturbation allowed for an insensitive variable"
    )

    parser.add_argument(
        "--details",
        type=str,
        default=None,
        help="File containing names of features and their bounds",
    )

    parser.add_argument(
        "--log",
        type=str,
        default= None, #'/tmp/senstivity/',
        help="File where dump the log of run.",
    )

    parser.add_argument(
        "--log_folder",
        type=str,
        default= '/tmp/',
        help="Folder where all the log run are dumped.",
    )
    parser.add_argument(
        "--no_remove",
        action="store_true",
        default= False,
        help="do not delete the intermediate folder for storing logs",
    )
    
    parser.add_argument(
        "--time",
        type=float,
        default=1e8,
        help="Stopping time (in seconds), only for veritas",
    )

    parser.add_argument(
        "--in_distro_clauses",
        type=str,
        default="",
        help="File containing clauses that encodes valid combinations in the seen data")

    parser.add_argument(
        "--data_file",
        type=str,
        default="",
        help="File containing training data")

    # parser.add_argument(
    #     "--pca_data",
    #     type=str,
    #     default="",
    #     help="Training CSV used to fit PCA constraints",
    # )

    parser.add_argument(
        "--pca",
        action="store_true",
        help="Add PCA in-distribution constraints, fitted on --data_file",
    )


    parser.add_argument(
        "--pca_d",
        type=int,
        default=0,
        help=" provide  d otherwise d will be automatically selected",
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Sets the level of verbosity of the tool!",
    )
    #---------------------------------------------------------
    #   monitor specific arguments
    #---------------------------------------------------------
    parser.add_argument(
        "--epsilon",
        type=float,
        help="monitor:Epsilon for the FRNN monitor",
        default=0.2
    )
    parser.add_argument(
        "--predcolname",
        type=str,
        help="monitor:Name of the column containing the model's prediction",
        default="pred"
    )
    parser.add_argument(
        "--cfeaturefile",
        type=str,
        help="monitor:File containing list of all features with type for clemont only" ,
        default=None
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["linf", "l2","l1","l0"],
        default="l2",
        help="monitor:Distance metric for FRNN(linf or l2) /data-ware"
    )
    parser.add_argument("--gowal",action='store_true',default=False,help="enable to include gowal distance")
    parser.add_argument("--binary_perturb",type=int,default=0,help=" binary/categorical features' perturbation")
    parser.add_argument("--warnings", action='store_true', default=False, help="Show Python warnings (hidden by default)")
    parser.add_argument(
        "--cmd",
        action="store_true",
        default= False,
        help="print the command used to run the tool with the given arguments and exit",
    )
    #---------------------------
    # glitch argument
    #--------------------------
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="glitch: the magnitude of the glitch to look for. 0 maximises it, "
             "A>0 asks whether one of magnitude A exists. With --features the "
             "search runs along that feature.",
    )
    
    # Parse the arguments
    args = parser.parse_args()
    if args.cmd:
        import sys
        print(" ".join(sys.argv))
    
    if args.solver == "monitor":
        if args.metric not in ["linf","l2"]:
            print(f"for monitor, select metric from linf,l2")
    
    if args.output_gap:
        utils.output_gap_check(args.output_gap)
        if args.output_gap[1] < args.output_gap[0]:
            args.output_gap = [args.output_gap[1],args.output_gap[0]]
            
    utils.multiclass_check(args.multiclass,args.truelabel,args.otherlabel)
    
    if args.all_opt:
        args.objective = False
        args.unaffected_cons = True
        args.affected_cons = True
        args.ancestor_cons = True
        pass
    
    if args.solver == None:
        args.solver = "pb"
    options = arguments_to_options(args)
    
    return args,options
