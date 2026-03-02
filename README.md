## Installations


Installing python dependencies
```
pip install -r requirements.txt
```

## Running the docker image 

If there is difficulty in installing python dependencies. One may use docker to run our tool.

Give the following commands building the docker image.
```
$docker build -t sensitivity .
```


Give the following command to start the docker
```
$docker run -it sensitivity
```
This will take you to a command line interface. 

## Common use case

To run use the following command:
```
python ./src/sensitive.py <model file> --solver <solvername> --output_gap <int pair> --precision <int> --features <int list> 
```

## Example commands 

Sample commands:

## pseudo-Boolean (pb) tool from [SENSITIVITY VERIFICATION FOR ADDITIVE DECISION TREE ENSEMBLES](https://openreview.net/pdf?id=h0vC0fm1q7)
```
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 3 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv 
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver pb
```

## kant (self implementation) of [Evasion and Hardening of Tree Ensemble Classifiers](https://arxiv.org/abs/1509.07892)
```
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp
```

## svim tool from [DataAware and Scalable Sensitivity Analysis for Decision Tree Ensembles](https://arxiv.org/abs/2602.07453)
```
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver pb --all_opt
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver pb --all_opt --prob  
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 7 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/breast_cancer/breast_cancer_train.csv
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 7 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/breast_cancer/breast_cancer_train.csv --in_distro_clauses outputs/output/learned-clauses_breast_cancer_robust.txt
python ./src/sensitive.py models/adult/adult_t200_d5.json --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/adult/details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/adult/train.csv --in_distro_clauses outputs/output/learned-clauses_adult_t200_d5.txt
python ./src/sensitive.py models/adult/adult_t200_d5.json --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/adult/details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/adult/train.csv --in_distro_clauses outputs/output/learned-clauses_adult_t200_d5.txt --local_check_file models/dataset/adult/test.csv 
python ./src/sensitive.py models/adult/adult_t200_d5.json --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/adult/details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/adult/train.csv --in_distro_clauses outputs/output/learned-clauses_adult_t200_d5.txt --local_check_sample  0.089 0.5625 0.09353 0.63 0.7 0.75 0.10 0.9 0.375 0.5 -0.4 0.4 1.45 
```

Sometimes the model does not have enough information such as names and the operating range for the feature. We need to give details file that may provide names of each feature and the operating range of the each feature.

We can run our tool as a local sensitivity search tool also. It takes a list of sensitive feature (--features option), a point around which we search for the sensitivity (--local_check option), and perturbation upto which distance we search for the sensitivity (--perturb option).

```
python ./src/sensitive.py models/adult/adult_t200_d5.json --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/adult/details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/adult/train.csv --in_distro_clauses outputs/output/learned-clauses_adult_t200_d5.txt --local_check_sample  0.089 0.5625 0.09353 0.63 0.7 0.75 0.10 0.9 0.375 0.5 -0.4 0.4 1.45 
```

Data conformal senstivity checking
----------------------------------

Enable checking distance between the data and the sensitivity pair

```
python ./src/sensitive.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 7 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/breast_cancer/breast_cancer_train.csv
```

Generating file that contains summary of patterns
```
./src/learn-data.py --model ./models/tree_verification_models/breast_cancer_robust/0004.resaved.json  --data ./models/dataset/breast_cancer/breast_cancer_train.csv --output ./outputs/learned-clauses_breast_cancer --details ./models/dataset/breast_cancer/breast_cancer_details.csv 
```

Distribution aware search for sensitive pairs

```
python ./src/sensitive.py models/adult/adult_t200_d5.json --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 --details models/dataset/adult/details.csv --solver milp --all_opt --prob --compute_data_distance --data_file models/dataset/adult/train.csv --in_distro_clauses outputs/output/learned-clauses_adult_t200_d5.txt
```

## Options

To view list of all available commands please look

```
python sensitive.py -h for help
```


## Installing roundingsat

```
cd ./utils
./installrounding.sh
```


## Saving and loading the docker image 
Saving image for sharing
```
docker save -o sensitivity.tar sensitivity:latest
```

Loading the docker image
```
docker load -i sensitivity.tar
```




