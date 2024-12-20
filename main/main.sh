
#!/bin/bash


data_path="data/"

dataset_name="ihdp"

python main_nets_loop_multimethod.py --data_path $data_path --dataset_name $dataset_name --pred_test

python main_metrics_datasets.py

python main_nets_loop_synthetic.py

python main_metrics_synthetic.py

python main_nets_loop_toy.py

python main_discovery_eval.py
