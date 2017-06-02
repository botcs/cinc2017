#! /bin/bash
name=$(basename "$1" ".json")
nohup python3.5 train_model_from_template.py "$1"  >> logs/$name &
echo 
