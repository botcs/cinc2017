#!/bin/sh

ln -sf **/$1.mat .
python3 entry_eval_ppkeitk.py $1
