#! /bin/bash
url="https://physionet.org/challenge/2017/training2017.zip"
mkdir -p "raw"
cd "raw"
wget -nc $url -O "training2017.zip"
unzip -nq "training2017.zip"
#rm "training2017.zip" -rf
