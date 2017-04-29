#! /bin/bash
url="https://physionet.org/challenge/2017/training2017.zip"
mkdir -p "raw"
curl $url -o "raw/training2017.zip"
unzip "training2017.zip"
rm "training2017.zip" -rf
