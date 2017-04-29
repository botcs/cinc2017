#! /bin/bash
url="https://physionet.org/challenge/2017/training2017.zip"
mkdir -p "raaw"
curl $url -o "raaw/training2017.zip"
unzip "training2017.zip"
rm "training2017.zip" -rf
