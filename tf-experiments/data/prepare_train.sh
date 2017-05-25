#! /bin/bash
./download_raw.sh
./separator.py --ref "raw/training2017/REFERENCE.csv" --train 80 --val 0
./writer.py --from_dir "raw/training2017/" --ref "raw/training2017/TRAIN.csv"  --to "TFRecords/orig_train" &
./augment.py 6000 --from_dir "raw/training2017" --ref "raw/training2017/TRAIN.csv" --to_dir "raw/augmented"
./separator.py --ref "raw/augmented/TRAIN.csv" --train 80 --val 20
./writer.py --from_dir "raw/augmented/" --ref "raw/augmented/TRAIN.csv"  --to "TFRecords/aug_train"
