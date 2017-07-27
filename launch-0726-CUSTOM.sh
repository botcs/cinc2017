nohup ./SKIP-FCN-selu-adam.py  0 &
nohup ./SKIP-FCN-halved-selu-adam.py  1 &
nohup ./SKIP-FCN-double-selu-adam.py  2 &

nohup ./SKIP-RES-halved-selu-adam.py  3 &
nohup ./SKIP-RES-selu-adam.py 4 &

nohup ./ENCODE-halved-selu-adam.py 5 &
nohup ./ENCODE-selu-adam.py 6 &


