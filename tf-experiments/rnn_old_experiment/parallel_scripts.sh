nohup python3.5 generator.py --gpu 7 --time_steps 1200 --rnn_sizes "128" --fc_sizes "64, 32" > logs/test1.out &
nohup python3.5 generator.py --gpu 6 --time_steps 1200 --rnn_sizes "128, 128" --fc_sizes "64" > logs/test2.out &
nohup python3.5 generator.py --gpu 5 --time_steps 1200 --rnn_sizes "256, 256" --fc_sizes "1" > logs/test3.out &
nohup python3.5 generator.py --gpu 4 --time_steps 1200 --rnn_sizes "256, 256" --fc_sizes "128" > logs/test4.out &
