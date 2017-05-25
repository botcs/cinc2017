#!/bin/sh

is_running="$(ps aux | grep entry_eval_ppkeitk.py | wc -l)"
# echo $is_running

echo $1 >> RECORDS
if [ $is_running -ne 2 ]; then
  python3.5 entry_eval_ppkeitk.py &
fi
