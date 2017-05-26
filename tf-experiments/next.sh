#!/bin/sh

is_running="$(ps aux | grep entry_eval_ppkeitk.py | wc -l)"
# echo $is_running

echo "Add $1 to RECORDS"
echo $1 >> RECORDS
if [ $is_running -ne 2 ]; then
  echo "restart background process"
  python3.5 entry_eval_ppkeitk.py &
  sleep 10
fi

is_ready="$(cat answers.txt | grep $1 | wc -l)"
while [ $is_ready -lt 1 ]; do
  echo "waiting for $1"
  sleep .1
  is_ready="$(cat answers.txt | grep $1 | wc -l)"
done
echo "ready"
