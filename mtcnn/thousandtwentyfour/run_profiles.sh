#!/bin/bash

for i in 8 9 10 11 12 13 14 15 16 17 18 19 20 
do
  nvprof --metrics flop_count_sp,flop_count_hp --log-file metric$i python main.py --num_result $i
done
