#!/bin/bash

for i in {0..100} 
  do
    nvprof --metrics flop_count_sp,flop_count_hp --log-file metric$i python main.py --num_result $i
  done
