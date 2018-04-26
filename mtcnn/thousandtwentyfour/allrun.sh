#!/bin/bash

for i in {0..100} 
  do
    nvprof --metrics flops_sp_add,flops_sp_mul,flops_sp_fma,flops_sp_special --log-file allmetrics$i python main.py --num_result $i
  done
