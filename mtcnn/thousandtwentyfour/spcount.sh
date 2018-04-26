#!/bin/bash

for i in {0..20} 
  do
    grep flop_count_sp ./profiles/metric$i | awk '{s+=$9} END {printf "%.0f,", s}' >> allflops.csv
  done
