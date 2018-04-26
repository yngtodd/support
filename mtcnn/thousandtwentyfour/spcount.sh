#!/bin/bash

for i in {0..100} 
  do
    grep flop_count_sp ./profiles/metric$i | awk '{s+=$9} END {printf "%.0f,", s}' >> hundredflops.csv
  done
