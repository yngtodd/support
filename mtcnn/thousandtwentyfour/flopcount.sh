#!/bin/bash

for i in {0..46} 
  do
    grep flop_count_sp ./profiles/mixedprecision101/metric$i | awk '{s+=$1*$9} END {printf "%.0f,", s}' >> mixedprecision.csv
  done
