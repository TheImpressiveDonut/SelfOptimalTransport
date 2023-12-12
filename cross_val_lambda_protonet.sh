#!/bin/bash

lambdas=(5 10 20)
n_iters=(10 20 50)
method="maml"
dataset="tabula_muris"

for n_iter in "${n_iters[@]}"
do
  for lambda in "${lambdas[@]}"
  do
    echo "Running with lambda $lambda and n_iter $n_iter"
    echo "-------------------------------------"
    python run.py exp.name=CV_"$method"_"$dataset" method="$method" dataset="$dataset" sot=true lambda_="$lambda" n_iters="$n_iter"
    echo "====================================="
  done
done
