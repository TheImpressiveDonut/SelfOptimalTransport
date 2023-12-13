#!/bin/bash

lambdas=(5 10)
n_iters=(10)
lrs=(0.1 0.01 0.001)
dataset="tabula_muris"
method="protonet"

for n_iter in "${n_iters[@]}"
do
  for lambda in "${lambdas[@]}"
  do
    for lr in "${lrs[@]}"
    do
      echo "Running with lambda $lambda, n_iter $n_iter and lr $lr"
      echo "-------------------------------------"
      python run.py exp.name=CV_"$method"_"$dataset"_lr method="$method" dataset="$dataset" sot=true lambda_="$lambda" n_iters="$n_iter" lr="$lr"
      echo "====================================="
    done
  done
done

python run.py exp.name=CV_"$method"_"$dataset" method="$method" dataset="$dataset"