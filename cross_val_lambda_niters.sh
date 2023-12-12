#!/bin/bash

lambdas=(5 10 20)
n_iters=(10 20 50)
dataset="tabula_muris"

method="maml"

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

python run.py exp.name=CV_"$method"_"$dataset" method="$method" dataset="$dataset"

method="protonet"

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

python run.py exp.name=CV_"$method"_"$dataset" method="$method" dataset="$dataset"

method="matchingnet"

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

python run.py exp.name=CV_"$method"_"$dataset" method="$method" dataset="$dataset"
