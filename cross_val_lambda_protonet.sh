#!/bin/bash

# Array of values
lambdas=(1, 10, 20)
method="protonet"
dataset="tabula_muris"


# Iterate over the array
for value in "${lambdas[@]}"
do
    echo "Running with lambda value $value"
    
    python run.py exp.name=CVlambda_"$value"_"$method"_"$dataset" method="$method" dataset="$dataset" sot=true lambda_="$value"
    
    echo "====================================="
done
