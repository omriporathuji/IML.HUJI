#!/usr/bin/env bash

id=0
cd "$HOME/Uni/IML/IML.HUJI/submissions/ex2"

base_path="$HOME/Uni/IML/IML.HUJI"

cp "$base_path/exercises/house_price_prediction.py" .
cp "$base_path/exercises/city_temperature_prediction.py" .
cp "$base_path/IMLearn/learners/regressors/polynomial_fitting.py" .
cp "$base_path/IMLearn/utils/utils.py" .
cp "$base_path/IMLearn/metrics/loss_functions.py" .
cp "$base_path/IMLearn/learners/regressors/linear_regression.py" .

cp "/Users/omriporat/Downloads/IML - Ex 2.pdf" .
mv "IML - Ex 2.pdf" "Answers.pdf"

rm "ex2_$id.tar"

tar -cf "ex2_$id.tar" polynomial_fitting.py linear_regression.py loss_functions.py utils.py house_price_prediction.py city_temperature_prediction.py Answers.pdf
