#!/usr/bin/env bash

id=0
cd "$HOME/Uni/IML/IML.HUJI/submissions/ex4"

base_path="$HOME/Uni/IML/IML.HUJI"

cp "$base_path/IMLearn/learners/classifiers/decision_stump.py" .
cp "$base_path/IMLearn/metalearners/adaboost.py" .
cp "$base_path/exercises/adaboost_scenario.py" .
cp "$base_path/IMLearn/model_selection/cross_validate.py" .
cp "$base_path/IMLearn/learners/regressors/ridge_regression.py" .
cp "$base_path/exercises/perform_model_selection.py" .

cp "$HOME/Downloads/IML - Ex 4.pdf" .
mv "IML - Ex 4.pdf" "Answers.pdf"

rm "ex4_$id.tar"

tar -cf "ex4_$id.tar" decision_stump.py adaboost.py adaboost_scenario.py cross_validate.py ridge_regression.py perform_model_selection.py Answers.pdf
