#!/usr/bin/env bash

id=0
cd "$HOME/Uni/IML/IML.HUJI/submissions/ex5"

base_path="$HOME/Uni/IML/IML.HUJI"

cp "$base_path/IMLearn/desent_methods/gradient_descent.py" .
cp "$base_path/IMLearn/desent_methods/learning_rate.py" .
cp "$base_path/IMLearn/desent_methods/modules.py" .
cp "$base_path/IMLearn/learners/classifiers/logistic_regression.py" .
cp "$base_path/exercises/gradient_descent_investigation.py" .

cp "$HOME/Downloads/IML - Ex 5.pdf" .
mv "IML - Ex 5.pdf" "Answers.pdf"

rm "ex5_$id.tar"

tar -cf "ex5_$id.tar" gradient_descent.py learning_rate.py modules.py logistic_regression.py gradient_descent_investigation.py
