#!/usr/bin/env bash

id=0
cd "$HOME/Uni/IML/IML.HUJI/submissions/ex3"

base_path="$HOME/Uni/IML/IML.HUJI"

cp "$base_path/utils.py" .
cp "$base_path/IMLearn/metrics/loss_functions.py" .
cp "$base_path/exercises/classifiers_evaluation.py" .
cp "$base_path/IMLearn/learners/classifiers/gaussian_naive_bayes.py" .
cp "$base_path/IMLearn/learners/classifiers/linear_discriminant_analysis.py" .
cp "$base_path/IMLearn/learners/classifiers/perceptron.py" .

cp "$HOME/Downloads/IML - Ex 3.pdf" .
mv "IML - Ex 3.pdf" "Answers.pdf"

rm "ex3_$id.tar"

tar -cf "ex3_$id.tar" loss_functions.py utils.py classifiers_evaluation.py gaussian_naive_bayes.py linear_discriminant_analysis.py perceptron.py Answers.pdf
