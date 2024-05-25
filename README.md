# README for COMPSCI 589 Homework 1

## Overview
This homework consists of two main parts and two extra credit tasks. The code for each part is developed in Jupyter Notebook and then converted into Python scripts for submission. Below are the instructions for running each script to replicate the experiments and analyses.

## Environment Setup
Before running the scripts, please ensure you have a Python 3 environment set up with the necessary libraries installed. 

## Note for TAs
The original code was developed in Jupyter Notebooks, it has been converted into Python scripts for submission as requested. To review the code's logic and the execution flow please see the notebook (.ipynb) version because this is how I was working on it, and it's easier to understand. I answer the questions in the notebook and write the code too.

### Part 1: k-NN algorithm
**File Name: 'part1.py' or the notebook version for interactive visualization 'part1.ipynb'.**
In this script I implemented the k-NN algorithm and evaluated on the Iris dataset.
It should output the 3 graphs showing the algorithm's accuracy on both training and testing sets for the various values of k and the testing graph for when we don't normalize the features.
I also added a couple of extra graphs to help me compare both sets and their accuracies to find an optimal value for k. (for both normalization and no normaliztion)
Please refer to the notebook to see the steps it helps understand the logic and implementation. <br />
To run this script, use the following command in your terminal: **python part1.py**

### Part 2: Decision Tree algorithm Information Gain
**File Name: 'part2.py' or the notebook version for interactive visualization 'part2.ipynb'.**
In this script I implemented the Decision Tree algorithm using Information Gain and evaluated it on the 1984 United States Congressional Voting dataset. It will generate histograms displaying the accuracy distribution over training and testing sets.
Please refer to the notebook to see the steps it helps understand the logic and implementation. <br />
To run this script, use the following command in your terminal: **python part2.py**

### Extra Credit 1: Decision Tree with Gini Criterion
**File Name: 'extra_credit1.py' or the notebook version for interactive visualization 'extra_credit1.ipynb'.**
This script is a variation of the Decision Tree implementation that uses the Gini criterion for node splitting. I evaluated it on the 1984 United States Congressional Voting dataset. It will generate histograms displaying the accuracy distribution over training and testing sets. (same as part 2)
Please refer to the notebook to see the steps it helps understand the logic and implementation. <br />
To run this script, use the following command in your terminal: **extra_credit1.py**

### Extra Credit 2: Preventing Overfitting in Decision Trees
**File Name: 'extra_credit2.py' or the notebook version for interactive visualization 'extra_credit2.ipynb'.**
This script modifies the Decision Tree algorithm to prevent overfitting by introducing a stopping criterion based on class majority.
Please refer to the notebook to see the steps it helps understand the logic and implementation. <br />
To run this script, use the following command in your terminal: **extra_credit2.py**

