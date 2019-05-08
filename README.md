# A-Comprehensive-Machine-Learning-Pipeline
*Yeol Ye*  
*University of Chicago*  
*ziyuye@uchicago.edu*  
  


## 0. Introduction
The motivation of this project is to build a comprehensive, modular, extensible, 
machine learning pipeline in Python.  

I recommend you to clone my repository to your local machine, and first view 
the [Pipeline Implementation.ipynb](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/notebook/Pipeline%20Implementation.ipynb)
to have a sense of the pipeline as a whole. (If it cannot open due to a lot of
graphs, you can alternatively open the [Pipeline Implementation.html](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/notebook/Pipeline%20Implementation.html) 
in your browser) After that, you can jump to my [code](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/tree/master/code) 
to see how I designed the functions used in the [notebook](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/notebook/Pipeline%20Implementation.ipynb).

The project build a complete machine learning pipeline. Specifically, it 
deploys the Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, 
Random Forests, Boosting, and Bagging, and uses the [DonorsChoose Project](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) 
prediction as an example. The goal of the example is to predict if 
a project can be fully funded in 60 days.

## 1. [Code](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/tree/master/code)
The directory contains codes designed to build up the pipeline. It is composed by five python
files: 
* Data Preparation: [prep.py](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/code/prep.py); 
* Data Exploration: [explore.py](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/code/explore.py); 
* Feature Engineering: [feature.py](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/code/feature.py); 
* Model Training and Evaluation: [model.py](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/code/model.py); 
* Report: [report.py](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/code/report.py).
  
## 2. [Notebook](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/tree/master/notebook) 
As illustrated above, this directory includes implementation of the pipeline in
the environment of Jupyter Notebook. If you would like to run it on your local 
machine, these packages are required:
* pandas
* numpy
* seaborn
* matplotlib
* datetime
* scipy
* sklearn
* graphviz

Note that there is a short report at the end of the notebook.

## 3. [Data](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/tree/master/data)
The data to be used as an example. Note that this pipeline only uses the [projects_2012_2013.csv](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/blob/master/data/source/projects_2012_2013.csv) 
file in the source directory.

## 4. [Report](https://github.com/ZIYU-DEEP/Machine-Learning-Pipeline/tree/master/data/report)
It contains a [table]() with results across train test splits over time and 
performance metrics (baseline, precision and recall at different thresholds 1%,
 2%, 5%, 10%, 20%, 30%, 50% and AUC_ROC).   
 
It also includes a short [report]() that 
compares the performance of the different classifiers across all the metrics 
for the data set used in this project. The report answers the following questions: 
* Which classifier does better on which metrics? 
* How do the results change over time? 
* What would be your recommendation to someone who's working on this model to identify 5% of posted projects that are at highest risk of not getting fully funded to intervene with? 
* Which model should they decide to go forward with and deploy?


From my result (check the last part in the notebook and also the excel file in the notebook repository or report repository), if there are no limits on intervention, then a simple decision tree classifier with a max depth of 50, a minimum samples split of 5, and calculating splits with a weighted gini score is the most predictive model, with an accuracy of 1.00 and a precision of 0.91 at 100% of the population. 

Comparing it with the baseline: In 2012, there were 84,550 fully funded projects and 33,076 not funded projects out of 117,626 projects, yielding a baseline of 71.88%. Similarly, in 2013, there were 92,399 fully funded projects and 38,930 not funded projects out of 131,329 projects, yielding a baseline of 70.36%. The precision of the decision tree classifier is indeed much higher than the baseline at 91%.

In comparison to these baselines, several of the classifier models outperformed. Simple logistic regression classifiers performed very well with an L1 penalty and C = 10 for both years, at the 1%, 2%, 5%, and 10%. In fact, at 1%, the logistic regression model achieves a precision of nearly 100%. In comparison, tree models like the decision tree and the random forest models did not perform as well at 1%, and quickly decreased in precision to ~85% at the 20% population mark, even when increasing max depth from 5 to 25. The K nearest neighbors model also yielded high precision of ~90% at 1% of the population when using 100 neighbors and the KD tree algorithm, but also declined to ~80% at the 20% population mark.

Therefore, I recommend to use a logistic regression model with an L1 penalty and C = 10. It yields the highest precision at every point from 1% to 50%, and a low recall.