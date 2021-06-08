import os
import warnings
import sys
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('glass.csv')
# Load Diabetes datasets

X = df.drop(['Type'],axis=1)
y = df['Type']

# Import mlflow
import mlflow
import mlflow.sklearn

# Evaluate metrics
def eval_metrics(actual, pred):
    recall = recall_score(actual, pred)
    precision = precision_score(actual, pred)
    return recall,precision

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)

    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["Type"], axis=1)
    test_x = test.drop(["Type"], axis=1)
    train_y = train[["Type"]]
    test_y = test[["Type"]]
    
    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    # Run ElasticNet
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (recall, precision) = eval_metrics(test_y, predicted_qualities)
    # Print out ElasticNet model metrics
    print("  Precision: %s" % precision)
    print("  Recall: %s" % recall)

    # Log mlflow attributes for mlflow UI
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(lr, "model")