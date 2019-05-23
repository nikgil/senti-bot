import pandas as pd
from sklearn.model_selection import train_test_split
from kernels.linear_svc import LinearSVC
from kernels.naive_bayes import NaiveBayes
from kernels.log_regression import LogisticRegression

df = pd.read_csv("training_files/train.csv", encoding="ISO-8859-1")
train, test = train_test_split(df, test_size=0.33, shuffle=True)

if __name__ == "__main__":
    classifiers = [NaiveBayes(), LinearSVC, LogisticRegression]

    for classifier in classifiers:
        classifier.train(train, save=True)