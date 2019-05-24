import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from kernels.linear_svc import LinearSVC
from kernels.naive_bayes import NaiveBayes
from kernels.log_regression import LogRegression

nltk.download('stopwords')
df = pd.read_csv("training_files/train.csv", encoding="ISO-8859-1")
train_frame, test = train_test_split(df, test_size=0.33, shuffle=True)

if __name__ == "__main__":
    classifiers = [NaiveBayes(), LinearSVC(), LogRegression()]

    for classifier in classifiers:
        classifier.train(train_frame, save=True)