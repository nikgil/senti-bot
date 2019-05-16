import nltk
import pytest
import pandas as pd
import numpy as np
from kernels import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@pytest.fixture()
def resource():
    nltk.download('stopwords')
    df = pd.read_csv("training_files/train.csv", encoding="ISO-8859-1")
    train, test = train_test_split(df, test_size=0.33)

    nb = naive_bayes.NaiveBayes()
    nb.train(train)

    yield nb, test


class TestClassifiers(object):
    def test_accuracy(self, resource):
        classifier = resource[0]
        test_set = resource[1]

        categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        classed = [classifier.is_banned(comment) for comment in test_set["comment_text"]]

        for category in categories:
            assert accuracy_score([x != 0 for x in test_set[category]], classed) > 0.9

        assert 1 == 0
        # for category in categories:
        #     print(test_set[category])
        #     assert accuracy_score(test_set[category], classed) > 0.99
