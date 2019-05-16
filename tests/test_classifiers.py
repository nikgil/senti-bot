import nltk
import pytest
import pandas as pd
from kernels import linear_svc, log_regression, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
df = pd.read_csv("training_files/train.csv", encoding="ISO-8859-1")
train, test = train_test_split(df, test_size=0.33, shuffle=True)

MIN_ACCURACY = 0.9


@pytest.fixture()
def get_nb():
    nb = naive_bayes.NaiveBayes()
    nb.train(train, force=True, save=False)

    yield nb, test


@pytest.fixture()
def get_lsvc():
    lsvc = linear_svc.LinearSVC()
    lsvc.train(train, force=True, save=False)

    yield lsvc, test


@pytest.fixture()
def get_logreg():
    log = log_regression.LogRegression()
    log.train(train, force=True, save=False)

    yield log, test


class TestClassifiers(object):
    def test_nb(self, get_nb):
        classifier = get_nb[0]
        test_set = get_nb[1]

        classed = [classifier.is_banned(comment) for comment in test_set["comment_text"]]

        assert accuracy_score(test_set['bannable'], classed) > MIN_ACCURACY

    def test_svc(self, get_lsvc):
        classifier = get_lsvc[0]
        test_set = get_lsvc[1]

        classed = [classifier.is_banned(comment) for comment in test_set["comment_text"]]

        assert accuracy_score(test_set['bannable'], classed) > MIN_ACCURACY

    def test_logreg(self, get_logreg):
        classifier = get_logreg[0]
        test_set = get_logreg[1]

        classed = [classifier.is_banned(comment) for comment in test_set["comment_text"]]

        assert accuracy_score(test_set['bannable'], classed) > MIN_ACCURACY
