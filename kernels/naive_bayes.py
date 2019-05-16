import re
import pandas as pd
from kernels.kernel import AbstractKernel
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


class NaiveBayes(AbstractKernel):

    def __init__(self):
        stop_words = set(stopwords.words('english'))
        self.NB_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])

        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def train(self, train_set: pd.DataFrame) -> None:
        for category in self.categories:
            self.NB_pipeline.train(train_set['comment_text'], train_set[category])

    def is_banned(self, message: str, threshold: float = 0.5) -> bool:
        prediction = self.NB_pipeline.predict(message)

        for category in self.categories:
            if prediction[category] == 1:
                return True

        return False

    # shamelessly stolen from
    # https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Multi%20label%20text%20classification.ipynb
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')

        return text
