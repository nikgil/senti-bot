import utils
import pandas as pd
from sklearn.externals import joblib

from kernels.kernel import AbstractKernel
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(AbstractKernel):

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.NB_pipeline = utils.load_from_file("nb_pipeline.pkl")

    def train(self, train_set: pd.DataFrame, force: bool = False, save: bool = True) -> None:
        if not force and self.NB_pipeline is not None:
            return

        self.NB_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
            ('clf', MultinomialNB()),
        ])

        self.NB_pipeline.fit(train_set['comment_text'].map(lambda com: utils.preprocess_text(com)),
                             train_set['bannable'])

        if save:
            utils.dump(self.NB_pipeline, "nb_pipeline.pkl")

    def is_banned(self, message: str, threshold: float = 0.5) -> bool:
        prediction = self.NB_pipeline.predict_proba([utils.preprocess_text(message)])[0]

        return prediction[1] > prediction[0] and prediction[1] > threshold
