import pandas as pd
import utils
from kernels.kernel import AbstractKernel
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


class LogRegression(AbstractKernel):

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.LogReg_pipeline = utils.load_from_file("log_pipeline")

        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def train(self, train_set: pd.DataFrame, force: bool = False, save: bool = True) -> None:
        if not force and self.LogReg_pipeline is not None:
            return

        self.LogReg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
            ('clf', LogisticRegression(solver='sag')),
        ])

        self.LogReg_pipeline.fit(train_set['comment_text'].map(lambda com: utils.preprocess_text(com)),
                                 train_set['bannable'])

        if save:
            utils.dump(self.LogReg_pipeline, "log_pipeline")

    def is_banned(self, message: str, threshold: float = 0.5) -> bool:
        prediction = self.LogReg_pipeline.predict_proba([utils.preprocess_text(message)])[0]

        return prediction[1] > prediction[0]
