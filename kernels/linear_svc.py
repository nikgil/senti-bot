import utils
import pandas as pd
from kernels.kernel import AbstractKernel
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC as LSVC


class LinearSVC(AbstractKernel):

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.SVC_pipeline = utils.load_from_file("lsvc_pipeline")

        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def train(self, train_set: pd.DataFrame, force: bool = False, save: bool = True) -> None:
        if not force and self.SVC_pipeline is not None:
            return

        self.SVC_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
            ('clf', LSVC()),
        ])

        self.SVC_pipeline.fit(train_set['comment_text'].map(lambda com: utils.preprocess_text(com)),
                              train_set['bannable'])

        if save:
            utils.dump(self.SVC_pipeline, "lsvc_pipeline")

    def is_banned(self, message: str, threshold: float = 0.5) -> bool:
        prediction = self.SVC_pipeline.predict([utils.preprocess_text(message)])

        return prediction[0] != 0
