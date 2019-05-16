import os
import re
# import nltk
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline


def dump(classifier: Pipeline, file_name: str):
    """
    Dumps a classifier pipeline into the cache folder under the given file_name.pkl

    :param classifier: Classifier to be pickled
    :param file_name: Name of file to be saved
    :return: Nothing
    """
    assert file_name is not None

    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"

    joblib.dump(classifier, os.path.join("cache", file_name))


def load_from_file(file_name):
    """
    Retrieves a classifier from the cache folder with provided name.

    :param file_name: Name of file to be retrieved as classifier
    :return: The classifier pipeline, if one exists. None if nothing found
    """
    assert file_name is not None

    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"

    try:
        return joblib.load(os.path.join("cache", file_name))
    except FileNotFoundError as e:
        return None


def preprocess_text(text):
    """
    Default pre-processor
    :param text:
    :return:
    """
    # one case
    text = text.lower()

    # remove various whitespaces
    text = text.strip()

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
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # # stem it all
    # sno = nltk.SnowballStemmer('english')
    # text = sno.stem(text)

    return text
