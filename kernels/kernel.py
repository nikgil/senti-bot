from abc import ABC, abstractmethod
from pandas import DataFrame

class AbstractKernel(ABC):
    @abstractmethod
    def train(self, train_set: DataFrame) -> None:
        """
        Method called at start to train kernel.
        Note: the provided training set is provided as is, with no filtering of special characters or stop words.

        :param train_set: Key-Value pair of messages with associated tag for training
        :return: Nothing
        """
        pass

    @abstractmethod
    def is_banned(self, message: str, threshold: float = 0.5) -> bool:
        """
        Returns true if a provided message is bad/should be acted on by the bot.

        :param message: The message sent by user to be tested
        :param threshold: float between 0 and 1 to be considered banned (higher means more assertion is needed to be considered bannable)
        :return: true if message is banned, false if passes test
        """
        assert 0.0 <= threshold <= 1.0
        pass
