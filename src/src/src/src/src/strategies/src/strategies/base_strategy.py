from abc import ABC, abstractmethod

class FineTuneStrategy(ABC):
    @abstractmethod
    def prepare_model(self, model):
        pass
