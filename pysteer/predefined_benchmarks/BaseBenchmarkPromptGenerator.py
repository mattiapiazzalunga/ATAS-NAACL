from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from enums.ModelFamilyEnum import ModelFamilyEnum
from prompt_generator.BasePromptGenerator import BasePromptGenerator


class BaseBenchmarkPromptGenerator(ABC):
    _task: Enum
    _model_family: ModelFamilyEnum = None
    _strategy: BasePromptGenerator = None

    def __init__(self, task: Enum, model_family: ModelFamilyEnum):
        self.task = task
        self.model_family = model_family

    @property
    def model_family(self) -> ModelFamilyEnum:
        return self._model_family

    @model_family.setter
    def model_family(self, model_family: ModelFamilyEnum):
        if model_family is None or not isinstance(model_family, ModelFamilyEnum):
            raise ValueError("model_family must be not None and an instance of ModelFamilyEnum.")
        self._model_family = model_family

    @property
    def strategy(self) -> BasePromptGenerator:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BasePromptGenerator):
        if strategy is None or not isinstance(strategy, BasePromptGenerator):
            raise ValueError("Strategy must be not None and an instance of BasePromptGenerator.")
        self._strategy = strategy

    @property
    @abstractmethod
    def task(self) -> Enum:
        """Get the current task."""
        pass

    @task.setter
    @abstractmethod
    def task(self, task: Enum):
        pass

    @abstractmethod
    def _get_who_you_are(self) -> str:
        pass

    @abstractmethod
    def _get_system_message(self) -> str:
        pass

    @abstractmethod
    def _get_user_message(self, *args, **kwargs) -> str:
        pass

    def _generate_prompt_based_on_the_model(
            self, system: str, user: str, who_you_are: Optional[str] = None
    ) -> str:
        if self.strategy is None:
            raise ValueError("Strategy must be set before generating prompt.")
        return self.strategy.generate_prompt(system, user, who_you_are)

    @abstractmethod
    def generate_prompt_for_the_task(self, *args, **kwargs) -> str:
        pass
