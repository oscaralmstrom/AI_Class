from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Environment(ABC):
    @abstractmethod
    def get_actions(self, state: State) -> List[int]:
        """ Returns available actions in that state

        @param state: state
        :return List[int]: actions
        """
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """ Returns whether or not state is solved

        @param state: state
        @return: bool that indicaates to whether or not the state is solved
        """
        pass

    @abstractmethod
    def state_action_dynamics(self, state: State, action: int) -> Tuple[float, List[State], List[float]]:
        """ Get transition dynamics for state and action

        @param state: state
        @param action: action
        @return: expected_reward, possible next states, transition probabilities
        """
        pass

    @abstractmethod
    def sample_transition(self, state: State, action: int) -> Tuple[State, float]:
        """ Sample a transition from the environment

        @param state: state
        @param action: action
        @return: next state, reward
        """
        pass

    @abstractmethod
    def states_to_nnet_input(self, state: List[State]) -> np.ndarray:
        """

        @return: state representation to nnet
        """
        pass
