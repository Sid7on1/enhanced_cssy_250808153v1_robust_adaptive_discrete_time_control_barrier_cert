import logging
import random
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """

    def __init__(self, capacity: int):
        """
        Initializes the replay buffer.

        :param capacity: Maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        """
        Returns the current number of transitions in the buffer.

        :return: Number of transitions in the buffer.
        """
        return len(self.buffer)

    def append(self, transition: Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]) -> None:
        """
        Appends a new transition to the buffer.

        :param transition: Tuple containing (state, action, reward, next_state, done).
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, ArrayLike]:
        """
        Samples a batch of transitions from the buffer.

        :param batch_size: Number of transitions to sample.
        :return: Dictionary containing the sampled transitions, with keys 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = {
            "states": np.array([transition[0] for transition in transitions]),
            "actions": np.array([transition[1] for transition in transitions]),
            "rewards": np.array([transition[2] for transition in transitions]),
            "next_states": np.array([transition[3] for transition in transitions]),
            "dones": np.array([transition[4] for transition in transitions]),
        }
        return batch

    def clear(self) -> None:
        """
        Clears the replay buffer, removing all stored transitions.
        """
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer that samples transitions with probabilities proportional to their priority.
    """

    def __init__(self, capacity: int, alpha: float):
        """
        Initializes the prioritized replay buffer.

        :param capacity: Maximum number of transitions to store in the buffer.
        :param alpha: Priority exponent for sampling probabilities.
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self.total_priority = 0.0

    def append(self, transition: Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool], priority: float) -> None:
        """
        Appends a new transition to the buffer with the specified priority.

        :param transition: Tuple containing (state, action, reward, next_state, done).
        :param priority: Priority of the transition for sampling.
        """
        super().append(transition)
        self.priorities.append(priority**self.alpha)
        self.total_priority += priority**self.alpha

    def sample(self, batch_size: int) -> Dict[str, ArrayLike]:
        """
        Samples a batch of transitions based on their priorities.

        :param batch_size: Number of transitions to sample.
        :return: Dictionary containing the sampled transitions and their importance weights, with keys 'states', 'actions', 'rewards',
                 'next_states', 'dones', 'weights'.
        """
        if self.total_priority == 0:
            raise ValueError("Total priority is zero. Cannot sample from an empty buffer.")

        priorities = np.array(self.priorities)
        probabilities = priorities / self.total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        transitions = [self.buffer[index] for index in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = {
            "states": np.array([transition[0] for transition in transitions]),
            "actions": np.array([transition[1] for transition in transitions]),
            "rewards": np.array([transition[2] for transition in transitions]),
            "next_states": np.array([transition[3] for transition in transitions]),
            "dones": np.array([transition[4] for transition in transitions]),
            "weights": weights,
        }
        return batch

    def update_priorities(self, indices: ArrayLike, priorities: ArrayLike) -> None:
        """
        Updates the priorities of specific transitions in the buffer.

        :param indices: Indices of the transitions to update.
        :param priorities: New priorities for the specified transitions.
        """
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority**self.alpha
            self.total_priority += priority**self.alpha - self.priorities[index]
            self.priorities[index] = priority**self.alpha


class TransitionDataset(Dataset):
    """
    Dataset for sampling transitions from a replay buffer.
    """

    def __init__(self, buffer: Union[ReplayBuffer, PrioritizedReplayBuffer]):
        """
        Initializes the transition dataset.

        :param buffer: Replay buffer containing the transitions.
        """
        self.buffer = buffer

    def __len__(self) -> int:
        """
        Returns the number of transitions in the dataset.

        :return: Number of transitions.
        """
        return len(self.buffer)

    def __getitem__(
        self, index: int
    ) -> Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool, float]:
        """
        Gets the transition at the specified index.

        :param index: Index of the transition to retrieve.
        :return: Tuple containing (state, action, reward, next_state, done, weight).
        """
        transition = self.buffer.buffer[index]
        weight = 1.0
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            weight = self.buffer.priorities[index]
        return transition + (weight,)


class Memory:
    """
    Memory module for storing and retrieving experiences.
    """

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        beta: float = 0.6,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initializes the memory module.

        :param capacity: Maximum number of transitions to store in the buffer.
        :param prioritized: Whether to use prioritized experience replay.
        :param beta: Importance sampling exponent for prioritized replay.
        :param device: Device on which to store the data.
        :param kwargs: Additional keyword arguments.
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.beta = beta
        self.device = device

        if prioritized:
            self.buffer = PrioritizedReplayBuffer(capacity, beta)
        else:
            self.buffer = ReplayBuffer(capacity)

    def append(
        self,
        state: ArrayLike,
        action: ArrayLike,
        reward: float,
        next_state: ArrayLike,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        Appends a new transition to the buffer.

        :param state: Current state observation.
        :param action: Action taken in the current state.
        :param reward: Reward received after taking the action.
        :param next_state: Next state observation.
        :param done: Whether the episode has terminated.
        :param priority: Priority of the transition for sampling (only used in prioritized replay).
        """
        transition = (state, action, reward, next_state, done)
        if self.prioritized:
            if priority is None:
                raise ValueError("Priority must be specified for prioritized replay.")
            self.buffer.append(transition, priority)
        else:
            self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, Union[Tensor, ArrayLike]]:
        """
        Samples a batch of transitions from the buffer.

        :param batch_size: Number of transitions to sample.
        :return: Dictionary containing the sampled transitions and their weights (if prioritized replay is enabled).
        """
        batch = self.buffer.sample(batch_size)
        data = {
            "states": torch.as_tensor(batch["states"], device=self.device),
            "actions": torch.as_tensor(batch["actions"], device=self.device),
            "rewards": torch.as_tensor(batch["rewards"], device=self.device),
            "next_states": torch.as_tensor(batch["next_states"], device=self.device),
            "dones": torch.as_tensor(batch["dones"], device=self.device),
        }
        if self.prioritized:
            data["weights"] = torch.as_tensor(batch["weights"], device=self.device)
        return data

    def update_priorities(self, indices: ArrayLike, priorities: ArrayLike) -> None:
        """
        Updates the priorities of specific transitions in the buffer (only used in prioritized replay).

        :param indices: Indices of the transitions to update.
        :param priorities: New priorities for the specified transitions.
        """
        if not self.prioritized:
            raise NotImplementedError("Priority updates are only supported with prioritized replay.")
        self.buffer.update_priorities(indices, priorities)

    def clear(self) -> None:
        """
        Clears the replay buffer, removing all stored transitions.
        """
        self.buffer.clear()


class MemoryManager:
    """
    Manages the memory for multiple agents and provides utilities for saving and loading data.
    """

    def __init__(self, capacity: int, num_agents: int, prioritized: bool = False, beta: float = 0.6):
        """
        Initializes the memory manager.

        :param capacity: Maximum number of transitions to store in each agent's buffer.
        :param num_agents: Number of agents.
        :param prioritized: Whether to use prioritized experience replay.
        :param beta: Importance sampling exponent for prioritized replay.
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.prioritized = prioritized
        self.beta = beta
        self.memories = [
            Memory(capacity, prioritized, beta, device=f"cuda:{i}") for i in range(num_agents)
        ]

    def append(
        self,
        agent_id: int,
        state: ArrayLike,
        action: ArrayLike,
        reward: float,
        next_state: ArrayLike,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        Appends a new transition to the specified agent's memory buffer.

        :param agent_id: ID of the agent.
        :param state: Current state observation.
        :param action: Action taken in the current state.
        :param reward: Reward received after taking the action.
        :param next_state: Next state observation.
        :param done: Whether the episode has terminated.
        :param priority: Priority of the transition for sampling (only used in prioritized replay).
        """
        self.memories[agent_id].append(state, action, reward, next_state, done, priority)

    def sample(self, agent_id: int, batch_size: int) -> Dict[str, Union[Tensor, ArrayLike]]:
        """
        Samples a batch of transitions from the specified agent's memory buffer.

        :param agent_id: ID of the agent.
        :param batch_size: Number of transitions to sample.
        :return: Dictionary containing the sampled transitions and their weights (if prioritized replay is enabled).
        """
        return self.memories[agent_id].sample(batch_size)

    def update_priorities(
        self, agent_id: int, indices: ArrayLike, priorities: ArrayLike
    ) -> None:
        """
        Updates the priorities of specific transitions in the specified agent's memory buffer (only used in prioritized replay).

        :param agent_id: ID of the agent.
        :param indices: Indices of the transitions to update.
        :param priorities: New priorities for the specified transitions.
        """
        if self.prioritized:
            self.memories[agent_id].update_priorities(indices, priorities)

    def clear(self) -> None:
        """
        Clears all the memory buffers, removing all stored transitions.
        """
        for memory in self.memories:
            memory.clear()

    def save(self, filepath: str) -> None:
        """
        Saves the memory data to a file.

        :param filepath: Path to the file where the data will be saved.
        """
        # Implement saving logic here

    def load(self, filepath: str) -> None:
        """
        Loads memory data from a file.

        :param filepath: Path to the file from which the data will be loaded.
        """
        # Implement loading logic here


# Example usage
if __name__ == "__main__":
    memory = Memory(1000, prioritized=True, beta=0.5)
    states = np.random.random((10, 5))
    actions = np.random.randint(0, 5, size=(10, 1))
    rewards = np.random.rand(10)
    next_states = np.random.random((10, 5))
    dones = np.random.randint(0, 2, size=(10, 1))

    for i in range(10):
        memory.append(states[i], actions[i], rewards[i], next_states[i], dones[i], priority=0.5)

    batch = memory.sample(32)
    print(batch)

    memory_manager = MemoryManager(1000, 3, prioritized=True)
    for i in range(3):
        memory_manager.append(i, states[i], actions[i], rewards[i], next_states[i], dones[i], priority=0.6)

    batch = memory_manager.sample(0, 16)
    print(batch)