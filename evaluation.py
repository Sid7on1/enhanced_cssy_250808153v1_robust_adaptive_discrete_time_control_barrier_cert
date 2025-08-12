import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from evaluation.metrics import calculate_velocity_threshold, calculate_flow_theory
from evaluation.constants import (
    VELOCITY_THRESHOLD_THRESHOLD,
    FLOW_THEORY_THRESHOLD,
    SAFE_SET_SIZE,
    MAX_ITERATIONS,
    VELOCITY_THRESHOLD_TOLERANCE,
    FLOW_THEORY_TOLERANCE,
)
from evaluation.exceptions import EvaluationError
from evaluation.utils import (
    validate_input,
    validate_config,
    calculate_safe_set,
    calculate_control_input,
)
from evaluation.models import SafeSet, ControlInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentEvaluator:
    def __init__(self, config: Dict):
        self.config = validate_config(config)
        self.safe_set = SafeSet(self.config["safe_set_size"])
        self.control_input = ControlInput(self.config["control_input"])

    def evaluate(self, agent_output: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the agent's output using the velocity-threshold and flow-theory metrics.

        Args:
            agent_output (np.ndarray): The agent's output.

        Returns:
            Tuple[float, float]: The velocity-threshold and flow-theory metrics.
        """
        try:
            velocity_threshold = calculate_velocity_threshold(
                agent_output, self.safe_set, self.control_input
            )
            flow_theory = calculate_flow_theory(
                agent_output, self.safe_set, self.control_input
            )
            return velocity_threshold, flow_theory
        except EvaluationError as e:
            logger.error(f"Error evaluating agent output: {e}")
            raise

    def calculate_velocity_threshold(self, agent_output: np.ndarray) -> float:
        """
        Calculate the velocity-threshold metric.

        Args:
            agent_output (np.ndarray): The agent's output.

        Returns:
            float: The velocity-threshold metric.
        """
        return calculate_velocity_threshold(
            agent_output, self.safe_set, self.control_input
        )

    def calculate_flow_theory(self, agent_output: np.ndarray) -> float:
        """
        Calculate the flow-theory metric.

        Args:
            agent_output (np.ndarray): The agent's output.

        Returns:
            float: The flow-theory metric.
        """
        return calculate_flow_theory(
            agent_output, self.safe_set, self.control_input
        )

class EvaluationError(Exception):
    pass

class SafeSet:
    def __init__(self, size: int):
        self.size = size
        self.points = np.random.rand(size, 2)

class ControlInput:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.input = np.random.rand(input_size)

def calculate_safe_set(config: Dict) -> SafeSet:
    """
    Calculate the safe set.

    Args:
        config (Dict): The configuration.

    Returns:
        SafeSet: The safe set.
    """
    return SafeSet(config["safe_set_size"])

def calculate_control_input(config: Dict) -> ControlInput:
    """
    Calculate the control input.

    Args:
        config (Dict): The configuration.

    Returns:
        ControlInput: The control input.
    """
    return ControlInput(config["control_input"])

def validate_input(input_data: np.ndarray) -> None:
    """
    Validate the input data.

    Args:
        input_data (np.ndarray): The input data.

    Raises:
        EvaluationError: If the input data is invalid.
    """
    if input_data.shape[0] != SAFE_SET_SIZE:
        raise EvaluationError("Invalid input size")

def validate_config(config: Dict) -> Dict:
    """
    Validate the configuration.

    Args:
        config (Dict): The configuration.

    Returns:
        Dict: The validated configuration.
    """
    if "safe_set_size" not in config:
        raise EvaluationError("Missing safe set size in configuration")
    if "control_input" not in config:
        raise EvaluationError("Missing control input in configuration")
    return config

def calculate_velocity_threshold(
    agent_output: np.ndarray, safe_set: SafeSet, control_input: ControlInput
) -> float:
    """
    Calculate the velocity-threshold metric.

    Args:
        agent_output (np.ndarray): The agent's output.
        safe_set (SafeSet): The safe set.
        control_input (ControlInput): The control input.

    Returns:
        float: The velocity-threshold metric.
    """
    # Calculate the velocity-threshold metric
    velocity_threshold = np.linalg.norm(agent_output - safe_set.points[0])
    return velocity_threshold

def calculate_flow_theory(
    agent_output: np.ndarray, safe_set: SafeSet, control_input: ControlInput
) -> float:
    """
    Calculate the flow-theory metric.

    Args:
        agent_output (np.ndarray): The agent's output.
        safe_set (SafeSet): The safe set.
        control_input (ControlInput): The control input.

    Returns:
        float: The flow-theory metric.
    """
    # Calculate the flow-theory metric
    flow_theory = np.dot(agent_output, control_input.input)
    return flow_theory

if __name__ == "__main__":
    # Example usage
    config = {
        "safe_set_size": SAFE_SET_SIZE,
        "control_input": MAX_ITERATIONS,
    }
    evaluator = AgentEvaluator(config)
    agent_output = np.random.rand(SAFE_SET_SIZE, 2)
    velocity_threshold, flow_theory = evaluator.evaluate(agent_output)
    print(f"Velocity-threshold metric: {velocity_threshold}")
    print(f"Flow-theory metric: {flow_theory}")