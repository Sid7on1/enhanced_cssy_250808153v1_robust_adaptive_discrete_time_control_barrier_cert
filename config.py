import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - Dict[str, Any]: Loaded configuration as a dictionary.
    """
    with open(config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML config file: {e}")
            raise

    return config


class AgentConfig:
    """
    Configuration class for the agent.

    Attributes:
    - env_name (str): Name of the environment.
    - algorithm (Dict[str, Any]): Algorithm-specific configuration.
    - model (Dict[str, Any]): Model-specific configuration.
    - training (Dict[str, Any]): Training-specific configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.env_name = config.get("env_name")
        self.algorithm = config.get("algorithm")
        self.model = config.get("model")
        self.training = config.get("training")

        # Perform configuration validation
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration and raise errors for missing/invalid fields."""
        if not self.env_name:
            raise ValueError("Environment name must be specified in the configuration.")

        if not isinstance(self.algorithm, dict):
            raise ValueError("Algorithm configuration is missing or invalid.")

        required_algo_params = ["velocity_threshold", "flow_theory_enabled"]
        missing_params = [param for param in required_algo_params if param not in self.algorithm]
        if missing_params:
            raise ValueError(
                f"Missing algorithm parameters: {', '.join(missing_params)} in configuration."
            )

        if not isinstance(self.model, dict):
            raise ValueError("Model configuration is missing or invalid.")

        required_model_params = ["input_size", "output_size", "hidden_layers"]
        missing_params = [param for param in required_model_params if param not in self.model]
        if missing_params:
            raise ValueError(
                f"Missing model parameters: {', '.join(missing_params)} in configuration."
            )

        if not isinstance(self.training, dict):
            raise ValueError("Training configuration is missing or invalid.")

        required_training_params = ["batch_size", "learning_rate", "epochs"]
        missing_params = [param for param in required_training_params if param not in self.training]
        if missing_params:
            raise ValueError(
                f"Missing training parameters: {', '.join(missing_params)} in configuration."
            )


class EnvironmentConfig:
    """
    Configuration class for the environment.

    Attributes:
    - observation_space (Dict[str, Any]): Observation space configuration.
    - action_space (Dict[str, Any]): Action space configuration.
    - reward_range (Tuple[float, float]): Range of possible reward values.
    - metadata (Dict[str, Any]): Additional environment metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        self.observation_space = config.get("observation_space")
        self.action_space = config.get("action_space")
        self.reward_range = config.get("reward_range")
        self.metadata = config.get("metadata")

        # Perform configuration validation
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration and raise errors for missing/invalid fields."""
        if not isinstance(self.observation_space, dict):
            raise ValueError("Observation space configuration is missing or invalid.")

        required_obs_params = ["shape", "high", "low", "dtype"]
        missing_params = [param for param in required_obs_params if param not in self.observation_space]
        if missing_params:
            raise ValueError(
                f"Missing observation space parameters: {', '.join(missing_params)} in configuration."
            )

        if not isinstance(self.action_space, dict):
            raise ValueError("Action space configuration is missing or invalid.")

        required_action_params = ["shape", "high", "low", "dtype"]
        missing_params = [param for param in required_action_params if param not in self.action_space]
        if missing_params:
            raise ValueError(
                f"Missing action space parameters: {', '.join(missing_params)} in configuration."
            )

        if not isinstance(self.reward_range, tuple) or len(self.reward_range) != 2:
            raise ValueError("Reward range must be specified as a tuple (min, max) in the configuration.")

        if not isinstance(self.metadata, dict):
            raise ValueError("Environment metadata is missing or invalid in the configuration.")


def create_agent_config(
    env_name: str,
    algorithm: Dict[str, Any],
    model: Dict[str, int],
    training: Dict[str, int],
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for the agent.

    Parameters:
    - env_name (str): Name of the environment.
    - algorithm (Dict[str, Any]): Algorithm-specific configuration.
    - model (Dict[str, int]): Model-specific configuration.
    - training (Dict[str, int]): Training-specific configuration.

    Returns:
    - Dict[str, Any]: Configuration dictionary for the agent.
    """
    config = {
        "env_name": env_name,
        "algorithm": algorithm,
        "model": model,
        "training": training,
    }
    return config


def create_environment_config(
    observation_space: Dict[str, Union[int, float, np.dtype]],
    action_space: Dict[str, Union[int, float, np.dtype]],
    reward_range: Tuple[float, float],
    metadata: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for the environment.

    Parameters:
    - observation_space (Dict[str, Union[int, float, np.dtype]]): Observation space configuration.
    - action_space (Dict[str, Union[int, float, np.dtype]]): Action space configuration.
    - reward_range (Tuple[float, float]): Range of possible reward values.
    - metadata (Dict[str, Any], optional): Additional environment metadata. Defaults to {}.

    Returns:
    - Dict[str, Any]: Configuration dictionary for the environment.
    """
    config = {
        "observation_space": observation_space,
        "action_space": action_space,
        "reward_range": reward_range,
        "metadata": metadata,
    }
    return config


def get_default_agent_config(env_name: str = "XRTrackingEnv") -> Dict[str, Any]:
    """
    Get the default configuration for the agent.

    Parameters:
    - env_name (str, optional): Name of the environment. Defaults to "XRTrackingEnv".

    Returns:
    - Dict[str, Any]: Default agent configuration.
    """
    algorithm_config = {
        "velocity_threshold": 0.5,
        "flow_theory_enabled": True,
        # Add other algorithm-specific parameters here
    }

    model_config = {
        "input_size": 32,
        "output_size": 1,
        "hidden_layers": [64, 32],
        # Add other model-specific parameters here
    }

    training_config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        # Add other training-specific parameters here
    }

    return create_agent_config(env_name, algorithm_config, model_config, training_config)


def get_default_environment_config() -> Dict[str, Any]:
    """
    Get the default configuration for the environment.

    Returns:
    - Dict[str, Any]: Default environment configuration.
    """
    observation_space = {
        "shape": (64, 64, 3),
        "high": 255,
        "low": 0,
        "dtype": np.uint8,
    }

    action_space = {
        "shape": (3,),
        "high": np.array([1.0, 1.0, np1.0]),
        "low": np.array([-1.0, -1.0, -1.0]),
        "dtype": np.float32,
    }

    reward_range = (-float("inf"), float("inf"))

    metadata = {
        "render.modes": ["human", "rgb_array"],
        # Add other metadata here
    }

    return create_environment_config(observation_space, action_space, reward_range, metadata)


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """
    Save the configuration to a YAML file.

    Parameters:
    - config (Dict[str, Any]): Configuration to be saved.
    - config_file (str): Path to the output YAML configuration file.

    Returns:
    - None
    """
    with open(config_file, "w") as file:
        yaml.dump(config, file)


def get_config_from_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse configuration from command-line arguments.

    Parameters:
    - args (List[str]): List of command-line arguments.

    Returns:
    - Dict[str, Any]: Configuration dictionary parsed from the arguments.
    """
    config = {}
    for arg in args:
        key, value = arg.split("=")
        config[key] = type(value)(value)

    return config


def main():
    # Example usage of the configuration classes and functions
    config_file = "agent_config.yaml"

    # Load configuration from a YAML file
    loaded_config = load_config(config_file)

    # Create AgentConfig and EnvironmentConfig instances
    agent_config = AgentConfig(loaded_config.get("agent"))
    env_config = EnvironmentConfig(loaded_config.get("environment"))

    # Access configuration values
    print("Agent Configuration:")
    print("Environment Name:", agent_config.env_name)
    print("Algorithm Velocity Threshold:", agent_config.algorithm["velocity_threshold"])
    print("Model Input Size:", agent_config.model["input_size"])

    print("\nEnvironment Configuration:")
    print("Observation Space Shape:", env_config.observation_space["shape"])
    print("Action Space High:", env_config.action_space["high"])
    print("Reward Range:", env_config.reward_range)

    # Create default configurations
    default_agent_config = get_default_agent_config()
    default_env_config = get_default_environment_config()

    # Save configurations to a new YAML file
    output_config_file = "new_config.yaml"
    save_config(default_agent_config, output_config_file)

    # Parse configuration from command-line arguments
    cmd_args = ["env_name=XRTrackingEnv", "batch_size=64", "learning_rate=0.0005"]
    parsed_config = get_config_from_args(cmd_args)
    print("Parsed Configuration:", parsed_config)


if __name__ == "__main__":
    main()