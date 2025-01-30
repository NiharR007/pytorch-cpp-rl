"""
Contains a class that trains an agent.
"""
import logging
from typing import Tuple
import numpy as np
import gymnasium as gym

from gym_server.envs import make_vec_envs
from gym_server.messages import (InfoMessage, MakeMessage, ResetMessage,
                                 StepMessage)
from gym_server.zmq_client import ZmqClient


RUNNING_REWARD_HORIZON = 10


class Server:
    """
    When `Server.serve()` is called, provides a ZMQ based API for training
    RL agents on OpenAI gym environments.
    """

    def __init__(self, zmq_client: ZmqClient):
        self.zmq_client: ZmqClient = zmq_client
        self.env: gym.Env = None
        logging.info("Gym server initialized")

    def serve(self):
        """
        Run the server.
        """
        logging.info("Serving")
        try:
            self.__serve()
        except KeyboardInterrupt:
            pass

    def _serve(self):
        """
        Serves forever.
        """
        while True:
            message = self.zmq_client.receive()
            if message["method"] == "make":
                self.make(**message["param"])
                self.zmq_client.send("OK")
            elif message["method"] == "info":
                info_dict = self.info()  # Get the info dictionary
                self.zmq_client.send(info_dict)  # Send the entire dictionary
            elif message["method"] == "reset":
                obs = self.reset()
                self.zmq_client.send(obs)
            elif message["method"] == "step":
                obs = self.step(**message["param"])
                self.zmq_client.send(obs)
            elif message["method"] == "close":
                self.close()
                self.zmq_client.send("OK")
                break

    def info(self) -> dict:
        """Returns information about the environment's spaces."""
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        
        return {
            "action_space_type": type(action_space).__name__,
            "action_space_shape": [int(action_space.n)],  # Convert to simple integer
            "observation_space_type": "Box",
            "observation_space_shape": [int(dim) for dim in observation_space.shape],  # Convert to simple integers
            "observation_space_bounds": {
                "low": observation_space.low.tolist(),  # Convert numpy arrays to lists
                "high": observation_space.high.tolist()
            }
        }


    def make(self, env_name, num_envs):
        """
        Makes a vectorized environment of the type and number specified.
        """
        logging.info("Making %d %ss", num_envs, env_name)
        self.env = make_vec_envs(env_name, 0, num_envs)
        print("Action space:", self.env.action_space)
        print("Observation space:", self.env.observation_space)

    def reset(self) -> dict:
        logging.info("Resetting environments")
        try:
            result = self.env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
                info = {}
            
            if isinstance(obs, np.ndarray):
                if len(obs.shape) == 2:
                    obs = obs.reshape(-1).tolist()
                else:
                    obs = obs.tolist()
            
            print("Debug - Observation:", obs)
            print("Debug - Info:", info)
            
            return {"observation": obs, "info": info}
            
        except Exception as e:
            logging.error(f"Reset error: {e}")
            raise


    def step(self, actions, render=False):
        """
        Steps the environments with the given actions.
        """
        # Convert actions to integers
        if isinstance(actions, list):
            actions = [int(a[0]) for a in actions]
        
        logging.info(f"Stepping environments with actions: {actions}")
        
        # Get step results
        obs, rewards, dones, infos = self.env.step(actions)
        
        # Ensure everything is a flat list
        obs_flat = obs.reshape(-1).tolist()  # Flatten observation
        rewards_flat = rewards.tolist()       # Keep rewards as is
        dones_flat = [bool(d) for d in dones] # Convert to simple boolean list
        
        logging.debug(f"Sending obs shape: {len(obs_flat)}")
        logging.debug(f"Sending rewards shape: {len(rewards_flat)}")
        
        return {
            "observation": obs_flat,          # Send as flat list
            "reward": rewards_flat,           # Single list of rewards
            "done": dones_flat,              # Single list of booleans
            "info": {"reward": rewards_flat}  # Keep info simple
        }
   

    __info = info
    __make = make
    __reset = reset
    __serve = _serve
    __step = step
