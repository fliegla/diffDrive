import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

import numpy as np

DEFAULT_CAMERA_CONFIG = {}

class DiffDriveEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description
    "DiffDrive is a robot with differential drive. The goal is to move the robot close to the goal
    at the end of the track.
    
    ### Action Space
    The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
    | Num | Action                                         |Control Min | Control Max | Name (in corresponding XML file) | Joint |      Unit    |
    |-----|------------------------------------------------|------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied at the left wheel's hinge       |    0       |     100     |         left-wheel               | hinge | torque (N m) |
    | 1   | Torque applied at the right wheel's hinge      |    0       |     100     |         right-wheel              | hinge | torque (N m) |
    
    ### Observation Space
    Observations consist of
    - The coordinates of the robot's chassis
    - The coordinates of the goal
    - The velocity of the robot
    - The vector between the goal and the robot's chassis (3 dimensional with the last element being 0)
    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:
    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | x-coordinate of the chassis                                                                    | -Inf | Inf | chassis                          | free  | position (m)             |
    | 1   | y-coordinate of the chassis                                                                    | -Inf | Inf | chassis                          | free  | position (m)             |
    | 3   | x-coordinate of the goal                                                                       | -Inf | Inf | goal                             | free  | position (m)             |
    | 4   | y-coordinate of the goal                                                                       | -Inf | Inf | goal                             | free  | position (m)             |
    | 6   | x-coordinate velocity of the chassis                                                           | -Inf | Inf | chassis                          | free  | velocity (m/s)           |
    | 7   | y-coordinate velocity of the chassis                                                           | -Inf | Inf | chassis                          | free  | velocity (m/s)           |
    | 8   | x-value of position_chassis - position_goal                                                    | -Inf | Inf | NA                               | free  | position (m)             |
    | 9   | y-value of position_chassis - position_goal                                                    | -Inf | Inf | NA                               | free  | position (m)             |
    | 10  | z-value of position_chassis - position_goal                                                    | -Inf | Inf | NA                               | free  | position (m)             |

    ### Rewards
    The reward consists of two parts:
    - *reward_distance*: This reward is a measure of how far the robot
     is from the target, with a more negative
    value assigned for when the robot is further away from the
    target. It is calculated as the negative vector norm of (position of
    the robot - position of goal), or *-norm("chassis" - "goal")*.
    - *reward_control*: A negative reward for penalising the robot if
    it takes actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.
    The total reward returned is ***reward*** *=* *reward_distance + reward_control*

    ### Starting State
    Observations start in state
    (-24.0, 0.0, 24.5, 0.0, 0.0, 0.0, 0.0, 0.0, 48.5, 0.0, 0.0)
  
    ### Episode End
    The episode ends when any of the following happens:
    1. Truncation: The episode duration reaches a 50 timesteps.
    2. Termination: Any of the state space values is no longer finite.
    
    ### Arguments
    No additional arguments
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 30,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        MujocoEnv.__init__(self, "diffdrive.xml", 4, observation_space=observation_space, **kwargs)

    def step(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return (obs, reward, False, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl))

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:2],
                self.data.qpos.flat[8:10],
                self.data.qvel.flat[:2],
                self.get_body_com("chassis") - self.get_body_com("goal"),
            ]
        )
