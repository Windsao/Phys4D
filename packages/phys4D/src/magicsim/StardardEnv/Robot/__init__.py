import gymnasium as gym

gym.register(
    id="TaskBaseEnv-V0",
    entry_point="magicsim.StardardEnv.Robot.TaskBaseEnv:TaskBaseEnv",
)
