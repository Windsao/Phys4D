




import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_DIR = os.path.join(ROOT_DIR, "source", "isaaclab_tasks", "isaaclab_tasks")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "tools", "template", "templates")


SINGLE_AGENT_ALGORITHMS = ["AMP", "PPO"]
MULTI_AGENT_ALGORITHMS = ["IPPO", "MAPPO"]
