REGISTRY = {}
from .cooperative_navigation_task_env import CoopNavEnv
from .vn_env import VNEnv
from .blockergame_env import BlockerGameEnv
REGISTRY["vn"] = VNEnv
REGISTRY["blocker"] = BlockerGameEnv
REGISTRY["CoopNav"] = CoopNavEnv


