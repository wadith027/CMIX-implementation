import gym
import numpy as np
from gym import spaces
import copy


class CoopNavEnv(gym.Env):
    # def __init__(self, n=5, height=5, width=5, acceleration=1, agent_priority=None, collision_threshold=0.5,
                 # collision_penalty=1, seed=None):
    def __init__(self, args):
        self.args = args
        # super(CoopNavEnv, self).__init__()
        self.step_count = 0
        self.n_agents = 5
        self.n = 5
        self.height = 5
        self.width = 5
        self._action_to_acc = 1 * np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
        # normalize agent priority so that it sums to 1
        # self.agent_priority = (agent_priority or np.ones(n)) / np.sum(agent_priority)
        self.agent_priority = np.ones(5) / 5
        self.collision_threshold = 0.5
        self.collision_penalty = 1

        # self.action_space = spaces.MultiDiscrete([5] * self.n_agents)
        self.action_space = [5] * self.n
        # self.observation_space = spaces.Box(low=np.tile(np.array([0, 0, -np.inf, -np.inf]), (self.n, 1)),
        #                                     high=np.tile(np.array([self.height, self.width, np.inf, np.inf]), (self.n, 1)),
        #                                     shape=(self.n, 4))
        # self.observation_space = [self.height * self.width * 2]* self.n
        self.observation_space = [4] * self.n
        self.state_space = 4 * self.n
        self.n_opponent_actions = 2
        self.reset()

    def get_rlinfo(self):
        return self.n, self.state_space, self.observation_space, self.action_space, self.n_opponent_actions

    def setup(self):
        return self.n, self.state_space, self.observation_space, self.action_space, self.n_opponent_actions

    def reset(self, **kwargs):
        self.rng = np.random.default_rng(22)
        self.pos = np.concatenate((self.rng.uniform(0, self.height, self.n)[np.newaxis].T,
                                   self.rng.uniform(0, self.width, self.n)[np.newaxis].T), axis=1)
        self.vel = np.zeros((self.n, 2))
        self.landmarks = np.concatenate((self.rng.uniform(0, self.height, self.n)[np.newaxis].T,
                                         self.rng.uniform(0, self.width, self.n)[np.newaxis].T), axis=1)
        obs_2 = []


        obs = np.concatenate((self.pos, self.vel), axis=1)
        state = obs.flatten()
        # print(state)
        return state, obs

    def set_scheme(self, scheme):
        self.scheme = scheme

    def set_logger(self, logdir):
        self.logdir = logdir
        self.cost_file = open("{}/cost.log".format(logdir), "w", 1)
        self.return_file = open("{}/return.log".format(logdir), "w", 1)
        self.peak_violation_file = open("{}/peak_violation.log".format(logdir), "w", 1)

    def init_cql(self):
        self.delta = -1

    def step(self, actions):

        # print(self._action_to_acc[actions])
        # update position and velocity
        del_vel = self._action_to_acc[actions]
        self.vel += del_vel
        self.pos += self.vel
        # prevent from going out of bounds
        self.pos[self.pos < 0] = 0
        self.pos[:, 0][self.pos[:, 0] > self.height] = self.height
        self.pos[:, 1][self.pos[:, 1] > self.width] = self.width

        # obs = np.concatenate((self.pos, self.vel), axis=1).flatten()
        obs = np.concatenate((self.pos, self.vel), axis=1)

        # collisions
        all_pair_dist = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)  # all pair distances
        np.fill_diagonal(all_pair_dist, np.inf)  # set diagonal to inf
        min_dist = np.min(all_pair_dist, axis=0)  # distance to nearest agent
        np.fill_diagonal(all_pair_dist, 0)  # set diagonal to 0
        collisions = min_dist < self.collision_threshold  # has collided with another agent

        # print(all_pair_dist)
        # print(min_dist)

        # compute reward
        landmark_dist = np.linalg.norm(self.pos - self.landmarks, axis=1)  # distance to respective landmark
        rewards = -landmark_dist  # closer the landmark, the higher the reward
        penalty = collisions * self.collision_penalty  # penalty for collision
        # cum_reward = np.sum((rewards - penalty) * self.agent_priority)

        # print("Cum_Reward ",cum_reward)
        # compute constraint reward
        constraint_rewards = min_dist
        # print(constraint_rewards)
        rew = rewards-penalty
        # r = np.array(r)
        # print(r)
        # cum_reward = np.concatenate(((rewards - penalty), constraint_rewards), axis=1)
        # global_reward = np.sum(cum_reward)
        global_reward = [np.sum(rew), np.sum(constraint_rewards)]
        local_rewards = [global_reward] * self.n_agents
        obses = obs
        state = obs.flatten()
        done_mask = True
        self.step_count = self.step_count + 1
        if self.step_count >= 64:
            done_mask = False
            self.step_count = 0
        # return obs, (rewards, constraint_rewards), False, {}
        # print("state ", state)
        # print("obses ", obses)
        # print("Local Reward ", local_rewards)
        # print("Global Reward ", global_reward)
        return state, obses, local_rewards, global_reward, done_mask

# env = CoopNavEnv(n=3, height=10, width=10, acceleration=0.5, agent_priority=[1, 1, 1], collision_threshold=0.5,
#                  collision_penalty=1)
# obs = env.reset()
# print(obs)
# print(env.step(np.array([0, 0, 0])))
# print(env.step(env.action_space.sample()))
# print(env.step(env.action_space.sample()))
# print(env.step(env.action_space.sample()))
