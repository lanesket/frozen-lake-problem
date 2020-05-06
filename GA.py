import gym
import numpy as np


class GA:
    PROBLEM_NAME = "FrozenLake-v0"

    def __init__(self):
        self.env = gym.make(self.PROBLEM_NAME)

    def random_genome(self):
        """
            Creates a random genome (policy) for the environment
        """
        max_val = self.env.action_space.n
        size = self.env.observation_space.n
        return np.random.randint(0, max_val, size=size)

    def episode(self, genome: np.ndarray, max_episode_len=100) -> float:
        """
            Runs the episode once and return result for this genome
        """
        res = 0
        obs_pos = self.env.reset()
        for _ in range(max_episode_len):
            action = genome[obs_pos]
            obs_pos, reward, is_goal, info = self.env.step(action)
            res += reward
            if is_goal:
                break

        return res

    def fitness(self, genome: np.ndarray, n: int) -> float:
        """
            Calculates the average value for `n` played episodes
        """
        results = []
        for _ in range(n):
            results.append(self.episode(genome))

        return np.average(results)

    def select(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def evolution(self, start_population_size: int, generations: int):
        population = [self.random_genome()
                      for _ in range(start_population_size)]


if __name__ == "__main__":
    ga = GA()
    ga.evolution(50, 10)
