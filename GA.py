from typing import List
import gym
import numpy as np
import heapq
import random


class GA:
    PROBLEM_NAME = "FrozenLake-v0"

    def __init__(self):
        self.env = gym.make(self.PROBLEM_NAME)
        self.max_val = self.env.action_space.n
        self.size = self.env.observation_space.n

    def random_genome(self) -> np.ndarray:
        """
            Creates a random genome (policy) for the environment
        """

        return np.random.randint(0, self.max_val, size=self.size)

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

    def fitness(self, genome: np.ndarray, n=10) -> float:
        """
            Calculates the average value for `n` played episodes
        """
        results = []
        for _ in range(n):
            results.append(self.episode(genome))

        return np.average(results)

    def rank_select(self, population: List, scores: List, k=2) -> List:
        """
            Rank Selection for `population` with their `scores`
        """
        scores = np.array(scores)
        top_indexes = heapq.nlargest(k, range(len(scores)), scores.take)

        return [population[i] for i in top_indexes]

    def select(self, population: List, k: int):
        """
            Selects by rank and copy the individuals
        """
        scores = [self.fitness(g) for g in population]
        selected = self.rank_select(population, scores, k)

        return [s.copy() for s in selected]

    def survive_select(self, population: List, k: int):
        """
            Survive select at the end of the iteration
            Also returns the avg value for correcting the mutation probability 
        """
        scores = [self.fitness(g) for g in population]
        selected = self.rank_select(population, scores, k)

        max = np.amax(scores)
        min = np.amin(scores)
        avg = np.average(scores)

        print(f"avg: {avg}, max: {max}, min: {min}")

        return selected, avg

    def crossover(self, offsprings: List) -> List:
        """
            One-point Crossover
        """
        count = len(offsprings) if len(
            offsprings) % 2 == 0 else len(offsprings) - 1
        offspring_indexes = random.sample(range(0, len(offsprings)), count)
        it = iter(offspring_indexes)
        crossover_point = random.randint(1, self.size)

        k = 0
        for i in it:
            k += 1
            j = next(it)
            child_1 = np.hstack((offsprings[i][0:crossover_point],
                                 offsprings[j][crossover_point:]))

            child_2 = np.hstack((offsprings[j][0:crossover_point],
                                 offsprings[i][crossover_point:]))

            offsprings[i] = child_1
            offsprings[j] = child_2

        return offsprings

    def mutate(self, offsprings: List, mutation_prob: int) -> List:
        """
            Bit-flip mutation with `mutation_prob`
        """
        offsprings = np.array(offsprings)
        rand_mutation = np.random.random(size=offsprings.shape)

        random_mutation_boolean = rand_mutation <= mutation_prob

        offsprings[random_mutation_boolean] = np.logical_not(
            offsprings[random_mutation_boolean])

        return list(offsprings)

    def terminate(self, population: List) -> np.ndarray:
        """
            Return the best individual
        """
        scores = np.array([self.fitness(g) for g in population])
        print(f"The average score of the best individual: {np.amax(scores)}")

        return population[scores.argmax()]

    def evolution(self, start_population_size: int, generations: int) -> None:
        """
            Genetic Algorithm in the flesh :D
        """
        population = [self.random_genome()
                      for _ in range(start_population_size)]
        k = int(start_population_size / 2)
        mutation_prob = 0.1

        for i in range(generations):
            print(f"Generation: {i + 1}")

            offsprings = self.select(population, k)

            offsprings = self.crossover(offsprings)

            offsprings = self.mutate(offsprings, mutation_prob)

            population.extend(offsprings)

            population, avg = self.survive_select(population, k)

            mutation_prob = (-1.1 * avg + 1) / 10

            print("____________\n")

        best = self.terminate(population)


if __name__ == "__main__":
    ga = GA()
    ga.evolution(150, 30)
