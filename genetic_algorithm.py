from math import ceil, log
import numpy as np
import random

class GeneticAlgorithm: 
    def __init__(self, func, coordinate_a, coordinate_b, initial_population, max_population, size_individual, probability_cross):
        self.func = func
        self.coordinate_a = coordinate_a
        self.coordinate_b = coordinate_b
        self.system_resolution = None
        self.initial_population = initial_population
        self.max_population = max_population
        self.size_individual = size_individual
        self.probability_cross = probability_cross
        
    def calculate_points_problem(self, rango: float, res_problem: float) -> int:
        return ceil(rango / res_problem) + 1

    def calculate_number_of_bits(self, problem_points: int) -> int:
        return ceil(log(problem_points, 2))

    def calculate_resolution_system(self, range: float, system_points: int) -> float:
        return round((range / (system_points - 1)), 4)

    def calculate_res_bits(self, range, res_problem) -> list[float, int]:
        if (res_problem <= 0):
            raise ValueError("The solution to the problem must be a positive floating-point number")

        if (range <= 0):
            raise ValueError("The range must be a positive number")

        problem_points = self.calculate_points_problem(range, res_problem)
        num_bits = self.calculate_number_of_bits(problem_points)
        system_points = 2 ** num_bits
        system_resolution = self.calculate_resolution_system(range, system_points)

        self.system_resolution = system_resolution

        return [system_resolution, num_bits]

    def calculate_phenotype(self, coordinate_a: int, idx: int, system_resolution: float) -> float:
        return coordinate_a + idx * system_resolution

    def convert_binary_to_integer(self, binary: list[int]):
        return int(''.join(map(str, binary)), 2)

    def create_population(self):
        population = []

        for _ in range(self.initial_population):         
            individual = []

            for _ in range(self.size_individual):      
                gene = random.randint(0, 1)
                individual.append(gene)

            population.append(individual)

        return population

    def evaluate_population(self, population: list[list[int]]):
        evaluated_population = []

        for individual in population:

            x = self.calculate_phenotype(
                self.coordinate_a, 
                self.convert_binary_to_integer(individual), 
                self.system_resolution)
            
            fitness = self.func(x)
            
            evaluated_population.append({
                "bit_string" : individual,
                "x" : x,
                "fitness" : fitness
            })

        return evaluated_population

    def generate_couples(self, population: list[list[object]]):
        couples = []

        for i in range(len(population)):
            for j in range(len(population)):
                 if i != j:
                     p = random.random()

                     if p <= self.probability_cross:
                       couple = (population[i], population[j])  
                       couples.append(couple)
                     

        return couples
        
    
             




    