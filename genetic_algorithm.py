from math import ceil, log
import random

class GeneticAlgorithm: 
    def __init__(
            self, 
            func, 
            coordinate_a, 
            coordinate_b,
            res_problem, 
            initial_population, 
            max_population, 
            maximization,
            probability_cross, 
            individual_mutation, 
            mut_gen,
            generations
        ):
        self.func = func
        self.coordinate_a = coordinate_a
        self.coordinate_b = coordinate_b
        self.res_problem = res_problem
        self.system_resolution = None
        self.num_bits = None
        self.initial_population = initial_population
        self.max_population = max_population
        self.maximization = maximization
        self.probability_cross = probability_cross
        self.individual_mutation = individual_mutation
        self.mut_gen = mut_gen
        self.generations = generations
        
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
        self.num_bits = num_bits

        return [system_resolution, num_bits]

    def calculate_phenotype(self, coordinate_a: int, idx: int, system_resolution: float) -> float:
        return coordinate_a + idx * system_resolution

    def convert_binary_to_integer(self, binary: list[int]):
        return int(''.join(map(str, binary)), 2)

    def create_population(self):
        population = []

        for _ in range(self.initial_population):         
            individual = []

            for _ in range(self.num_bits):      
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
                "fitness" : round(fitness, 3)
            })

        return evaluated_population

    def generate_couples(self, population: list[list[object]]):
        couples = []

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                p = random.random()

                if p <= self.probability_cross:
                    couple = (population[i], population[j])  
                    couples.append(couple)
                     
        return couples
        
    def cross(self, parent1, parent2):
        pos = random.randint(1, self.num_bits)

        len_bits_parent =  len(parent1["bit_string"])

        children1 = parent1["bit_string"][0:pos] + parent2["bit_string"][pos:len_bits_parent]
        children2 = parent2["bit_string"][0:pos] + parent1["bit_string"][pos:len_bits_parent]

        return children1, children2
    
    def mutation(self, children):
        for child in children:
            if random.random() <= self.individual_mutation:
                for i in range(len(child)):
                    if random.random() <= self.mut_gen:
                        child[i] = 1 - child[i]

    def prune(self, population_generation: list[list[object]]):
        new_population = []
        
        new_population.append(population_generation.pop(0))

        while len(new_population) < self.max_population and len(population_generation) > 0:
            pos = random.randint(0, len(population_generation) - 1)
            new_population.append(population_generation.pop(pos))

        return new_population
        






    