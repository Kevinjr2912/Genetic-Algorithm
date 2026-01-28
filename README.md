# Genetic-Algorithm

# import numpy as np
# import matplotlib.pyplot as plt
# from genetic_algorithm import Genetic
#     return x * np.cos(7*x) + np.sin(3*x)

# def main():
    
#     algorithm = GeneticAlgorithm(
#         func=func,
#         coordinate_a=20,
#         coordinate_b=22,
#         res_problem=0.30,
#         initial_population=12,
#         max_population=40,
#         size_individual=6,
#         maximization=True,
#         probability_cross=0.25,
#         individual_mutation=0.25,
#         mut_gen=0.25
#     )
    
#     algorithm.calculate_res_bits(
#         algorithm.coordinate_b - algorithm.coordinate_a,
#         algorithm.res_problem
#     )

#     Crear población inicial
#     population = algorithm.create_population()
#     evaluated_population = algorithm.evaluate_population(population)

#     generations = 30
#     best_fitness_per_generation = []

#     for gen in range(generations):

#         Guardamos el mejor fitness
#         best = max(
#             evaluated_population,
#             key=lambda ind: ind["fitness"]
#         )
        
#         best_fitness_per_generation.append(best["fitness"])

#         Cruza
#         couples = algorithm.generate_couples(evaluated_population)
#         children = []

#         for p1, p2 in couples:
#             c1, c2 = algorithm.cross(p1, p2)
#             children.append(c1)
#             children.append(c2)

#         Mutación
#         algorithm.mutation(children)

#         Poda => nueva generación
#         evaluated_population = algorithm.prune(
#             evaluated_population,
#             children
#         )

#     plt.plot(best_fitness_per_generation, marker='o')
#     plt.xlabel("Generación")
#     plt.ylabel("Mejor fitness")
#     plt.title("Evolución del fitness - Algoritmo Genético")
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     main()

# def prune(self, parents: list[list[object]], children: list[list[int]]):
#     population = parents + self.evaluate_population(children)

#     # Ordenamos la poda
#     population.sort(key=lambda individual: individual['fitness'], reverse=self.maximization)

#     # Tomamos el mejor, y la sacamos de la lista
#     best_individual = population.pop(0)

#     new_population = []
    
#     new_population.append(best_individual)
    
#     while len(new_population) < self.max_population and len(population_generation) > 0:
#         pos = random.randint(0, len(population_generation) - 1)
#         new_population.append(population_generation.pop(pos))

#     return new_population
# Genetic-Algorithm
