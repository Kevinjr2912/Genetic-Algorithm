from genetic_algorithm import GeneticAlgorithm
from graph_function import graph_aptitude_evolution, graph_population_evolution, create_video_from_images
import numpy as np
import os
import shutil

def func(x):
    return np.sin(x**2 + 3 * x) * np.cos(2*x) + 0.45 + x

def main():
    
    algorithm = GeneticAlgorithm(
        func=func,
        coordinate_a=-1000,
        coordinate_b=800,
        res_problem=0.02,  
        initial_population=5,
        max_population=20,
        maximization=True,  
        probability_cross=0.7,
        individual_mutation=0.12,
        mut_gen=0.20,
        generations=20
    )
    
    result = algorithm.calculate_res_bits((algorithm.coordinate_b - algorithm.coordinate_a), algorithm.res_problem)
    print(f"System resolution: {result[0]}, Number of bits: {result[1]}")

    # Crear carpeta para guardar los frames
    frames_folder = "frames"
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)

    # 1. Creamos la población inicial (soluciones aleatorias)
    population = algorithm.create_population()        

    # 2. Evaluamos dicha población para obtener de cada individuo qué tan bueno es (fitness) 
    evaluated_population = algorithm.evaluate_population(population)

    best_individuals_per_generation = []
    worst_individuals_per_generation = []
    average_per_generation = []

    for generation in range(algorithm.generations):
        print(len(evaluated_population))
        # 3. Generamos parejas (padres) 
        cuples = algorithm.generate_couples(evaluated_population)

        children = []
        
        # 4. Mezclamos la información genética de dos padres (cruza) para crear nuevos individuos
        for parent1, parent2 in cuples:
            children.append(algorithm.cross(parent1, parent2))

        children = [lis for tuple in children for lis in tuple]

        # 5. Mutar x individuos aleatoriamente
        algorithm.mutation(children)

        # Organizar la población actual + hijos y evaluar
        population_generation = evaluated_population + algorithm.evaluate_population(children)

        # 6. Ordenar la población por aptitud, y obtener mejores, peores y promedio
        population_generation.sort(key=lambda individual: individual["fitness"], reverse=algorithm.maximization)

        best_individual = population_generation[0]
        worst_individual = population_generation[len(population_generation)-1]
        average = sum(individual["fitness"] for individual in population_generation) / len(population_generation)

        best_individuals_per_generation.append(best_individual)
        worst_individuals_per_generation.append(worst_individual)
        average_per_generation.append(average)

        # 7. Podar la población (quiénes se quedan en la siguiente generación)
        evaluated_population = algorithm.prune(population_generation)

        # Guardar la gráfica como imagen
        frame_path = os.path.join(frames_folder, f"frame_{generation:03d}.png")
        graph_population_evolution(algorithm, evaluated_population, generation + 1, save_path=frame_path)
        print(f"Generación {generation + 1} guardada")


    # Crear el video
    create_video_from_images(
        image_folder=frames_folder,
        output_video="evolucion_genetica.mp4",
        fps=1
    )

    # Parte del proceso para graficar la evolución de la aptitud
    graph_aptitude_evolution(
        best_individuals_per_generation, 
        worst_individuals_per_generation,
        average_per_generation
    )

if __name__ == "__main__":
    main()

