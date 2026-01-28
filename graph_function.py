import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from genetic_algorithm import GeneticAlgorithm

def graph_aptitude_evolution(best_individuals_per_generation: list[object], worst_individuals_per_generation: list[object], average_per_generation: list[float]):
    fitness_values = [individual['fitness'] for individual in best_individuals_per_generation]
    worst_values = [individual['fitness'] for individual in worst_individuals_per_generation]

    generations = list(range(len(fitness_values)))

    plt.plot(generations, fitness_values, label='Best Fitness', color='blue', linewidth=2, marker='o', markevery=5)
    plt.plot(generations, worst_values, label='Worst Fitness', color='red', linestyle='--', alpha=0.7)
    plt.plot(generations, average_per_generation, label='Average Fitness', color='green', linewidth=2, alpha=0.8)

    plt.title('Evolution of Fitness Over Generations', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.show()

def graph_population_evolution(algorithm: GeneticAlgorithm, evaluated_population: list[object], generation_number: int, save_path: str = None):

    x_values = np.arange(algorithm.coordinate_a, algorithm.coordinate_b, algorithm.system_resolution)
    y_values = [algorithm.func(x) for x in x_values]
    
    # Configurar el estilo de la figura
    plt.figure(figsize=(10, 6), facecolor='#f5f5dc')
    ax = plt.gca()
    ax.set_facecolor('#f5f5dc')
    
    # Plot the function with fill effect
    plt.plot(x_values, y_values, color='#9999ff', linewidth=0.8, alpha=0.6, label='Función')
    plt.fill_between(x_values, y_values, alpha=0.4, color='#9999ff', edgecolor='none')
    
    best_individual = evaluated_population[0]
    worst_individual = evaluated_population[-1]
    others = evaluated_population[1:-1]

    x_best = [best_individual['x']]
    fitness_best = [best_individual['fitness']]

    x_worst = [worst_individual['x']]
    fitness_worst = [worst_individual['fitness']]

    x_others = [ind['x'] for ind in others]
    fitness_others = [ind['fitness'] for ind in others]

    plt.scatter(x_others, fitness_others, color='black', label='Población', alpha=0.7, s=50, zorder=5)
    plt.scatter(x_best, fitness_best, color='green', label='Mejor', s=200, marker='o', zorder=6, edgecolors='darkgreen', linewidth=2)
    plt.scatter(x_worst, fitness_worst, color='red', label='Peor', s=200, marker='o', zorder=6, edgecolors='darkred', linewidth=2)

    plt.title(f'Generación {generation_number}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('x', fontsize=13, fontweight='bold')
    plt.ylabel('f(x)', fontsize=13, fontweight='bold')
    
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    plt.grid(True, alpha=0.5, color='gray', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, facecolor='#f5f5dc')
    
    plt.close() 


def create_video_from_images(image_folder: str, output_video: str, fps: int = 2):
    """
    Crea un video a partir de las imágenes generadas
    
    Args:
        image_folder: Carpeta donde están las imágenes
        output_video: Nombre del archivo de video de salida
        fps: Frames por segundo (menor = más lento, mayor = más rápido)
             fps=1 -> 1 segundo por frame
             fps=2 -> 0.5 segundos por frame
             fps=0.5 -> 2 segundos por frame
    """
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    
    if not images:
        print("No se encontraron imágenes en la carpeta")
        return
    
    # Leer la primera imagen para obtener dimensiones
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Configurar el video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Creando video con {len(images)} frames a {fps} fps...")
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    video.release()
    print(f"Video creado exitosamente: {output_video}")