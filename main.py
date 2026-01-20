from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np

def main():
    coordinate_a = 20
    coordinate_b = 22
    range_val = coordinate_b - coordinate_a
    res_problem = 0.25

    def func_ejemplo(x):
        return x**2
    
    algoritmo = GeneticAlgorithm(
        func=func_ejemplo,
        coordinate_a=coordinate_a,
        coordinate_b=coordinate_b,
        initial_population=10,
        max_population=100,
        size_individual=5,
        probability_cross=0.7
    )
    
    algoritmo.calculate_res_bits(range_val, res_problem)
    # print(f"System resolution: {result[0]}, Number of bits: {result[1]}")

    population = algoritmo.create_population()
    evaluated_pouplatuin = algoritmo.evaluate_population(population)
    print(algoritmo.generate_couples(evaluated_pouplatuin))

    # Grafica
    # x = np.linspace(20, 22, 50)

    # y = x * np.cos(7*x) + np.sin(3*x)

    # plt.figure(figsize=(10, 6))
    # plt.plot(x, y, label=r'$f(x) = x \cos(7x) + \sin(3x)$', color='blue')

    # plt.title(r'Gráfica de $f(x) = x \cos(7x) + \sin(3x)$ en [20, 22]')
    # plt.xlabel('Eje X')
    # plt.ylabel('Eje Y')
    # plt.grid(True, alpha=0.3)
    # plt.legend()

    # plt.show()

if __name__ == "__main__":
    main()
