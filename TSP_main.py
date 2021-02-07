from TravelingSalesmanProblem import TSP
from deap import tools, base, algorithms, creator
import array, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TSP_NAME = "bayg29"
tsp = TSP(TSP_NAME)

# Genetic Constants
MAX_GENERATION = 300
POPULATION = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.1
HALL_OF_FAME_NUMBER = 5


# Define the fitness strategy
creator.create("FitnessMin", base.Fitness, weights=(-1.,))

# Creating the chromosome

creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))
toolbox.register("IndividualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.IndividualCreator)


def tspDistance(individual):
    return (tsp.getTotalDistance(individual),)

toolbox.register("evaluate", tspDistance)

# register three genetic operators : Select, mate, mutate

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1./len(tsp))

def main():
    population = toolbox.populationCreator(n=POPULATION)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("MIN", np.min)
    stats.register("MEAN", np.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_NUMBER)

    population, logbook = algorithms.eaSimple(population=population,
                                              stats=stats,
                                              toolbox=toolbox,
                                              halloffame=hof,
                                              ngen=MAX_GENERATION,
                                              mutpb=P_MUTATION,
                                              cxpb=P_CROSSOVER,
                                              verbose=True,)
    maxFitnessValues, meanFitnessValues = logbook.select("MIN", "MEAN")
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.title("Traveling Salesman Problem")
    plt.ylabel("Max/Mean Values")
    plt.xlabel("Generations")
    plt.legend(["MAX", "MEAN"])
    plt.show()

    print(f"--Best ever individual: {hof.items[0]}")
    print(f"--Best ever fitness: {hof.items[0].fitness.values[0]}")
    plot = tsp.plotData(hof.items[0])
    plot.show()


if __name__ == '__main__':
    main()

