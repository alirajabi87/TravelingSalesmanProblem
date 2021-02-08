from deap import tools, algorithms
import numpy as np


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    """
    We use this algorithm instead of eaSimple of deap to pick the best individual
    and insert it to the next generation.
    Augments:
    :param population: Population for each generation
    :param toolbox: base.toolbox
    :param cxpb: probability of crossover
    :param mutpb: probability of mutation
    :param ngen: number of generation
    :param stats: Statistics
    :param halloffame: Hall of Fame to peak the beast individual
    :param verbose: showing the results (Default = True)
    :return: Population, logbook
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("Hall of Fame parameter must not be empty!!!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    current_mean = logbook.select("MEAN")[0]

    for gen in range(1, ngen + 1):

        # Select the next generation
        offSpring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        # print(f"mutpb: {mutpb}")
        offSpring = algorithms.varAnd(offSpring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offSpring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best to the generation

        offSpring.extend(halloffame.items)

        # Update Hall of Fame
        halloffame.update(offSpring)

        # Replace the corrent population with offSpring
        population[:] = offSpring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # if current_mean - logbook.select("MEAN")[gen] < 0:
        #     temp = np.random.random()
        #     mutpb = temp if temp < 0.3 else 0.15
        #     toolbox.register("select", tools.selTournament, tournsize=4)
        #     # print(f"The Minimum is {logbook.select('MEAN')[gen]} and previous Min is {current_mean}, mutpb is: {mutpb}")
        # else:
        #     current_mean = logbook.select("MEAN")[gen]
        #     toolbox.register("select", tools.selTournament, tournsize=2)
        #     # print(f"Return to previous settings")
        #     mutpb = 0.1

    return population, logbook
