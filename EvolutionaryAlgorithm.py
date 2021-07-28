import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, num_individuals, genomes, mutation_rate):
        self.num_individuals = num_individuals
        self.mutation_rate = mutation_rate
        self.genomes = genomes
        self.evals = []

    def evolve(self,evals):
        self.evals = evals
        self.select_populate(self.tournament_selection)
        self.crossover_mutate_single(self.mutation_rate)
        return self.genomes


    def select_populate(self, func):
        # Get indices from sorted eval list (ascending)
        i = np.array(self.evals).argsort()[::-1]
        self.evals.sort(reverse = True)

        # Apply eval sorting to genome list
        self.genomes = func(self.genomes[i])

    def crossover_mutate(self, mutation_rate):
        genome_amount = self.num_individuals
        genome_length = len(self.genomes[0])

        for genome in self.genomes:
            parent_gene = random.randrange(0, genome_amount)

            begin = random.randrange(0, genome_length)
            end = random.randrange(0, genome_length)
            if begin > end:
                begin, end = end, begin

            for i in range(begin, end):
                genome[i] = self.genomes[parent_gene][i]

        print("###############")

        for genome in self.genomes:
            for i in range(genome_length):
                if random.random() * 100 <= mutation_rate:
                    genome[i] = abs(genome[i] - 1)

    def crossover_mutate_single(self, mutation_rate):
        # Match parents so that no parent is matched twice or whith itself
        sample = np.arange(self.num_individuals)
        parent1_index = random.sample(list(sample), self.num_individuals // 2)
        sample = np.delete(sample, parent1_index)
        parent2_index = random.sample(list(sample), self.num_individuals // 2)

        children = []

        for i in range(self.num_individuals // 2):
            parent1 = self.genomes[parent1_index[i]]
            parent2 = self.genomes[parent2_index[i]]
            genome_length = len(self.genomes[0])
            begin = random.randrange(genome_length)

            child1 = []
            child2 = []

            child1.extend(parent1[begin:])
            child1.extend(parent2[:begin])
            child2.extend(parent2[begin:])
            child2.extend(parent1[:begin])

            children.append(child1)
            children.append(child2)

        self.genoms = children

        for genome in self.genomes:
            for i in range(genome_length):
                if random.random() * 100 <= mutation_rate:
                    print("mutate")
                    genome[i] = abs(genome[i] - 1)

        print("###############")

    ########## Different Selection Methods ##########
    def truncated_rank(self, sorted_genomes):
        num_selected = 3

        repeat = self.num_individuals // num_selected
        selected_genomes = np.repeat(sorted_genomes[:num_selected], repeat, 0)

        # rest after repetition
        add_repeat = self.num_individuals % num_selected
        for i in range(add_repeat):
            selected_genomes = np.append(selected_genomes, [sorted_genomes[i]], 0)

        return selected_genomes

    def proportional_selection(self, sorted_genomes):
        # Inverse as lower numbers are better
        inverse_evals = [1 / x for x in self.evals]
        proportional_evals = inverse_evals / sum(inverse_evals)
        # Randomly sample indices from the proportional probs
        i = np.random.choice(sorted_genomes.shape[0], self.num_individuals, p=proportional_evals)
        return sorted_genomes[i]

    def rank_based_selection(self, sorted_genomes):
        # Slightly different implementation
        inv_ranks = np.flip(np.array(range(1, self.num_individuals + 1)))
        proportional_probs = inv_ranks / sum(inv_ranks)
        i = np.random.choice(sorted_genomes.shape[0], self.num_individuals, p=proportional_probs)
        return sorted_genomes[i]

    def tournament_selection(self, sorted_genomes):
        k = 7
        winning_indices = []
        for i in range(self.num_individuals):
            # Pick random participants (indices)
            tournament_indices = random.sample(range(len(sorted_genomes)), k)
            # Select winning indice by highest eval
            winning_indices.append(tournament_indices[np.array(self.evals)[tournament_indices].argmax()])
        return sorted_genomes[winning_indices]
#################################################