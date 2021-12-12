import numpy as np
from tqdm.auto import trange

# Data structures

'''
pool  of features/hyperparameters
pool = {
    'feature': [0, 1], # NOTE: 0 => off, 1 => on
    'time_lag': [0, 1, 2, 3, 4, 5],
    'layer_1_units': [1, 4, 8, 12, 16],
    'layer_2_units': [0, 1, 4, 8, 12, 16],
    'layer_3_units': [0, 1, 4, 8, 12, 16]
}

'''

'''
member of a population
member_data_model = {
    'genes': {
        values from keys of pool
        'X': 1,
        'Y': 2,
        ...
    },
    'model': None, # NOTE: actual tf model,
    'X_scaler': fn,
    'Y_scaler': fn,
    'train': {
        'X': [],
        'Y': []
    },
    'valid': {
        'X': [],
        'Y': []
    },
    'test': {
        'X': [],
        'Y': []
    },
    'fitness': float, 
    'train_loss': float,
    'valid_loss': float,
    'rank': int
}
'''

def generate_random_member(pool):
    """
    Generates random member based on a pool of hyper-parameters
    """
    return {
        'genes': {
            k: np.random.choice(pool[k]) for k in pool.keys()
        }
    }

def mutate(member, pool):
    """
    Mutates a specific gene in a member, 
    sampled randomly from the gene pool
    """
    gene = np.random.choice(list(pool.keys()))
    member['genes'][gene] = np.random.choice(pool[gene])

def crossover_binary(m1, m2, pool, mutate_chance=.1):
    """
    Binary crossover of alleles from either parent (50/50 from each parent)
    """
    keys = list(pool.keys())
    choices = np.random.choice([0, 1], len(keys))
    offspring = {
        'genes': {}
    }
    for i in range(0, len(keys)):
        if choices[i] == 0:
            offspring['genes'][keys[i]] = m1['genes'][keys[i]]
        else:
            offspring['genes'][keys[i]] = m2['genes'][keys[i]]
    if (np.random.choice([True, False], p=[mutate_chance, 1-mutate_chance])):
        mutate(offspring, pool)
    return offspring

def generate_offspring(m1, m2, pool, mutate_chance=.1):
    """
    Generates a child based on 2 parent members of a population
    """
    split = np.random.choice([i for i in range(0, len(pool))])
    m1_genes = list(pool.keys())[0:split]
    m2_genes = list(pool.keys())[split:]
    offspring = {
        'genes': {}
    }
    for k in m1_genes:
        offspring['genes'][k] = m1['genes'][k]
    for k in m2_genes:
        offspring['genes'][k] = m2['genes'][k]
    if (np.random.choice([True, False], p=[mutate_chance, 1-mutate_chance])):
        mutate(offspring, pool)
    return offspring

def fitness_val_loss(population):
    """
    Fitness value based on weighted validation loss against population 
    """
    pop_loss = 0
    for m in population:
        pop_loss += m['val_loss'] 
    for m in population:
        m['fitness'] = m['val_loss'] / pop_loss


def rank_val_loss(population):
    """
    Function to rank each member in a population based on validation loss (low valid loss => high rank)
    """
    losses = np.array([m['val_loss'] for m in population])
    s = np.sort(losses)
    ranks = np.array([np.argwhere(s == losses[i])[0][0] for i in range(0, len(losses))])

    for i in range(0, len(population)):
        population[i]['rank'] = ranks[i] + 1

def fitness_rank(population):
    """
    Fitness function based on rank weighted by rank against population 
    """
    n = len(population)
    total = 0
    for m in population:
        total += n + 1 - m['rank']
    for m in population:
        m['fitness'] = (n + 1 - m['rank']) / total

def member_selection_rank(population):
    """
    Computes probability of selection based on fitness using rank
    """
    return [m['fitness'] for m in population]

def member_selection_prob_val_loss(population):
    """
    Computes probability of selection based on fitness using valid loss
    """
    fitness_vals = np.array([m['fitness'] for m in population])
    t = 1 / fitness_vals
    s = np.sum(t)
    return t / s

def get_member_with_rank(rank, population):
    """
    Grabs member with 'rank' from population
    """
    for m in population:
        if m['rank'] == rank:
            return m 

def genetic_algorithm(pool, size=20, generations=3, mutation_prob=.1, train_epochs=5, fill_member=None):
    """
    Genetic algorithm for generating a successive models based on a given pool of genes to choose from 
    """
    
    # create initial population
    init_pop = [generate_random_member(pool) for _ in range(size)]
    print('Generating Initial Population')
    for i in trange(len(init_pop)):
        member = init_pop[i]
        fill_member(member, train_epochs)
    rank_val_loss(init_pop)
    fitness_rank(init_pop)
    history = [
        {
            'val_loss': [m['val_loss'] for m in init_pop],
            'fitness': [m['fitness'] for m in init_pop]
        }
    ]
    print('Initial Population Generated')
    
    # generate x generations
    current_pop = init_pop
    for g in range(1, generations+1):
        print('Generating Population %s' % g)
        sel_prob = member_selection_rank(current_pop)
        new_pop = []
        for _ in trange(size):
            parents = np.random.choice(current_pop, size=2, replace=False, p=sel_prob)
            m1, m2 = parents[0], parents[1]
            new_pop.append(crossover_binary(m1, m2, pool, mutate_chance=mutation_prob))
        for member in new_pop:
            fill_member(member, train_epochs)
        rank_val_loss(new_pop)
        fitness_rank(new_pop)
            
        current_pop = new_pop

        history.append(
            {
                'val_loss': [m['val_loss'] for m in current_pop],
                'fitness': [m['fitness'] for m in current_pop]
            }
        )

        print('Generation %s complete' % g)

    # return results after x generations
    return history, get_member_with_rank(1, current_pop)