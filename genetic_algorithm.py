import numpy as np

'''
pool = {
    'time_lag': [0, 1, 2, 3, 4, 5],
    'layer_1_units': [1, 4, 8, 12, 16], # NOTE: layer 1 cannot be 0
    'layer_2_units': [0, 1, 4, 8, 12, 16],
    'layer_3_units': [0, 1, 4, 8, 12, 16]
}

member_data_model = {
    'genes': {
        'feature_al': [0, 1],
        'feature_By': [0, 1],
        'time_lag': 2,
        'layer_1_units': 4, # NOTE: layer 1 cannot be 0
        'layer_2_units': 12,
        'layer_3_units': 0 # NOTE: 0 stands for no layer here
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

# TODO: determine if should return 2 offspring, like in paper
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

# TODO: need better name here
# TODO: determine if should return 2 offspring, like in paper
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
    pop_loss = 0
    for m in population:
        pop_loss += m['val_loss'] 
    for m in population:
        m['fitness'] = m['val_loss'] / pop_loss


def rank_val_loss(population):
    losses = np.array([m['val_loss'] for m in population])
    # TODO: need to look at how argsort works, not working currently
    #ranks = np.argsort(losses)
    # NOTE: should work for now
    s = np.sort(losses)
    ranks = np.array([np.argwhere(s == losses[i])[0][0] for i in range(0, len(losses))])

    for i in range(0, len(population)):
        population[i]['rank'] = ranks[i] + 1

def fitness_rank(population):
    n = len(population)
    total = 0
    for m in population:
        total += n + 1 - m['rank']
    for m in population:
        m['fitness'] = (n + 1 - m['rank']) / total

def member_selection_rank(population):
    return [m['fitness'] for m in population]

def member_selection_prob_val_loss(population):
    fitness_vals = np.array([m['fitness'] for m in population])
    t = 1 / fitness_vals
    s = np.sum(t)
    return t / s

def get_member_with_rank(rank, population):
    for m in population:
        if m['rank'] == rank:
            return m 

def genetic_algorithm(pool, size=20, generations=3, mutation_prob=.1, train_epochs=5, fill_member=None):
    
    # create initial population
    init_pop = [generate_random_member(pool) for _ in range(size)]
    for member in init_pop:
        fill_member(member, train_epochs)
    # need all val_loss filled in before calculating fitness vals
    rank_val_loss(init_pop)
    #fitness_val_loss(init_pop)
    fitness_rank(init_pop)
    history = [
        {
            'val_loss': [m['val_loss'] for m in init_pop],
            'fitness': [m['fitness'] for m in init_pop]
        }
    ]
    print('Initial Population Generated')
    
    current_pop = init_pop
    for g in range(1, generations+1):
        #sel_prob = member_selection_prob_val_loss(current_pop)
        sel_prob = member_selection_rank(current_pop)
        new_pop = []
        for _ in range(size):
            # TODO: need to make sure this does a good job of getting to good params, look at ieee
            parents = np.random.choice(current_pop, size=2, replace=False, p=sel_prob)
            m1, m2 = parents[0], parents[1]
            new_pop.append(crossover_binary(m1, m2, pool, mutate_chance=mutation_prob))
        for member in new_pop:
            fill_member(member, train_epochs)
        rank_val_loss(new_pop)
        #if ((generations+1) / 2) < g:
        #fitness_val_loss(new_pop)
        #else:
        fitness_rank(new_pop)
            
        current_pop = new_pop

        history.append(
            {
                'val_loss': [m['val_loss'] for m in current_pop],
                'fitness': [m['fitness'] for m in current_pop]
            }
        )

        print('Generation %s complete' % g)
        
    # at end, return history of training and final pop
    #rank_val_loss(current_pop)
    
    return history, get_member_with_rank(1, current_pop)