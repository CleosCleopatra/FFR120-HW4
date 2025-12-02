import numpy as np 
import matplotlib.pyplot as plt
    
def prisoner_dilemma(Nr, T, R, P, S, na, nb):
    """
    Function to simulate the outcome of a prisoner's dilemma game 
    over Nr rounds.
    
    Parameters
    ==========
    Nr : Total number of rounds.
    T : Punishment for defecting prisoner (partner non-defecting).
    R : Punishment for both prisoners, both non defecting.
    P : Punishment for both prisoners, both defecting.
    S : Punishment for non-defecting prisoner (partner defecting).
    na : strategy of player A
    nb : strategy of player B
    """
    
    n = np.min([na, nb])
    m = np.max([na, nb])

    if n == m:
        punishment = R * n + P * (Nr - n)
        punishment_A = punishment
        punishment_B = punishment
    else:
        if n > 0:
            punishment_defector = n * R + T + P * (Nr - n - 1)
            punishment_partner = n * R + S + P * (Nr - n - 1)
        else:
            punishment_defector = T + P * (Nr - 1)
            punishment_partner = S + P * (Nr - 1)
        if na > nb:
            punishment_A = punishment_partner
            punishment_B = punishment_defector
        else:
            punishment_B = punishment_partner
            punishment_A = punishment_defector
    
    return punishment_A, punishment_B



def lattice_turn(L, Nr, T, R, P, S, mu):
    """
    Function to simulate the outcome of a prisoner's dilemma game 
    over Nr rounds on a lattice (i.e., over one turn).
    
    Parameters
    ==========
    L : Lattice. Contains the current strategies.
    Nr : Total number of rounds.
    T : Punishment for defecting prisoner (partner non-defecting).
    R : Punishment for both prisoners, both non defecting.
    P : Punishment for both prisoners, both defecting.
    S : Punishment for non-defecting prisoner (partner defecting).
    """
    
    Lx, Ly = L.shape
    game_scores = np.zeros([Lx, Ly])  # Saves the scores each round.
        
    Ln = np.zeros([Lx, Ly])  # New strategies.
    
    # Competition.
    for i in range(Lx):
        for j in range(Ly):
            # Game with neighbour below.
            pa, pb = prisoner_dilemma(Nr, T, R, P, S, 
                                      int(L[i, j]), 
                                      int(L[i, (j + 1) % Ly]))
            game_scores[i, j] += pa
            game_scores[i, (j + 1) % Ly] += pb
            
            # Game with neighbour to the right.
            pa, pb = prisoner_dilemma(Nr, T, R, P, S, 
                                      int(L[i, j]), 
                                      int(L[(i + 1) % Lx, j]))
            game_scores[i, j] += pa
            game_scores[(i + 1) % Lx, j] += pb

    # Revision.
    for i in range(Lx):
        for j in range(Ly):
            # Check the neighbors' scores.
            best_strategy = np.argsort([game_scores[i, j], 
                                        game_scores[i, (j + 1) % Ly],
                                        game_scores[(i + 1) % Lx, j],
                                        game_scores[i, (j - 1) % Ly],
                                        game_scores[(i - 1) % Lx, j]])
            
            if best_strategy[0] == 0:
                Ln[i, j] = L[i, j]
            elif best_strategy[0] == 1:
                Ln[i, j] = L[i, (j + 1) % Ly]
            elif best_strategy[0] == 2:
                Ln[i, j] = L[(i + 1) % Lx, j]
            elif best_strategy[0] == 3:
                Ln[i, j] = L[i, (j - 1) % Ly]
            else: # best_strategy[0] == 4:
                Ln[i, j] = L[(i - 1) % Lx, j]
    
    # Mutation.
    if mu > 0:
        p = np.random.rand(Lx, Ly)
        mutants = np.where(p < mu)
        for i in range(len(mutants)):
            Ln[mutants[0][i], mutants[1][i]] = np.random.randint(Nr + 1)
    
    return Ln


def lattice_turn_mod(L, Nr, T, R, P, S, mu):
    """
    Modified prisoners' dilemma revision step.
    
    Revision: each player compares with neighbors, picks the one with the LOWEST score (least punishment).
    If n_self >= n_best, adopt n_best.
    If n_self < n_best, adopt a random integer between n_self+1 and N_rounds.
    
    Parameters
    ==========
    L : Lattice. Contains the current strategies.
    Nr : Total number of rounds.
    T, R, P, S : Game parameters.
    mu : Mutation rate.
    """
    
    Lx, Ly = L.shape
    game_scores = np.zeros([Lx, Ly])  # Saves the scores each round.
    Ln = np.zeros([Lx, Ly])  # New strategies.
    
    # Competition (same as classical).
    for i in range(Lx):
        for j in range(Ly):
            # Game with neighbour below.
            pa, pb = prisoner_dilemma(Nr, T, R, P, S, 
                                      int(L[i, j]), 
                                      int(L[i, (j + 1) % Ly]))
            game_scores[i, j] += pa
            game_scores[i, (j + 1) % Ly] += pb
            
            # Game with neighbour to the right.
            pa, pb = prisoner_dilemma(Nr, T, R, P, S, 
                                      int(L[i, j]), 
                                      int(L[(i + 1) % Lx, j]))
            game_scores[i, j] += pa
            game_scores[(i + 1) % Lx, j] += pb

    # Modified Revision.
    for i in range(Lx):
        for j in range(Ly):
            # Get scores of self and four neighbors (von Neumann neighborhood).
            n_self = int(L[i, j])
            best_strategy = np.argsort([game_scores[i, j], 
                                        game_scores[i, (j + 1) % Ly],
                                        game_scores[(i + 1) % Lx, j],
                                        game_scores[i, (j - 1) % Ly],
                                        game_scores[(i - 1) % Lx, j]])
            
            n_best = best_strategy[0]
            n_self = int(L[i, j])
            
            if n_self >= n_best:
                Ln[i, j] = n_best
            else:  # n_self < n_best
                if n_self < Nr:
                    Ln[i, j] = np.random.randint(n_self + 1, Nr + 1)
                else:
                    Ln[i, j] = n_self
    
    # Mutation (same as classical).
    if mu > 0:
        p = np.random.rand(Lx, Ly)
        mutants = np.where(p < mu)
        for i in range(len(mutants)):
            Ln[mutants[0][i], mutants[1][i]] = np.random.randint(Nr + 1)
    
    return Ln


def count_strategies(L, Nr):
    """
    Function to calculate the number of players with a given
    stategy of prisoner's dilemma game 
    on a lattice.
    
    Parameters
    ==========
    L : Lattice. Contains the current strategies.
    Nr : Total number of rounds.
    """
    
    stat_players = np.zeros(Nr + 1)  # Contains the statistics.
    
    for i in range(Nr + 1):
        stat_players[i] = np.sum(L == i)

    return stat_players


L_side = 50  # Size of the lattice with the players.
mu = 0.00  # Mutation rate. No mutation.
Nr = 10  # Number of rounds in a Prisoner's dilemma's game.

T = 0  # Punishment for defecting prisoner (partner non-defecting)
P = 1  # Punishment for both prisoners, both defecting.
S = 1.8  # Punishment for non-defecting prisoner (partner defecting)
# S: Always 1 < S < 2

# R: Always 0 < R < 1. Tries different values.
R = 0.65  # Punishment for both prisoners, both non-defecting.

T_max = 300

# Initialize random number generator.
np.random.seed(42)


game_scores = np.zeros([L_side, L_side])  # Saves the scores each round.

N_rep = 8

C_all_mod = []
fig_mod, axes_mod = plt.subplots(2, N_rep, figsize=(3*N_rep, 8))
for rep in range(N_rep):
    print(f"Part 2 - rep {rep}")
    # Initialize the strategies at random, 0 and Nr and values in between.
    L = np.random.randint(Nr + 1, size=(L_side, L_side)) 
    L_init = L.copy()
    nstrategies_mod = np.zeros([Nr + 1, T_max])
    C_t_mod = np.zeros(T_max)
    for r in range(T_max):
        L = lattice_turn_mod(L, Nr, T, R, P, S, mu)
        nstrategies_mod[:, r] = count_strategies(L, Nr)
        C_t_mod[r] = np.sum(np.arange(Nr + 1) * nstrategies_mod[:, r]) / (L_side * L_side)
    C_all_mod.append(C_t_mod)
    
    im0_mod = axes_mod[0, rep].imshow(L_init, cmap='viridis', vmin=0, vmax=Nr)
    fig_mod.colorbar(im0_mod, ax=axes_mod[0, rep], label='Strategy')
    axes_mod[0, rep].set_title('Initial strategies')

    # Final
    im1_mod = axes_mod[1, rep].imshow(L, cmap='viridis', vmin=0, vmax=Nr)
    fig_mod.colorbar(im1_mod, ax=axes_mod[1, rep], label='Strategy')
    axes_mod[1, rep].set_title('Final strategies')
plt.tight_layout()
plt.show()


cell_color_list = ['#8F0000', '#FF7F00', '#FFC000', '#00FF00', 
                   '#007F00', '#0080FF', '#000080', '#7F00FF',  
                   '#8888FF', '#808080', '#000000'  ]


# Plot average strategy C_mod(t) for all repetitions
plt.figure(figsize=(8, 6))
for re in range(N_rep):
    plt.plot(C_all_mod[re], color=cell_color_list[re % len(cell_color_list)], label=f'rep {re+1}')
    
plt.xlabel("t")
plt.ylabel("Average strategy C_mod(t)")
plt.title("Average strategies (modified)")
plt.legend()
plt.show()