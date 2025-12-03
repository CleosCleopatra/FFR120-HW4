#4ab
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
    
def check_family_happiness(L, i, j, f_types):
    """
    Function to check whether the family in i, j is happy.
    
    Parameters
    ==========
    L : Lattice representing the town. 
    i, j : Location of the family to be checked.
    f_types : List containing the family types. 
              Ex: [1, - 1] for the types of families considered above.
    """
    
    
    Nx, Ny = L.shape
    
    ci = np.array([(i - 1) % Nx, (i - 1) % Nx, (i - 1) % Nx, 
                    i, i, i,
                    (i + 1) % Nx, (i + 1) % Nx, (i + 1) % Nx]).astype(int)
    cj = np.array([(j - 1) % Ny, j, (j + 1) % Ny,
                   (j - 1) % Ny, j, (j + 1) % Ny,
                   (j - 1) % Ny, j, (j + 1) % Ny]).astype(int)

    neighborhood = L[ci, cj]

    neighborhood[4] = 0 #not self

    me = L[i, j]
    if me == 0:
        return 0  # Empty house, no happiness to evaluate.
    
    if me == C:
        if np.any(neighborhood == A) or np.any(neighborhood == B):
            return 1  # happy C family.
        else:
            return -1  # Unhappy C family.
    
    N_types = np.size(f_types)
    f_counts = np.zeros(N_types)
    
    for n in range(N_types):
        f_counts[n] = np.size(np.where(neighborhood == f_types[n])[0])
    
    count_A = np.sum(neighborhood == A)
    count_B = np.sum(neighborhood == B)
    count_C = np.sum(neighborhood == C)

    if me == A:
        c_similar = count_A
        c_other = count_B + count_C
    elif me == B:
        c_similar = count_B
        c_other = count_A + count_C
    
    # Happy if not in the minority.
    # '''
    if c_similar >= c_other:
        happiness = 1
    else:
        happiness = - 1
    # '''
        
    return happiness


N = 50  # Lattice side.
f_empty= 0.1  # Fraction of empty houses.
f_A = 0.40
f_B = 0.40
f_C = 0.10

A = 1  # Type A.
B = - 1  
C = 2
EMPTY = 0


H = N * N  # Total number of houses (cells).


num_E = int(np.round(H * f_empty))
num_A = int(np.round(H * f_A))
num_B = int(np.round(H * f_B))
num_C = H - num_E - num_A - num_B

pop = np.array([A]*num_A + [B]*num_B + [C]*num_C + [0]*num_E)
np.random.shuffle(pop)

L = pop.reshape(N, N)
available_cells = np.argwhere(L == 0)

# Create empty houses.
#count = 0
#running = True
"""
while running:
    rn = np.random.randint(H)
    j = rn % N
    i = int((rn - j) / N)
    if not (L[i, j] == 0):
        # print(f'i={i}, j={j}, count={count}, E={E}')
        #L[i, j] = 0
        available_cells[count, 0] = i
        available_cells[count, 1] = j
        count += 1
        if (count == E):
            running = False
"""

available_cells = available_cells.astype(int)  

# print(available_cells)

num_A = np.size(np.where(L == A)[0])
num_B = np.size(np.where(L == B)[0])
num_E = np.size(np.where(L == 0)[0])

f_types = [A, B]

print(f'Town: {N} x {N}. Total houses: {H}')
print(f'Empty apartments: {num_E}')
print(f'Type A apartments: {num_A}')
print(f'Type B apartments: {num_B}')
print(f"Type C apartments: {num_C}")

L_initial = L.copy()

L_side = N  # Side of the lattice representing the town.




def simulate(seed):
    max_step = 5000
    stable = False
    step = 0
    np.random.seed(seed)
    while not stable and step < max_step:
        moves_this_step = 0

        # Select a random family.
        rns = np.random.permutation(H)
        for rn in rns:
            j = rn % N
            i = int((rn - j) / N)
            if (L[i, j] == 0):
                continue  # Empty house, skip.

            # Check family happiness.
            happiness = check_family_happiness(L, i, j, f_types)
    
            # If unhappy, relocate randomly.
            if happiness == -1 and available_cells.shape[0] > 0:
                n = np.random.randint(available_cells.shape[0])
                # Family moves to n-th available cell.
                L[available_cells[n, 0], 
                available_cells[n, 1]] = L[i, j]
                # Former cell is made available.
                L[i, j] = 0
                available_cells[n, 0] = i
                available_cells[n, 1] = j
                moves_this_step += 1
    
        if moves_this_step == 0:
            stable = True
        step += 1

    return L_initial, L

L0, Lfa = simulate(seed=17)
L1, Lfb = simulate(seed=18)


plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(L0, cmap='viridis')
plt.title('Initial town, scenario A')

plt.subplot(2, 2, 2)
plt.imshow(L1, cmap='viridis')
plt.title('Initial town, scenario B')

plt.subplot(2, 2, 3)
plt.imshow(Lfa, cmap='viridis')
plt.title('Final town, scenario A')


plt.subplot(2, 2, 4)
plt.imshow(Lfb, cmap='viridis')
plt.title('Final town, scenario B')

plt.tight_layout(h_pad = 2.0)
plt.show()
