#4cd
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
        if np.any(neighborhood == A) and np.any(neighborhood == B):
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



def simulate(seed):
    pop = np.array([A]*num_A + [B]*num_B + [C]*num_C + [0]*num_E)
    np.random.shuffle(pop)

    L = pop.reshape(N, N)
    available_cells = np.argwhere(L == 0)
    available_cells = available_cells.astype(int)  

    f_types = [A, B, C]

    L_initial = L.copy()

    L_side = N  # Side of the lattice representing the town.


    time = []
    num_happy_A = []
    num_happy_B = []
    num_happy_C = []
    num_happy_total = []
    max_step = 5000
    stable = False
    step = 0
    np.random.seed(seed)
    while not stable and step < max_step:
        print(step)
        time.append(step)
        moves_this_step = 0

        # Select a random family.
        rns = np.random.permutation(H)
        A_hap = 0
        B_hap = 0
        C_hap = 0
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
            else:
                if L[i, j] == A:
                    A_hap += 1
                elif L[i, j] == B:
                    B_hap += 1
                elif L[i, j] == C:
                    C_hap += 1
        num_happy_A.append(A_hap)
        num_happy_B.append(B_hap)
        num_happy_C.append(C_hap)
        num_happy_total.append(A_hap + B_hap + C_hap)
    
        if moves_this_step == 0:
            stable = True
        step += 1

    return L_initial, L, time, num_happy_A, num_happy_B, num_happy_C, num_happy_total

L0, Lfa, t_a, A_a, B_a, C_a, T_a = simulate(seed=17)
L1, Lfb, t_b, A_b, B_b, C_b, T_b = simulate(seed=18)

from matplotlib.colors import ListedColormap

#Empty(0): white, A(1): Blue, B(-1): red, C(2): green
cmap = ListedColormap(['white', 'blue', 'red', 'green'])
norm = mcolors.BoundaryNorm(boundaries=[-1.5, -0.5, 0.5, 1.5, 2.5], ncolors=4)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(L0, cmap=cmap, norm=norm)
plt.title('Initial town, scenario A')

plt.subplot(2, 2, 2)
plt.imshow(L1, cmap=cmap, norm=norm)
plt.title('Initial town, scenario B')

plt.subplot(2, 2, 3)
plt.imshow(Lfa, cmap=cmap, norm=norm)
plt.title('Final town, scenario A')


plt.subplot(2, 2, 4)
plt.imshow(Lfb, cmap=cmap, norm=norm)
plt.title('Final town, scenario B')

plt.tight_layout(h_pad = 2.0)
plt.show()

def portions(A, B, C, T):
    T_nonzero = np.where(np.array(T) == 0, 1, T)
    return np.array(A) / T_nonzero, np.array(B) / T_nonzero, np.array(C) / T_nonzero

port_A_a, port_B_a, port_C_a = portions(A_a, B_a, C_a, T_a)
port_A_b, port_B_b, port_C_b = portions(A_b, B_b, C_b, T_b)
plt.figure(figsize=(10, 5))
plt.plot(t_a, A_a, '-', label='Type A')
plt.plot(t_a, B_a, '-', label='Type B')
plt.plot(t_a, C_a, '-', label='Type C')
plt.plot(t_a, T_a, '-', label='Total')
plt.xlabel('Time step')
plt.ylabel('Number of happy families')
plt.legend()
plt.title('Family happiness over time (Scenario A)')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(t_b, A_b, '-', label='Type A')
plt.plot(t_b, B_b, '-', label='Type B')
plt.plot(t_b, C_b, '-', label='Type C')
plt.plot(t_b, T_b, '-', label='Total')
plt.xlabel('Time step')
plt.ylabel('Number of happy families')
plt.legend()
plt.title('Family happiness over time (Scenario B)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_a, port_A_a, '-', label='portion A')
plt.plot(t_a, port_B_a, '-', label='portion B')
plt.plot(t_a, port_C_a, '-', label='portion C')
plt.xlabel('Time step')
plt.ylabel('Portion of happy families')
plt.legend()
plt.title('Portion of happy families by type over time (Scenario A)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_b, port_A_b, '-', label='portion A')
plt.plot(t_b, port_B_b, '-', label='portion B')
plt.plot(t_b, port_C_b, '-', label='portion C')
plt.xlabel('Time step')
plt.ylabel('Portion of happy families')
plt.legend()
plt.title('Portion of happy families by type over time (Scenario B)')
plt.show()