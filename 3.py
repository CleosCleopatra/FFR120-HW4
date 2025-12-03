from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)

def generate_triangular_graph_base(Nx, Ny):
    # Initially just integer coordinates, to build the graph easily.
    x_pts = np.arange(Nx)
    y_pts = np.arange(Ny)
    xg, yg = np.meshgrid(x_pts, y_pts)
    # I number the grid points.
    Nv = np.size(xg)
    nl = np.arange(Nv)
    ng = nl.reshape(xg.shape)
    
    A = np.zeros([Nv, Nv])
    
    for i in range(Nx):  # Column, second index in xg, yg, ng.
        for j in range (Ny):  # Row, first index in xg, yg, ng.
            ng0 = ng[j, i]
            
            # Connect with node to the left, if any.
            if i < (Nx - 1):
                ng1 = ng[j, i + 1]
                A[ng0, ng1] = 1
                A[ng1, ng0] = 1
                                
            # Connect with node upwards, if any.
            if j < (Ny - 1):
                ng3 = ng[j + 1, i]
                A[ng0, ng3] = 1
                A[ng3, ng0] = 1
            
            # Connect diagonally, depending on the row number.
            # If row number is even
            if j < (Ny - 1):
                if np.mod(j, 2) == 0:
                    if i > 0:
                        ng4 = ng[j + 1, i - 1]
                        A[ng0, ng4] = 1
                        A[ng4, ng0] = 1
                else:
                    if i < (Nx - 1):
                        ng2 = ng[j + 1, i + 1]
                        A[ng0, ng2] = 1
                        A[ng2, ng0] = 1
    
    return xg, yg, ng, A


def generate_triangular_graph(Nx, Ny, l_triangle):
    xg0, yg0, ng, A = generate_triangular_graph_base(Nx, Ny)
    
    xg = np.zeros(xg0.shape)

    for i in range(Nx):
        for j in range(Ny):
            if np.mod(j, 2) == 1:
                xg[j, i] = xg0[j, i] + 0.5
            else:
                xg[j, i] = xg0[j, i]
                           
    xg = xg * l_triangle
    yg = yg0 * l_triangle * np.sqrt(3) / 2
    
    return xg, yg, ng, A


def perturb_triangular_mesh(xg, yg, l_triangle, dr0, scale):
    Nx = xg.shape[1]
    Ny = xg.shape[0]
    
    for i in range(Nx):
        for j in range(Ny):
            # random radial displacement
            dr = dr0 + np.random.rand() * scale * l_triangle
            # random angle
            phi = np.random.rand() * 2 * np.pi
            xg[j, i] = xg[j, i] + dr * np.cos(phi)
            yg[j, i] = yg[j, i] + dr * np.sin(phi)
                        
    return xg, yg

Nx = 31
Ny = 21
l_triangle = 10
xg, yg, ng, A = generate_triangular_graph(Nx, Ny, l_triangle)
xv = xg.reshape(xg.size)
yv = yg.reshape(yg.size)
nv = ng.reshape(ng.size)

Nv = nv.size


dr_max = 0.25 * l_triangle
dr0 = 0.0

scale = dr_max

xgp, ygp = perturb_triangular_mesh(xg, yg, l_triangle, dr0, scale)

xvp = xgp.reshape(xgp.size)
yvp = ygp.reshape(ygp.size)


def distance_weight_matrices(x, y, connections):
    N = np.size(x)

    distances = np.zeros([N, N])
    weights = np.zeros([N, N])
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            distances[i, j] = d
            distances[j, i] = d
            weights[i, j] = 1 / d
            weights[j, i] = 1 / d
    
    return distances, weights




def branch_decision_rule(w_i, tau_i, connections_i, alpha, beta):
    N = np.size(w_i)   
    
    # Calculate the probability of the different connected nodes.
    p_i = (tau_i ** alpha) * (w_i ** beta)    
    p_i /= np.sum(p_i)
    
    # Calculate the cumulative probability.
    cp = np.cumsum(p_i)
    
    # Generate random number.
    rn = np.random.rand()
    
    # Select the next node.
    j = np.searchsorted(cp, rn)

    index_next_node = connections_i[j]

    return index_next_node
def simplify_path(path, distances):
    N = np.size(distances[0,:])
    Np = np.size(path)
    current_path = np.zeros(Np)
    current_path[:] = path[:]
    current_path = current_path.astype(int)
    
    running = True
    
    iterations = 0
        
    while running: 
        
        iterations += 1
    
        loops = np.zeros(N)
        occurrences = np.zeros([N, 2])
        simplified_lengths = np.zeros(N)

        # Check for loops.
        for i in range(N):
            spots = np.where(current_path == i)[0]
            if np.size(spots) > 1:
                # There is a loop.
                loops[i] = 1
                occurrences[i, 0] = spots[0]
                occurrences[i, 1] = spots[-1]
                
        occurrences = occurrences.astype(int)

        N_loops = np.sum(loops)
        
        if N_loops > 0:
            # Check what length would be the path without a certain loop.
            looping = np.where(loops == 1)[0]
        
            for i in looping:
                path_length = 0
                for j in range(occurrences[i, 0] - 1):
                    path_length += distances[current_path[j], current_path[j + 1]]
                for j in range(occurrences[i, 1], Np - 1):
                    path_length += distances[current_path[j], current_path[j + 1]]
                simplified_lengths[i] = path_length
            
            if N_loops == 1:
                i0 = looping[0]
            else:
                p_loops = np.cumsum(simplified_lengths[looping])
                p_loops /= np.sum(simplified_lengths[looping])
                rn = np.random.rand()
                # Select the next node.
                i0 = looping[np.searchsorted(p_loops, rn)]

            simplified_path = []
            for j in range(occurrences[i0, 0]):
                simplified_path.append(current_path[j])
            for j in range(occurrences[i0, 1], Np):
                simplified_path.append(current_path[j])

            simplified_path = np.array(simplified_path)
            Np = np.size(simplified_path)
            current_path = np.zeros(Np)
            current_path[:] = simplified_path[:]
            current_path = current_path.astype(int)
          
        else:
            running = False
    
    simplified = current_path
    
    return simplified

def path_length(path, distances):
    Np = np.size(path)
    
    p_length = 0
    for j in range(Np - 1):
        p_length += distances[path[j], path[j + 1]]
            
    return p_length

xg, yg, ng, A = generate_triangular_graph(Nx, Ny, l_triangle)


xv = xg.reshape(xg.size)
yv = yg.reshape(yg.size)
nv = ng.reshape(ng.size)

Nv = nv.size


xgp, ygp = perturb_triangular_mesh(xg, yg, l_triangle, dr0, scale)

xvp = xgp.reshape(xgp.size)
yvp = ygp.reshape(ygp.size)

connections = np.copy(A)


distances, weights = distance_weight_matrices(xvp, yvp, connections)

tau = connections * 1.0

# Choose initial node.
ref_x = np.min(xvp)
ref_y = np.min(yvp)
dist_start = np.sqrt((xvp - ref_x) ** 2 + (yvp- ref_y) ** 2)
i_ini = np.argmin(dist_start)

# Choose final node.
ref_x = np.max(xvp)
ref_y = np.max(yvp)
dist_end = np.sqrt((xvp - ref_x) ** 2 + (yvp - ref_y) ** 2)
i_fin = np.argmin(dist_end)


def run(N_ants, S_rounds, N_rounds, alpha, beta, rho, Q):
    Nv = np.size(xv)
    tau = np.copy(connections)
    shortest_path = []  # shortest_path[n] has shortest path at round n.
    l_path = np.zeros(N_rounds)  # Length of the shortest path. 
    reaching_target = np.zeros(N_rounds)  # Number of ants reaching the end.
    tau_list = [tau.copy()]  # To keep track of the pheromone matrix.

    for n in range(N_rounds):
    
        print(f'Round : {n}')

        # Variables to keep track of the ants in each rounds.
        paths = np.zeros([N_ants, S_rounds]).astype(int)
        steps = np.zeros(N_ants).astype(int)
        on_target = np.zeros(N_ants).astype(int)
        delta_tau = np.zeros([Nv, Nv])
        lengths = np.zeros(N_ants)
    
        paths[:, 0] = i_ini

        for ant in range(N_ants):
            running = True
            while running and (steps[ant] < S_rounds - 1):
                # Ant steps forward.
                i = paths[ant, steps[ant]]  # Current node for ant i.
                # Nodes adjacent to node i.
                adj_i = np.where(connections[i,:] == 1)[0]  
                adj_i = adj_i.astype(int)
                i_next_node = branch_decision_rule(
                    weights[i, adj_i], 
                    tau[i, adj_i], 
                    adj_i, alpha, beta)
                steps[ant] += 1
                paths[ant, steps[ant]] = i_next_node
        
                if i_next_node == i_fin:
                    on_target[ant] = 1
                    running = False

        # Select the ants reaching the final node.
        reach_target = np.where(on_target == 1)[0]
        reach_target = reach_target.astype(int)

        for ant in reach_target:
            # Here simplify the path if needed.
            single_path = paths[ant, :steps[ant] + 1]
            simplified_path = simplify_path(single_path, distances)
            Np = np.size(simplified_path) - 1
            steps[ant] = Np
            paths[ant, :steps[ant] + 1] = simplified_path
        
            # Calculate path length.
            lengths[ant] = path_length(simplified_path, distances)
            dq = Q / Np
    
            # Calculate pheromone increment.
            for i in range(Np):
                delta_tau[simplified_path[i], simplified_path[i + 1]] += dq
        
    
        # Store the shortest target of the round.
        if reach_target.size > 0:
            i_shortest = reach_target[np.argsort(lengths[reach_target])[0]]
    
            shortest_path.append(paths[i_shortest, :steps[i_shortest] + 1])
            l_path[n] = lengths[i_shortest]
            reaching_target[n] = np.size(reach_target)
        else:
            shortest_path.append([])
            l_path[n] = np.inf
            reaching_target[n] = 0
        
        # Now calculate the new pheromone matrix for next round.    
        tau = (1 - rho) * tau + delta_tau
        tau_list.append(tau)

        print(f'Ants on target = {reaching_target[n]} - Path length = {l_path[n]}')

    l_min = np.amin(l_path)
    n_min = np.where(l_path <= l_min)[0][0]
    p_min = shortest_path[n_min]

    return np.array(shortest_path, dtype=object), l_path, reaching_target, tau_list

param_sets = [
    (200, 1500, 30, 1.0, 1.2, 0.1, np.ceil(Nv/2)),
    (250, 2000, 30, 0.8, 2.0, 0.6, np.ceil(Nv/3)),
    (150, 1500, 30, 1.2, 4.0, 0.3, np.ceil(Nv/2))
]

results = []
best_idx = None
best_len = np.inf

for idx, (N_ants, S_rounds, N_rounds, alpha, beta, rho, Q) in enumerate(param_sets):

    shortest_paths, l_path, reaching_target, tau_list = run(
        N_ants, S_rounds, N_rounds, alpha, beta, rho, Q)
    if np.min(l_path) < best_len:
        best_len = np.min(l_path)
        best_idx = idx
    results.append((shortest_paths, l_path, reaching_target, tau_list))


plt.figure(figsize=[8, 8])

for ind, res in enumerate(results):
    (N_ants, S_rounds, N_rounds, alpha, beta, rho, Q) = param_sets[ind]
    label = f'Ants={N_ants}, S={S_rounds}, a={alpha}, b={beta}, r={rho}, Q={Q}'
    plt.plot(np.arange(len(res[1])) +1, res[1], '-o', label = label)

plt.xlabel('Round number')
plt.ylabel('Length of shortest path')
plt.title('Shortest path length over rounds')
plt.legend()
plt.show()

print(f'Best parameter set index: {best_idx} with length {best_len}')

         


plt.figure(figsize=[5,5])
for i in range(Nv):
    for j in range(i + 1, Nv):
        if connections[i, j] == 1:
            plt.plot([xvp[i], xvp[j]], [yvp[i], yvp[j]], 'k-', lw=0.5, alpha=0.3)
plt.plot(xvp, yvp, '.', markersize=12)

best_shortest = results[best_idx][0]
best_length_path = results[best_idx][1]
best_round = np.argmin(best_length_path)
absolute_best_path = best_shortest[best_round]

for i in range(len(absolute_best_path)-1):
    a = absolute_best_path[i]
    b = absolute_best_path[i + 1]
    plt.plot([xvp[a], xvp[b]],
             [yvp[a], yvp[b]],
             '-', color= 'c', linewidth=4)

plt.axis("Equal")
plt.title('Shortest path')
plt.show()


plt.figure(figsize=[5,5])

for i in range(Nv):
    for j in range(i + 1, Nv):
        if connections[i, j] == 1:
            plt.plot([xvp[i], xvp[j]], [yvp[i], yvp[j]], 'k-', lw=0.5, alpha=0.3)

plt.plot(xvp, yvp, '.', markersize=10)

first_round_path = results[best_idx][0][0]

for i in range(len(first_round_path)-1):
    a = first_round_path[i]
    b = first_round_path[i + 1]
    plt.plot([xvp[a], xvp[b]],
             [yvp[a], yvp[b]],
             '-', color= 'r', linewidth=4)
    
plt.title("Shortest path in first round")
plt.axis("Equal")
plt.show()


#Last round
plt.figure(figsize=[5,5])

for i in range(Nv):
    for j in range(i+1, Nv):
        if connections[i, j] == 1:
            plt.plot([xvp[i], xvp[j]], [yvp[i], yvp[j]], 'k-', lw=0.5, alpha=0.3)

plt.plot(xvp, yvp, '.', markersize=10)
last_round_path = results[best_idx][0][-1]
for i in range(len(last_round_path)-1):
    a = last_round_path[i]
    b = last_round_path[i + 1]
    plt.plot([xvp[a], xvp[b]],
             [yvp[a], yvp[b]],
             '-', color= 'g', linewidth=4)

plt.title("Shortest path in last round")
plt.axis("Equal")
plt.show()

