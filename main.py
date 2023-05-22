import numpy as np
import math
import multiprocessing as mp

# Define the Himmelblau's function
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Define the PSO function for parallel processing
def pso(num_particles, num_dimensions, max_iter, lb, ub, func):
    # Initialize the particles
    particles = np.zeros((num_particles, num_dimensions))
    for i in range(num_particles):
        particles[i] = np.random.uniform(lb, ub, num_dimensions)
    # Initialize the velocities
    velocities = np.zeros((num_particles, num_dimensions))
    # Initialize the personal best positions and values
    personal_best_pos = np.copy(particles)
    personal_best_val = np.array([func(p) for p in particles])
    # Initialize the global best position and value
    global_best_pos = np.copy(personal_best_pos[np.argmin(personal_best_val)])
    global_best_val = np.min(personal_best_val)
    # Set the inertia weight and acceleration coefficients
    w = 0.7
    c1 = 1.4
    c2 = 1.4
    # Initialize the pool of processes for parallel computation
    pool = mp.Pool(processes=mp.cpu_count())
    # Perform the iterations of PSO
    for it in range(max_iter):
        # Update the velocities
        velocities = w * velocities + \
            c1 * np.random.rand(num_particles, num_dimensions) * (personal_best_pos - particles) + \
            c2 * np.random.rand(num_particles, num_dimensions) * (global_best_pos - particles)
        # Update the particles
        particles = particles + velocities
        # Apply the boundary conditions
        particles = np.clip(particles, lb, ub)
        # Evaluate the new fitness values in parallel
        fitness_values = np.array(pool.map(func, particles))
        # Update the personal best positions and values
        personal_best_mask = fitness_values < personal_best_val
        personal_best_pos[personal_best_mask] = np.copy(particles[personal_best_mask])
        personal_best_val[personal_best_mask] = np.copy(fitness_values[personal_best_mask])
        # Update the global best position and value
        global_best_mask = personal_best_val < global_best_val
        global_best_pos = np.copy(personal_best_pos[np.argmin(personal_best_val)])
        global_best_val = np.min(personal_best_val)
        # Print the current best value for debugging
        print("Iteration {}: Best Value = {:.5f}".format(it+1, global_best_val))
    # Close the pool of processes
    pool.close()
    pool.join()
    # Return the best position and value
    return global_best_pos, global_best_val

# Define the main function
if __name__ == '__main__':
    bounds = [(-5, 5), (-5, 5)]
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'n_particles': 10, 'n_iterations': 100}

    # Find the roots of Himmelblau's function
    pso = PSO(2, Himmelblau().evaluate, bounds, **options)
    results = pso.run()

    # Print the results
    print("The roots of Himmelblau's function are:")
    for i, solution in enumerate(results):
        print(f"Root {i+1}: {solution}")