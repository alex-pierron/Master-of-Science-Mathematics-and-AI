import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


# Define the games with descriptions and payoff matrices
zero_sum_games = {
    "Rock-Paper-Scissors": {
        "description": "The classic Rock-Paper-Scissors game.",
        "payoff_matrix": np.array([[0, -1, 1],
                                    [1, 0, -1],
                                    [-1, 1, 0]])
    },
    "Modified Rock-Paper-Scissors": {
        "description": "An extended version of Rock-Paper-Scissors with additional choices: lizard and Spock.",
        "payoff_matrix": np.array([[0, -1, 1, 1, -1],
                                    [1, 0, -1, -1, 1],
                                    [-1, 1, 0, 1, -1],
                                    [-1, 1, -1, 0, 1],
                                    [1, -1, 1, -1, 0]])
    },
    "Trading Game": {
        "description": "A game where players decide to buy, sell, or hold a stock against the market that can go up or down right after the decision but are supposed simultaneous.",
        "payoff_matrix": np.array([[5, -10],
                                   [-5, 5],
                                   [10, -5]])
    },
    "Guess the answer": {
        "description": "A game where player 1 can choose among 4 answers and player 2 has to guess if it is right or wrong.",
        "payoff_matrix": np.array([[3, -2],
                                    [1, 4],
                                    [-1, 5],
                                    [2, -3]])
    },
    "Stock Market Duel": {
        "description": "A larger-scale game where players are investors in the stock market and decide whether to buy, sell, or hold stocks. This game has a larger payoff matrix (9x21) to provide a more complex decision-making environment.",
        "payoff_matrix": 4 * np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                    [3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3, -2, 3],
                                    [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                    [-2, 3, -1, -1, -2, 3, -1, -1, -2, 3, -1, -1, -2, 3, -1, -1, -2, 3, -1, -1, -2],
                                    [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                                    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]])
    }
}



def random_simplex(m):
    """
    Generate a random vector in the m-dimensional simplex.
    Args:
    - m (int): Dimension of the simplex.
    Returns:
    - x (numpy.ndarray): Random vector in the m-dimensional simplex.
    """
    x = np.random.rand(m)
    x = x / np.sum(x)
    return x

def gap_function(payoff_matrix, a, b):
    """
    Calculate the gap between the maximum payoff for player B and the minimum payoff for player A.

    Args:
    - payoff_matrix (numpy.ndarray): Payoff matrix of the zero-sum game.
    - a (numpy.ndarray): Strategy distribution for player A.
    - b (numpy.ndarray): Strategy distribution for player B.
    
    Returns:
    - gap (float): Gap between the maximum payoff for player B and the minimum payoff for player A.
    """
    return np.max(np.dot(payoff_matrix, b)) - np.min(np.dot(payoff_matrix.T, a))



def extrapolation_algorithm(payoff_matrix, epsilon=1e-6, max_iter=8000, 
                            gamma=0.01, display_gap=1000, seed=42, 
                            show_upper_bounds=True, verbose=True):
    """
    Implements the extrapolation algorithm to solve a zero-sum game with a given payoff matrix. It is the main algorithm of this study.

    Arguments:
    - payoff_matrix : np.array
        Payoff matrix of the zero-sum game.
    - epsilon : float, optional (default 1e-6)
        Stopping criterion based on the difference between successive iterations.
    - max_iter : int, optional (default 8000)
        Maximum number of iterations before stopping the algorithm.
    - gamma : float, optional (default 0.01)
        Step size parameter for the extrapolation algorithm.
    - display_gap (int): Frequency at which to print the current gap.
    - seed (int): Seed for random number generation.
    - show_upper_bounds (bool): Whether or not to display the theoretical upper bounds on the final plot.
    - verbose : bool, optional (default True)
        Print progress information at each iteration.

    Returns:
    - t : int
        Number of iterations performed before reaching the stopping criterion.
    - gap : list
        List of gap values at each iteration.
    - upper_bounds : list
        List of theoretical upper bounds on the gap at each iteration.
    - a : list
        List of calculated vectors 'a' at each iteration.
    - b : list
        List of calculated vectors 'b' at each iteration.
    """

    np.random.seed(seed)
    m, n = payoff_matrix.shape
    L = np.max([np.linalg.norm(payoff_matrix, ord=np.inf),np.linalg.norm(payoff_matrix.T, ord=np.inf)])

    a = [random_simplex(m)]
    b = [random_simplex(n)]
    gap = [gap_function(payoff_matrix, a[0], b[0])]
    
    if verbose:
        print(f"Iteration 0: Gap = {gap[0]}")
    
    for t in range(1, max_iter + 1):

        G_t = (-gamma * np.dot(payoff_matrix, b[-1]), gamma * np.dot(payoff_matrix.T, a[-1]))
        
        a_prime = np.exp(np.log(a[-1]) - G_t[0]) / np.sum(np.exp(np.log(a[-1]) - G_t[0]))
        b_prime = np.exp(np.log(b[-1]) - G_t[1]) / np.sum(np.exp(np.log(b[-1]) - G_t[1]))
        w_t = (a_prime,b_prime)
        
        g_prime_t = (-gamma * np.dot(payoff_matrix, b_prime), gamma * np.dot(payoff_matrix.T, a_prime))
        
        a_t = np.exp(np.log(a[-1]) - g_prime_t[0]) / np.sum(np.exp(np.log(a[-1]) - g_prime_t[0]))
        b_t = np.exp(np.log(b[-1]) - g_prime_t[1]) / np.sum(np.exp(np.log(b[-1]) - g_prime_t[1]))
        
        a.append(a_t)
        b.append(b_t)

        
        gap_value = gap_function(payoff_matrix, np.mean(a, axis=0), np.mean(b, axis=0))
        gap.append(gap_value)
        
        if verbose and t % display_gap == 0:
            print(f"Iteration {t}: Gap = {gap_value}")
        
        if gap_value <= epsilon:
            if verbose:
                print("Convergence achieved.")
            break

    # Calculate theoretical upper bounds on the gap
    #upper_bounds = (inf_norm) * (np.sqrt(m)+np.sqrt(n)) * ( np.sqrt(2 / np.arange(1,t+2)))
    upper_bounds = 2 * L * (np.sqrt(m)+np.sqrt(n)) / (np.arange(1,t+2))

    if verbose:
        #upper_bounds = (np.max(payoff_matrix) -np.min(payoff_matrix)) * (np.sqrt(m)+np.sqrt(n))/(np.sqrt(np.arange(1,t+2)))
        print("Max_iter reached.")
        plt.plot(np.arange(0, t + 1), gap,color='r',label="Gap")
        if show_upper_bounds == True:
            plt.plot(np.arange(0, t + 1), upper_bounds,color = "b",label="Theoretical upper bounds")
        plt.title("Convergence of Extrapolation algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Gap")
        plt.legend()
        plt.show()
            
    return t, gap, upper_bounds, a, b




def grid_search(payoff_matrix, parameter_grid, epsilon, max_iter=8000, display_gap=1000, verbose=True, show_plot=True, seed=42):
    """
    Conduct a grid search for the parameters of the extrapolation algorithm.

    Arguments:
    - payoff_matrix (np.array):
        Payoff matrix of the zero-sum game.
    - parameter_grid (dict):
        Dictionary containing the values for parameters to search over (gamma and max_iter).
        Example: parameter_grid = {'gamma': [0.01, 0.1, 1], 'max_iter': [5000, 10000]}
    - epsilon (float) :
        Value of epsilon for the extrapolation algorithm.
    - max_iter (int): optional (default 8000)
        Maximum number of iterations before stopping the algorithm.
    - display_gap (int): Frequency at which to print the current gap.
    - verbose (bool): optional (default True)
        Print progress information at each iteration.
    - show_plot (bool): optional (default True)
        Whether to display the plot of the final gap of each model through time.
    - seed (int): Seed for random number generation.

    Returns:
    - results : dict
        Dictionary containing the important parameters and final metrics for each model tested.
    - best_a : np.ndarray
        Strategy distribution for player A that provides the lowest gap.
    - best_b : np.ndarray
        Strategy distribution for player B that provides the lowest gap.
    - best_gap : float
        The lowest gap recorded.
    - best_time : int
        The time at which the lowest gap occurs.
    - best_model_for_gap (tuple):
        The parameters (gamma, max_iter) corresponding to the model with the lowest gap among all recorded gaps.
    """
    results = {}
    gap_history = {}

    L = np.max([np.linalg.norm(payoff_matrix, ord=np.inf), np.linalg.norm(payoff_matrix.T, ord=np.inf)])
    gamma_proposed = 1 / (2 * L)
    parameter_grid['gamma'].append(gamma_proposed)

    param_combinations = list(itertools.product(*parameter_grid.values()))

    for params in tqdm(param_combinations, desc="Parameter combinations"):
        gamma, max_iter = params
        if verbose:
            print(f"Running with gamma={gamma}, max_iter={max_iter}")

        _, gap,_, _, _ = extrapolation_algorithm(payoff_matrix, epsilon, max_iter, gamma, 
                                                 verbose=False, display_gap=display_gap, seed=seed)
        results[(gamma, max_iter)] = {'final_gap': gap[-1]}
        gap_history[(gamma, max_iter)] = gap

    if show_plot:
        # Plot the final gap of each model through time
        plt.figure(figsize=(10, 6))
        for params, gap_values in gap_history.items():
            gamma, max_iter = params
            plt.plot(range(len(gap_values)), gap_values, label=f"gamma={gamma}, max_iter={max_iter}")
        plt.xlabel("Iteration")
        plt.ylabel("Gap")
        plt.title("Final Gap of Each Model Through Time")
        plt.legend()
        plt.show()

    # Find the model with the lowest recorded gap
    best_model = min(results, key=lambda x: results[x]['final_gap'])
    print(f"Model with the lowest final gap: gamma={best_model[0]}, max_iter={best_model[1]}, final_gap={results[best_model]['final_gap']}")

    # Find the lowest gap among all recorded gaps and its time
    best_gap = np.inf
    best_time = -1
    for params, gap_values in gap_history.items():
        min_gap = min(gap_values)
        min_time = gap_values.index(min_gap)
        if min_gap < best_gap:
            best_gap = min_gap
            best_time = min_time
            best_model_for_gap = params

    print(f"Model with the lowest gap among all recorded gaps: gamma={best_model_for_gap[0]}, max_iter={best_model_for_gap[1]}, gap={best_gap}, time={best_time}")

    # Return the strategy distributions for the lowest gap
    _, _,_, best_a, best_b = extrapolation_algorithm(payoff_matrix, epsilon, 
                                                     best_model_for_gap[1], best_model_for_gap[0], verbose=False, seed=seed)
    return results, best_a, best_b, best_gap, best_time, best_model_for_gap





def exponential_algorithm(payoff_matrix, epsilon=1e-6, max_iter=8000, verbose=True, 
                          display_gap = 1000, seed = 42, show_upper_bounds = True):
    """
    Implements the exponential algorithm to solve a zero-sum game with a given payoff matrix.

    Arguments:
    - payoff_matrix : np.array
        Payoff matrix of the zero-sum game.
    - epsilon : float, optional (default 1e-6)
        Stopping criterion based on the difference between successive iterations.
    - max_iter : int, optional (default 8000)
        Maximum number of iterations before stopping the algorithm.
    - verbose : bool, optional (default True)
        Print progress information at each iteration and the final plot.
    - display_gap (int): Frequency at which to print the current gap.
    - seed (int): Seed for random number generation.
    - show_upper_bounds (bool): Wether or not to display the theoretical upper bounds on the final plot.

    Returns:
    - t : int
        Number of iterations performed before reaching the stopping criterion.
    - gap : list
        List of gap values at each iteration.
    - a : list
        List of calculated vectors 'a' at each iteration.
    - b : list
        List of calculated vectors 'b' at each iteration.
    """
    np.random.seed(seed)
    m, n = payoff_matrix.shape
    inf_norm = np.linalg.norm(payoff_matrix, ord=np.inf)
    a = [random_simplex(m)]
    b = [random_simplex(n)]
    Ab_s = [np.dot(payoff_matrix, b[0])]
    Aa_s = [np.dot(payoff_matrix.T, a[0])]
    gap = [gap_function(payoff_matrix, a[0], b[0])]
    
    if verbose:
        print(f"Iteration 0: Gap = {gap[0]}")
    
    for t in range(1, max_iter + 1):
        eta_t = np.sqrt(2 * np.log(m)) / (inf_norm * np.sqrt(t))
        eta_t_prime = np.sqrt(2 * np.log(n)) / (inf_norm * np.sqrt(t))
        
        y_t = eta_t * np.sum(Ab_s, axis=0)
        z_t = -eta_t_prime * np.sum(Aa_s, axis=0)
        
        a_t = np.exp(y_t) / np.sum(np.exp(y_t))
        b_t = np.exp(z_t) / np.sum(np.exp(z_t))
        
        a.append(a_t)
        b.append(b_t)
        
        Ab_s.append(np.dot(payoff_matrix, b_t))
        Aa_s.append(np.dot(payoff_matrix.T, a_t))
        
        gap_value = gap_function(payoff_matrix, np.mean(a, axis=0), np.mean(b, axis=0))
        gap.append(gap_value)
        
        if verbose and t % display_gap == 0:
            print(f"Iteration {t}: Gap = {gap_value}")
        
        if gap_value <= epsilon:
            if verbose:
                print("Convergence achieved.")
            break

    # Calculate theoretical upper bounds on the gap
    upper_bounds = (inf_norm) * (np.sqrt(m)+np.sqrt(n)) * ( np.sqrt(2 / np.arange(1,t+2)))
    
    if verbose:
        upper_bounds = (np.max(payoff_matrix) -np.min(payoff_matrix)) * (np.sqrt(m)+np.sqrt(n))/(np.sqrt(np.arange(1,t+2)))
        print("Max_iter reached.")
        plt.plot(np.arange(0, t + 1), gap,color='r',label="Gap")
        if show_upper_bounds:
            plt.plot(np.arange(0, t + 1), upper_bounds,color = "b",label="Theoretical upper bounds")
        plt.title("Convergence of Exponential Weights algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Gap")
        plt.legend()
        plt.show()
            
    return t, gap, upper_bounds, a, b





def RM_algorithm(payoff_matrix, epsilon=1e-6, max_iter=8000, verbose=True, display_gap = 1000, 
                 seed = 42,show_upper_bounds = True):
    """
    Run the Regularized Mirror (RM) algorithm to solve a zero-sum game.

    Args:
    - payoff_matrix (numpy.ndarray): The payoff matrix of the game.
    - epsilon (float): Convergence threshold.
    - max_iter (int): Maximum number of iterations.
    - verbose (bool): Whether to print iteration information.
    - display_gap (int): Frequency at which to print the current gap.
    - seed (int): Seed for random number generation.
    - show_upper_bounds (bool): Wether or not to display the theoretical upper bounds on the final plot.

    Returns:
    - t (int): Number of iterations performed.
    - gap (list): List of gaps at each iteration.
    - a (list): List of player A's strategy distributions.
    - b (list): List of player B's strategy distributions.
    """
    np.random.seed(seed)
    m, n = payoff_matrix.shape
    
    # Initialize random strategy distributions
    a_0 = random_simplex(m)
    b_0 = random_simplex(n)
    a, b = [a_0], [b_0]
    
    # Initialize matrices for faster computation
    Ab_s = [np.dot(payoff_matrix, b_0)]
    inner_ab_m = [np.ones(m) * np.inner(a_0, Ab_s[0])]
    inner_ab_n = [np.ones(n) * np.inner(a_0, Ab_s[0])]
    Aa_s = [np.dot(payoff_matrix.T, a_0)]
    
    # Calculate initial gap
    gap = [gap_function(payoff_matrix, a_0, b_0)]
    
    if verbose:
        print(f"Iteration 0: Gap = {gap[0]}")
    
    for t in range(1, max_iter + 1):
        # Update player A's strategy
        x_t_plus = np.maximum(np.sum(np.array(Ab_s) - np.array(inner_ab_m), axis=0), 0)
        a_t = x_t_plus / np.linalg.norm(x_t_plus, ord=1) if np.any(x_t_plus != 0) else np.copy(a_0)
        
        # Update player B's strategy
        w_t_plus = np.maximum(np.sum(np.array(inner_ab_n) - np.array(Aa_s), axis=0), 0)
        b_t = w_t_plus / np.linalg.norm(w_t_plus, ord=1) if np.any(w_t_plus != 0) else np.copy(b_0)
            
        # Append new strategy distributions
        a.append(a_t)
        b.append(b_t)
        
        # Update matrices for faster computation
        Ab_s.append(np.dot(payoff_matrix, b_t))
        Aa_s.append(np.dot(payoff_matrix.T, a_t))
        inner_ab_m.append(np.ones(m) * np.inner(a_t, Ab_s[-1]))
        inner_ab_n.append(np.ones(n) * np.inner(a_t, Ab_s[-1]))
        
        # Calculate gap
        gap_value = gap_function(payoff_matrix, np.sum(a, axis=0)/(t+1), np.sum(b, axis=0)/(t+1))
        gap.append(gap_value)
        
        if verbose and t % display_gap == 0:
            print(f"Iteration {t}: Gap = {gap_value}")
        
        # Check convergence
        if gap_value <= epsilon:
            if verbose:
                print("Convergence achieved.")
            break
    
    # Calculate theoretical upper bounds on the gap
    upper_bounds = (np.max(payoff_matrix) -np.min(payoff_matrix)) * (np.sqrt(m)+np.sqrt(n))/(np.sqrt(np.arange(1,t+2)))
    
    if verbose:
            upper_bounds = (np.max(payoff_matrix) -np.min(payoff_matrix)) * (np.sqrt(m)+np.sqrt(n))/(np.sqrt(np.arange(1,t+2)))
            print("Max_iter reached.")
            plt.plot(np.arange(0, t + 1), gap,color='r',label="Gap")
            if show_upper_bounds:
                plt.plot(np.arange(0, t + 1), upper_bounds,color = "b",label="Theoretical upper bounds")
            plt.title("Convergence of RM Algorithm")
            plt.xlabel("Iteration")
            plt.ylabel("Gap")
            plt.legend()
            plt.show()
    
    return t, gap, upper_bounds, a, b



def RM_plus_algorithm(payoff_matrix, epsilon=1e-6, max_iter=8000, verbose=True, display_gap = 1000,  
                      seed=42, show_upper_bounds = True):
    """
    Run the Regularized Mirror + (RM+) algorithm to solve a zero-sum game.

    Args:
    - payoff_matrix (numpy.ndarray): The payoff matrix of the game.
    - epsilon (float): Convergence threshold.
    - max_iter (int): Maximum number of iterations.
    - verbose (bool): Whether to print iteration information.
    - display_gap (int): Frequency at which to print the current gap.
    - seed (int): Seed for random number generation.
    - show_upper_bounds (bool): Wether or not to display the theoretical upper bounds on the final plot.

    Returns:
    - t (int): Number of iterations performed.
    - gap (list): List of gaps at each iteration.
    - upper_bounds (list): Theoretical upper bounds on the gap at each iteration.
    - a (list): List of player A's strategy distributions.
    - b (list): List of player B's strategy distributions.
    """
    np.random.seed(seed)
    m, n = payoff_matrix.shape
    
    # Initialize random strategy distributions
    a_t = random_simplex(m)
    b_t = random_simplex(n)
    a_0, b_0 = np.copy(a_t), np.copy(b_t)
    a, b = [a_t], [b_t]
    x_t, w_t = np.zeros(m), np.zeros(n)
    
    # Initialize matrices for faster computation
    Ab_s = [np.dot(payoff_matrix, b_t)]
    inner_ab_m = [np.ones(m) * np.inner(a_t, Ab_s[0])]
    inner_ab_n = [np.ones(n) * np.inner(a_t, Ab_s[0])]
    Aa_s = [np.dot(payoff_matrix.T, a_t)]
    
    # Calculate initial gap
    gap = [gap_function(payoff_matrix, a_t, b_t)]
    
    if verbose:
        print(f"Iteration 0: Gap = {gap[0]}")
    
    for t in range(1, max_iter + 1):
        # Update player A's strategy
        x_t = np.maximum(x_t + Ab_s[-1] - np.ones(m) * np.inner(a_t, Ab_s[-1]), 0)
        a_t = x_t / np.linalg.norm(x_t, ord=1) if np.any(x_t != 0) else np.copy(a_0)
        
        # Update player B's strategy
        w_t = np.maximum(w_t + np.array(inner_ab_n[-1]) - np.array(Aa_s[-1]), 0)
        b_t = w_t / np.linalg.norm(w_t, ord=1) if np.any(w_t != 0) else np.copy(b_0)
            
        # Append new strategy distributions
        a.append(a_t)
        b.append(b_t)
        
        # Update matrices for faster computation
        Ab_s.append(np.dot(payoff_matrix, b_t))
        Aa_s.append(np.dot(payoff_matrix.T, a_t))
        inner_ab_m.append(np.ones(m) * np.inner(a_t, Ab_s[-1]))
        inner_ab_n.append(np.ones(n) * np.inner(a_t, Ab_s[-1]))
        
        # Calculate gap
        gap_value = gap_function(payoff_matrix, np.sum(a, axis=0) / (t + 1), np.sum(b, axis=0) / (t + 1))
        gap.append(gap_value)
        
        if verbose and t % display_gap == 0:
            print(f"Iteration {t}: Gap = {gap_value}")
        
        # Check convergence
        if gap_value <= epsilon:
            if verbose:
                print("Convergence achieved.")
            break
    
    # Calculate theoretical upper bounds on the gap
    upper_bounds = (np.max(payoff_matrix) - np.min(payoff_matrix)) * (np.sqrt(m) + np.sqrt(n)) / (np.sqrt(np.arange(1, t + 2)))
    
    if verbose:
        print("Max_iter reached.")
        plt.plot(np.arange(0, t + 1), gap, color='r', label="Gap")
        if show_upper_bounds:
            plt.plot(np.arange(0, t + 1), upper_bounds, color="b", label="Theoretical upper bounds")
        plt.title("Convergence of RM+ Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Gap")
        plt.legend()
        plt.show()
    
    return t, gap, upper_bounds, a, b




def compare_algorithms(payoff_matrix, epsilon=1e-6, max_iter=8000, display_gap=1000, verbose=True, seed=42, 
                       extrapolation_params=None, default_extrapolation_gamma=False):
    """
    Compare four algorithms: Exponential Weights, Regularized Mirror (RM), Regularized Mirror + (RM+), and Extrapolation.

    Args:
    - payoff_matrix (numpy.ndarray): The payoff matrix of the game.
    - epsilon (float): Convergence threshold.
    - max_iter (int): Maximum number of iterations.
    - display_gap (int): Frequency at which to print the current gap.
    - verbose (bool): Whether to print final plot.
    - seed (int): Seed for random number generation.
    - extrapolation_params (dict): Parameters for the extrapolation algorithm.
    - default_extrapolation_gamma (bool): If True, use the default proposed value of gamma for the extrapolation algorithm.

    Returns:
    - results (dict): Dictionary containing the metrics for each algorithm.
    """
    results = {}

    # Define models to evaluate
    models_to_evaluate = ['Exponential Weights', 'Regularized Mirror', 'Regularized Mirror +']

    if default_extrapolation_gamma:
            models_to_evaluate.append('Extrapolation')
            L = np.max([np.linalg.norm(payoff_matrix, ord=np.inf), np.linalg.norm(payoff_matrix.T, ord=np.inf)])
            gamma_proposed = 1 / (2 * L)
            if extrapolation_params:
                extrapolation_params["gamma"] = gamma_proposed
            else:
                extrapolation_params = {"gamma":gamma_proposed}

    elif extrapolation_params:
        models_to_evaluate.append('Extrapolation')

    # Run algorithms for each model
    with tqdm(total=len(models_to_evaluate), desc="Models") as pbar:
        for model in models_to_evaluate:
            if model == 'Exponential Weights':
                t, gap_exp, _, a_exp, b_exp = exponential_algorithm(payoff_matrix, epsilon, max_iter,
                                                                    verbose=False, display_gap=display_gap, seed=seed)
                results[model] = {'t': t, 'gap': gap_exp, 'a': a_exp, 'b': b_exp}
            elif model == 'Regularized Mirror':
                t, gap_rm, _, a_rm, b_rm = RM_algorithm(payoff_matrix, epsilon, max_iter,
                                                        verbose=False, display_gap=display_gap, seed=seed)
                results[model] = {'t': t, 'gap': gap_rm, 'a': a_rm, 'b': b_rm}
            elif model == 'Regularized Mirror +':
                t, gap_rm_plus, _, a_rm_plus, b_rm_plus = RM_plus_algorithm(payoff_matrix, epsilon, 
                                                                            max_iter, verbose=False, display_gap=display_gap,
                                                                             seed=seed)
                results[model] = {'t': t, 'gap': gap_rm_plus, 'a': a_rm_plus, 'b': b_rm_plus}
            elif model == 'Extrapolation':
                t, gap_extra,upper_bounds_extra, a_extra, b_extra = extrapolation_algorithm(payoff_matrix, 
                                                                                            epsilon, max_iter,verbose = False,
                                                                                            **extrapolation_params)
                results[model] = {'t': t, 'gap': gap_extra,'upper_bounds': upper_bounds_extra, 'a': a_extra, 'b': b_extra}
            pbar.update(1)

    if verbose:
        # Plot convergence curves
        plt.figure(figsize=(10, 6))
        for algo, data in results.items():
            plt.plot(range(len(data['gap'])), data['gap'], label=algo)
        plt.xlabel("Iteration")
        plt.ylabel("Gap")
        plt.title("Convergence of Different Algorithms")
        plt.legend()
        plt.show()
        # Find the model with the lowest gap
        lowest_gap_model = min(results, key=lambda x: results[x]['gap'][-1])
        print("Model with the lowest final gap:", lowest_gap_model)

    return results



def random_payoff_matrix(a, b, m, n, dtype=float, seed=None):
    """
    Generate a random payoff matrix with values between a and b with dimensions m x n.

    Parameters:
        a (float or int): Lower bound of payoff values.
        b (float or int): Upper bound of payoff values.
        m (int): Number of rows in the matrix.
        n (int): Number of columns in the matrix.
        dtype (type, optional): Data type of the elements in the matrix (float or int). Defaults to float.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        numpy.ndarray: Payoff matrix with dimensions m x n.
    """
    if seed is not None:
        np.random.seed(seed)
    if dtype == float:
        return np.random.uniform(a, b, size=(m, n))
    elif dtype == int:
        return np.random.randint(a, b+1, size=(m, n))
    else:
        raise ValueError("Invalid dtype. Must be either 'float' or 'int'.")