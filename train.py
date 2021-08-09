"""
Train a Bayesian-optimized Gabor Neural Network
@author: Thanh Le
"""
import GPyOpt
from model.model import BayesOptimizedGNN
import numpy as np


def bayesian_optimization(num_block=3, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
    network = BayesOptimizedGNN(num_block=num_block, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = network.evaluate()  # evaluate the network
    network.model.save('./checkpoints/acc_' + str(result[1]) + '_loss_' + str(result[0]) + '.h5')
    return result


def objective_function():
    """
    Define the objective function
    :param hyperparams: list of hyper-parameters (i.e., optimizable variables)
    :return: loss
    """
    print(hyperparams)
    global loss_acc  # access the global variable, just for traceback  
    result = bayesian_optimization(num_block=int(hyperparams[:, 0]),
                                   learning_rate=float(hyperparams[:, 1]),
                                   beta_1=float(hyperparams[:, 2]),
                                   beta_2=float(hyperparams[:, 3]))

    print(" Loss:\t{0} \t Accuracy:\t{1}".format(result[0], result[1]))
    loss_acc.append([result[0], result[1], hyperparams[:, 0], hyperparams[:, 1], hyperparams[:, 2], hyperparams[:, 3]])
    return result[0]


if __name__ == '__main__':
    loss_acc = []

    # Define the bounds for hyperparameters
    hyperparams = [{'name': 'num_block', 'type': 'discrete', 'domain': (2, 3)},
                   {'name': 'learning_rate', 'type': 'continuous', 'domain': (10 ** -3, 5 * 10 ** -2)},
                   {'name': 'beta_1', 'type': 'continuous', 'domain': (0.87, 0.93)},
                   {'name': 'beta_2', 'type': 'continuous', 'domain': (0.97, 0.999)}]

    # Run the Bayesian Optimization process
    optimization_process = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=hyperparams)
    optimization_process.run_optimization(max_iter=50)

    print("""
    Optimized hyper-parameters:
    \t{0}:\t{1}
    \t{2}:\t{3}
    \t{4}:\t{5}
    \t{6}:\t{7}
    """.format(hyperparams[0]["name"], optimization_process.x_opt[0],
               hyperparams[1]["name"], optimization_process.x_opt[1],
               hyperparams[2]["name"], optimization_process.x_opt[2],
               hyperparams[3]["name"], optimization_process.x_opt[3]))

    print("Optimized loss: {0}".format(optimization_process.fx_opt))
    loss_acc = np.array(loss_acc)
