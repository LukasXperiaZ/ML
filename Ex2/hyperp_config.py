import time
from enum import Enum
from random import seed, random
from typing import List

from sklearn.neural_network import MLPClassifier


class Solver(Enum):
    adam = 'adam'
    lbfgs = 'lbfgs'
    sgd = 'sgd'


class LearningRate(Enum):
    constant = 'constant'
    # is a constant learning rate given by 'learning_rate_init'.

    invscaling = 'invscaling'
    # gradually decreases the learning rate at each time step 't' using an
    # inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)

    adaptive = 'adaptive'
    # keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing.
    # Each time two consecutive epochs fail to decrease training loss by at least tol,
    #   or fail to increase validation score by at least tol if 'early_stopping' is on,
    #   the current learning rate is divided by 5.


class Activation(Enum):
    identity = 'identity'
    # 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x

    logistic = 'logistic'
    # 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).

    tanh = 'tanh'
    # 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).

    relu = 'relu'
    # 'relu', the rectified linear unit function, returns f(x) = max(0, x)


class HyperpConfig(object):
    # Size of each hidden layer (number of nodes)
    hidden_layer_sizes: List[int] = [1]

    # Activation function for the hidden layer.
    activation: Activation = Activation.relu

    # Solver
    solver: Solver = Solver.adam

    # Strength of the L2 regularization term.
    #   The L2 regularization term is divided by the sample size when added to the loss.
    # See also: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
    alpha: float = 0.0001

    # Learning rate
    learning_rate: LearningRate = LearningRate.constant

    # Precondition solver=sgd or adam
    # The initial learning rate used. It controls the step-size in updating the weights.
    learning_rate_init: float = 0.001

    # Precondition: solver='sgd', learning_rate='invscaling'
    # The exponent for inverse scaling learning rate. It is used in updating effective learning rate when
    #   the learning_rate is set to 'invscaling'.
    power_t: float = 0.5

    # Precondition: solver='sgd' or 'adam'
    # Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number
    #   of iterations. Note that this determines the number of epochs
    #   (how many times each data point will be used), not the number of gradient steps.
    max_iter: int = 200

    # Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change
    #   consecutive iterations, unless learning_rate is set to 'adaptive', convergence is considered to be reached
    #   and training stops.
    tol: float = 1e-4

    # Precondition: solver='sgd'
    # Momentum for gradient descent update. Should be between 0 and 1. Only used when solver='sgd'.
    momentum: float = 0.9

    # Precondition: solver='sgd'
    # Whether to use Nesterov's momentum. Only used when solver='sgd' and momentum > 0.
    nesterovs_momentum: bool = True

    # Precondition: solver='adam'
    # Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1).
    beta_1: float = 0.9

    # Precondition: solver='adam'
    # Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1).
    beta_2: float = 0.999

    # Precondition: solver='sgd' or 'adam'
    # Maximum number of epochs to not meet tol improvement.
    n_iter_no_change: int = 10

    def create_initialized_mlp(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation.value,
            solver=self.solver.value,
            alpha=self.alpha,
            learning_rate=self.learning_rate.value,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            tol=self.tol,
            nesterovs_momentum=self.nesterovs_momentum,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            n_iter_no_change=self.n_iter_no_change
        )

    def random_init(self):
        # min max definitions
        min_hidden_layers = 1
        max_hidden_layers = 10

        min_nodes_per_layer = 1
        max_nodes_per_layer = 10

        alpha_min = 0.00001
        alpha_max = 0.001

        learning_rate_init_min = 0.01
        learning_rate_init_max = 0.0001

        power_t_min = 0.2
        power_t_max = 0.8

        max_iter_min = 50
        max_iter_max = 1000

        r_tol_min = 1e-3
        r_tol_max = 1e-5

        momentum_min = 0.5
        momentum_max = 0.99

        beta_1_min = 0.8
        beta_1_max = 0.99

        beta_2_min = 0.9
        beta_2_max = 0.99999

        n_iter_no_change_min = 5
        n_iter_no_change_max = 30

        # random initialization
        seed(time.time())
        hidden_layers: int = int(min_hidden_layers + (random() * (max_hidden_layers - min_hidden_layers)))
        hidden_layers_sizes: List[int] = []
        for i in range(hidden_layers):
            hidden_layers_sizes.append(int(min_nodes_per_layer + (random() * (max_nodes_per_layer - min_nodes_per_layer))))

        # hidden_layer_sizes
        self.hidden_layer_sizes = hidden_layers_sizes

        # activation
        r_activation = random()
        if r_activation < 0.25:
            self.activation = Activation.relu
        elif 0.25 <= r_activation < 0.5:
            self.activation = Activation.tanh
        elif 0.5 <= r_activation < 0.75:
            self.activation = Activation.identity
        else:
            self.activation = Activation.logistic

        # solver
        r_solver = random()
        if r_solver < 0.3333:
            self.solver = Solver.adam
        elif 0.3333 <= r_solver < 0.6666:
            self.solver = Solver.sgd
        else:
            self.solver = Solver.lbfgs

        # alpha
        r_alpha = alpha_min + (random() * (alpha_max - alpha_min))
        self.alpha = r_alpha

        # learning_rate
        r_learning_rate = random()
        if r_learning_rate < 0.3333:
            self.learning_rate = LearningRate.constant
        elif 0.3333 <= r_learning_rate < 0.6666:
            self.learning_rate = LearningRate.adaptive
        else:
            self.learning_rate = LearningRate.invscaling

        # learning_rate_init
        if self.solver == Solver.sgd or self.solver == Solver.adam:
            r_learning_rate_init = learning_rate_init_min + (random() * (learning_rate_init_max - learning_rate_init_min))
            self.learning_rate_init = r_learning_rate_init

        # power_t
        if self.solver == Solver.sgd and self.learning_rate == LearningRate.invscaling:
            r_power_t = power_t_min + (random() * (power_t_max - power_t_min))
            self.power_t = r_power_t

        # max_iter
        if self.solver == Solver.sgd or self.solver == Solver.adam:
            r_max_iter = int(max_iter_min + (random() * (max_iter_max - max_iter_min)))
            self.max_iter = r_max_iter

        # tol
        if not self.learning_rate == LearningRate.adaptive:
            r_tol = r_tol_min + (random() * (r_tol_max - r_tol_min))
            self.tol = r_tol

        # momentum
        if self.solver == Solver.sgd:
            r_momentum = momentum_min + (random() * (momentum_max - momentum_min))
            self.momentum = r_momentum

            # nesterovs_momentum
            if random() < 0.5:
                self.nesterovs_momentum = True
            else:
                self.nesterovs_momentum = False

        # beta_1
        if self.solver == Solver.adam:
            r_beta_1 = beta_1_min + (random() * (beta_1_max - beta_1_min))
            if r_beta_1 >= 1.0:
                r_beta_1 = 0.999999
            self.beta_1 = r_beta_1

            # beta_2
            r_beta_2 = beta_2_min + (random() * (beta_2_max - beta_1_min))
            if r_beta_2 >= 1.0:
                r_beta_2 = 0.999999
            self.beta_2 = r_beta_2

        # n_iter_no_change
        if self.solver == Solver.sgd or self.solver == Solver.adam:
            r_n_iter_no_change = int(n_iter_no_change_min + (random() * (n_iter_no_change_max - n_iter_no_change_min)))
            self.n_iter_no_change = r_n_iter_no_change

