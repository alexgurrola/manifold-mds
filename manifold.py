# cost
import tensorflow as tf
import autograd.numpy as np

# manifolds
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent

# (1) Instantiate a manifold
manifold = Stiefel(5, 2)

"""
# (2) Define the cost function
X = tf.Variable(tf.placeholder(tf.float32))
cost = tf.exp(-tf.reduce_sum(tf.square(X)))
problem = Problem(manifold=manifold, cost=cost, arg=X, egrad=None, ehess=None, grad=None, hess=None, precon=None,
                  verbosity=2)
"""


# (2) Define the cost function (here using autograd.numpy)
def cost(X): return np.sum(X)


problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
solver = SteepestDescent()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)
