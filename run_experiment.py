import numpy as np
import matplotlib.pyplot as plt

from UCB import UCB
from Experiment import Experiment
from Bernoulli import Bernoulli
from Environment import Environment
from Execute import execute
from StructType import StructType
from Result import Result


# set up the backend for the plot
plt.rcdefaults()

# construct the arms and build the corresponding environment
env = Environment([Bernoulli(p) for p in [0.7, 0.3]])
n_arms = env.n_arms
print("### Environment\n" + str(env))

# prepare the algorithm
ucb_params = StructType()
ucb_params.scaling = 1.0
alg = UCB(n_arms, ucb_params)
print("### Algorithm\n" + str(alg))

# prepare the experiment
exp = Experiment(1000)

# execute the experiment and collect the results
res = execute(exp, env, alg)  # type : Result

# compute statistics
stats = res.compute_statistics()

# plot a chart with the cumulative regret
cum_exp_regret = np.cumsum(stats.exp_regret)
print(cum_exp_regret)

plt.plot(np.arange(exp.n_steps), cum_exp_regret, linewidth=2.5, linestyle="-")
plt.legend(loc='upper left', frameon=False)
plt.xlabel("steps")
plt.ylabel("cumulative regret")


plt.savefig("exercice_2.png",dpi=72)
plt.show()
