import random_progens as rp
import numpy as np


# Initialize the code
pt = rp.persistent_table()

# Generate 1000 iterations where we want to
# grab simulations that appear persistent
pt.gen_avg_lmxb_density(N_iters=1000)

# compare our simulated systems to observed
# this table contains the number of
# 0 : total LMXBs similar to observed produced
# 1 : ucxbs similar to observed produced
# 2 : short period systems similar to observed produced
# 3 : mediume period similar to observed produced
# 4 : long period similar to observed produced
num_obs = pt.compare_to_obs()

# Get the individual averages of each class
average_tot_LMXB = np.average(num_obs[:, 0])
average_UCXB = np.average(num_obs[:, 1])
average_short = np.average(num_obs[:, 2])
average_medium = np.average(num_obs[:, 3])
average_long = np.average(num_obs[:, 4])
