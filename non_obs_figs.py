import non_obs
import matplotlib.pyplot as plt
import numpy as np

color_map = 'jet'

pf = non_obs.progenitor_file(file_name="010_2A_1822-371_max.dat")
pf.plot_many_sims(frequency=4, init_sys=7, fin_sys=10,
                  min_color=-12, max_color=-6, num_color=13,
                  title="Region 'A'", color_map=color_map,
                  direction='lower', pt_freq=5)

plt.xlim(-0.05, 1.1)
plt.savefig("A_short_non_obs.pdf")

pf = non_obs.progenitor_file(file_name="011_Sco_X-1_max.dat")
pf.add_progen_file(file_name="012_GX_349+2_max.dat")

shifted_data = pf._shift_mass("upper")
unique_data = pf._return_limits(shifted_data)
unique_mass = unique_data[:, 0]
unique_period = unique_data[:, 2]
long_p_inds = np.where(unique_period > 0.2)[0]
mass = unique_mass[long_p_inds]
period = unique_period[long_p_inds]

pf.plot_many_sims(frequency=1, init_sys=11, fin_sys=12,
                  min_color=-12, max_color=-6, num_color=13,
                  mass=mass, period=period,
                  title="Region 'E'", color_map=color_map)
plt.xlim(0, 1.2)
plt.savefig("E_long_non_obs.pdf")

pf = non_obs.progenitor_file(file_name="013_Cyg_X-2_max.dat")
modified_mass_upper = np.array([3.6, 4.1, 4.5])
modified_period_upper = np.array([0.24, 0.76, 1.14])

pf.plot_many_sims(frequency=1, init_sys=13, fin_sys=14,
                  min_color=-12, max_color=-6, num_color=13,
                  mass=modified_mass_upper, period=modified_period_upper,
                  pt_freq=6, title="Region 'F'", color_map=color_map)

plt.xlim(0, 3.5)
plt.ylim(-0.25, 1.8)
plt.savefig("F_high_init_M.pdf")


shifted_data = pf._shift_mass("lower")
unique_data = pf._return_limits(shifted_data)
unique_mass = unique_data[:, 0]
unique_period = unique_data[:, 2]
mass = unique_mass[::3]
period = unique_period[::3]

modified_mass_lower = np.array([4.1, 4.1, 4.1, 4.1])

modified_period_lower = np.array([1.30, 1.34, 1.38, 1.42])

pf.plot_many_sims(frequency=1, init_sys=13, fin_sys=14,
                  min_color=-12, max_color=-6, num_color=13,
                  mass=modified_mass_lower, period=modified_period_lower,
                  pt_freq=6, title="Region 'G'", color_map=color_map)

plt.xlim(0, 3.5)
plt.ylim(-0.25, 1.8)
plt.savefig("G_high_init_PM.pdf")

pf = non_obs.progenitor_file(file_name="007_4U_1636-536_max.dat")
shifted_data = pf._shift_mass(direction="upper")
unique_data = pf._return_limits(shifted_data)
unique_mass = unique_data[:, 0]
unique_period = unique_data[:, 2]
mass = unique_mass
period = unique_period

pf.plot_many_sims(frequency=1, init_sys=0, fin_sys=10,
                  min_color=-12, max_color=-6, num_color=13,
                  pt_freq=5, direction="upper",
                  mass=mass[:3], period=period[:3],
                  title="Region 'D'", color_map=color_map)
plt.xlim(-0.25, 2.25)

plt.savefig("D_sub_UCXB_pres.pdf")

pf = non_obs.progenitor_file(file_name="011_Sco_X-1_max.dat")
pf.plot_many_sims(frequency=1, init_sys=0, fin_sys=12,
                  min_color=-12, max_color=-6, num_color=13,
                  pt_freq=5, direction="lower",
                  title="Region 'C'", color_map=color_map)

plt.savefig("C_super_UCXB_pres.pdf")

pf = non_obs.progenitor_file(file_name="010_2A_1822-371_max.dat")
shifted_data = pf._shift_mass(direction="upper")
unique_data = pf._return_limits(shifted_data)
unique_mass = unique_data[:, 0]
unique_period = unique_data[:, 2]

lower_lim = np.where(unique_mass >= 1.5)

upper_mass = unique_mass[lower_lim]
upper_period = unique_period[lower_lim]

upper_lim = np.where(upper_mass <= 2.7)

mass = upper_mass[upper_lim]
period = upper_period[upper_lim]
pf.plot_many_sims(frequency=1, init_sys=0, fin_sys=8,
                  min_color=-12, max_color=-6, num_color=13,
                  mass=mass, period=period, pt_freq=5,
                  title="Region 'B'", color_map=color_map)
plt.xlim(-0.25, 2.25)

plt.savefig("B_super_short_Pres.pdf")

