import rate_calc
import matplotlib.pyplot as plt
import numpy as np

rt = rate_calc.rate_table()
rt.plot_density("rate_to_plot.dat", max_color=6, min_color=0,
                cb_label=r"$\log_{10}(\Gamma_{ij}$ / Systems Gyr$^{-1})$", color_map='jet')

plt.ylim(1, 5)
plt.xlim(-0.5, 2.5)

min_data = np.loadtxt("min_mass.txt", delimiter=',')

y_line, x_line1, x_line2 = min_data.T
plt.plot(np.log10(x_line1), y_line, 'r--')
plt.plot(np.log10(x_line2), y_line, 'k--')

plt.text(-0.45, 2, "A", fontsize=18)
plt.text(-0.25, 2.2, "B", fontsize=18)
plt.text(-0.10, 1.5, "C", fontsize=18)
plt.text(0.01, 2.7, "D", fontsize=18)
plt.text(0.3, 2, "E", fontsize=18)
plt.text(0.25, 4.00, "F", fontsize=18)
plt.text(0.7, 3.50, "G", fontsize=18)

plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)

plt.title("Formation Rate Estimate", fontsize=20)
plt.savefig("rate_density_labels.pdf")
