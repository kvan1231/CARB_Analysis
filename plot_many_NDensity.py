import NDensity as nd
import glob
import matplotlib.pyplot as plt
import matplotlib

# Set the limits to the colorbar and number of color bins
vmin = -4
vmax = 0
ncolor = 17
cmap_name = 'jet'

# Grab all of the files we're going to read in
dc_files = glob.glob("*dc.dat")
dc_files.sort()

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)

# Generate the color data for colorbar
cmap = plt.get_cmap(cmap_name)
cNorm = matplotlib.colors.Normalize(vmin=vmin,
                                    vmax=vmax)
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
levels = matplotlib.ticker.MaxNLocator(nbins=ncolor).\
    tick_values(vmin, vmax)

z = [[-100 for i in range(100)] for j in range(100)]
im = plt.contourf(z, levels=levels, cmap=cmap)

# Move to the first subplot
plt.subplot(221)

# Limit files to UCXBs
start_sys = 1
end_sys = 7
files_to_plot = dc_files[start_sys:end_sys]

# Plot the UCXBs
for num in range(len(files_to_plot)):
    system = files_to_plot[num]
    print(system)
    dp = nd.density_profile(file_name=system)
    dp.plot_density(solo_fig=False, colorbar=False,
                    min_color=vmin, max_color=vmax,
                    color_map=cmap_name)

# Set the x and y limits
plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)

# Remove the labels and add a title
plt.tick_params(labelbottom='off', labelsize=15)
plt.title("UCXB Observed Ratio", fontsize=16)

# Move to second subplot
plt.subplot(222)

# Limit files to short period systems
start_sys = 7
end_sys = 11
files_to_plot = dc_files[start_sys:end_sys]

# Plot the short period systems
for num in range(len(files_to_plot)):
    system = files_to_plot[num]
    print(system)
    dp = nd.density_profile(file_name=system)
    dp.plot_density(solo_fig=False, colorbar=False,
                    min_color=vmin, max_color=vmax,
                    color_map=cmap_name)

# Set the x and y limits
plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)

# Remove the labels and add a title
plt.tick_params(labelbottom='off', labelsize=15)
plt.title(r"Short Period Observed Ratio", fontsize=16)

# Move to the third subplot
plt.subplot(223)

# Limit files to medium period systems
start_sys = 11
end_sys = 13
files_to_plot = dc_files[start_sys:end_sys]

# Plot the medium period systems
for num in range(len(files_to_plot)):
    system = files_to_plot[num]
    print(system)
    dp = nd.density_profile(file_name=system)
    dp.plot_density(solo_fig=False, colorbar=False,
                    min_color=vmin, max_color=vmax,
                    color_map=cmap_name)

# Set the x and y limits
plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)

# Remove labels and add a title
plt.tick_params(labelbottom='off', labelsize=15)
plt.title(r"Medium Period Observed Ratio", fontsize=16)
# plt.text(s="Medium Period Observed Ratio", fontsize=16, x=-0.1, y=3.8)

# Move to fourth subplot
plt.subplot(224)

# Limit files to long period system
start_sys = 13
end_sys = 14
files_to_plot = dc_files[start_sys:end_sys]

# Plot the long period system
for num in range(len(files_to_plot)):
    system = files_to_plot[num]
    print(system)
    dp = nd.density_profile(file_name=system)
    dp.plot_density(solo_fig=False, colorbar=False,
                    min_color=vmin, max_color=vmax,
                    color_map=cmap_name)

# Set the x and y limits
plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)
# plt.ylim(0.925, 4)
# plt.xlim(-0.7, 1)

# Remove labels and add a title
plt.tick_params(labelleft='off', labelsize=15)
plt.title(r"Long Period Observed Ratio", fontsize=16)

# Adjust the spacing of the figure
# fig.tight_layout()

# Create space on right hand side for color bar
plt.subplots_adjust(bottom=0.08, left=0.08, right=0.84, top=0.92)

# Create the colorbar
position = fig.add_axes([0.86, 0.08, 0.02, 0.84])
cb = fig.colorbar(im, cax=position)
cb.set_label(r'$\log_{10}(f_{ \mathrm{obs}, i})$', fontsize=22)
cb.ax.tick_params(labelsize=15)

# Create an empty plot to place labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='None', top='off', bottom='off',
                left='off', right='off')

# Create labels for entire group of plots
plt.xlabel(r'$\log_{10}$ (Period/days)', fontsize=22)
plt.ylabel(r'Mass ($M_\odot$)', fontsize=22)
plt.savefig("NDensity_Grid.pdf")
