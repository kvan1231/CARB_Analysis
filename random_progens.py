import glob
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib


class persistent_table():
    def __init__(self, pers_table="persistent_table.dat"):
        if not os.path.exists(pers_table):
            print(pers_table + " not found, generating...")
            self.gen_persistent_table_file()
        else:
            self.persistent_table = np.loadtxt(pers_table, delimiter=',')

    def gen_persistent_table_file(self,
                                  pers_table="persistent_table.dat",
                                  pers_dirs="period*/group*/",
                                  pers_files="persistent_sys.dat"):

        """
        Parse through the files at the directory inputted to combine the
        files together into one large table so it is easier to work with.

        Inputs
        ------
        pers_table: string
            Name of output file we are saving the persistent table to.
        pers_dirs : string
            The path to the files containing the persistent LMXB information.
        pers_files : string
            The name of the files that we're going to parse and combine.
        """

        # Combine the directory path and the file name together
        path_to_pers = pers_dirs + pers_files

        # Use glob to search for the files inputted
        persistent_sub_files = glob.glob(path_to_pers, recursive=True)

        # sort the files
        persistent_sub_files.sort()

        # loop through the files
        for sub_ind in range(len(persistent_sub_files)):
            sub_file = persistent_sub_files[sub_ind]

            # read in the file and split it into header and data
            header, temp_array = self._read_persistent_file(sub_file)

            # if its the first file we're parsing then we're using it to
            # initialize the array
            if sub_ind == 0:
                persistent_table = temp_array
            # otherwise simply stack the read in data onto the array
            else:
                persistent_table = np.vstack((persistent_table, temp_array))

        # Sort the array by mass then by period
        sorted_indices = np.lexsort((persistent_table[:, 0],
                                     persistent_table[:, 1]))
        sorted_persistent_table = persistent_table[sorted_indices]

        # store the entire table, the masses and periods internally
        self.persistent_table = sorted_persistent_table[:, :3]
        self.masses = sorted_persistent_table[:, 1]
        self.periods = sorted_persistent_table[:, 0]

        # save the table to be loaded in next time
        np.savetxt(pers_table, self.persistent_table,
                   fmt="%.3f, %.3f, %.3f")

    def get_bin_sizes(self, period, mass):
        """
        Find the corner of each bin and the size of the bins to
        be plotted. The x and y values of the density file are
        the initial conditions of the system and are not necessarily
        the limits of the plotted bin.

        Inputs
        ------
        period : array
            The period values that we are looking for the bin sizes of.
        mass : array
            The mass values that we are looking for the bin sizes of.
        """
        print("no bin size file found, calculating...")

        # Shorten our x and y columns to only have the unique values
        unique_xvals = np.unique(mass)
        unique_yvals = np.unique(period)

        # Calculate the edges and bin sizes of all our x values
        dx_part, xcorner_part = self._calc_bin_props(
            unique_xvals)

        # Calculate the edges and bin sizes of all our y values
        dy_part, ycorner_part = self._calc_bin_props(
            unique_yvals)

        # Repeat the bin corners to span our entire parameter space
        # with all possible combinations of x and y
        x_corner = np.repeat(xcorner_part, len(ycorner_part))
        y_corner = np.tile(ycorner_part, len(xcorner_part))

        # Repeat all the bin sizes as well
        x_bins = np.repeat(dx_part, len(dy_part))
        y_bins = np.tile(dy_part, len(dx_part))

        self.dm = x_bins
        self.dp = y_bins
        self.mbin = x_corner
        self.pbin = y_corner

    def gen_lmxb_density(self, N_lmxb=14, min_age=0, max_age=1e9,
                         age_lim=5e8):

        """
        Generates a table of N_lmxb random persistent binaries based on
        user inputted values

        N_lmxb : int
            An integer number of persistent LMXBs we want to generate
        min_age : float
            The lower limit of our age array that we will randomly select from
        max_age : float
            The upper limit of our age array that we'll randomly select from
        age_lim : float
            The limit for the age offset we want to handle when randomly
            selecting a system
        """

        # Make a copy of the persistent table and zero the third column
        density_table = copy.copy(self.persistent_table)
        density_table[:, -1] = density_table[:, -1] * 0.0

        # define the mass limits
        min_mass = min(self.masses)
        max_mass = max(self.masses)

        # define the period limits
        min_period = min(self.periods)
        max_period = max(self.periods)

        # check for the mass and period bins, if they arent defined yet
        # the define them here
        try:
            mbin = self.mbin
            pbin = self.pbin
        except AttributeError:
            self.get_bin_sizes(self.periods, self.masses)
            mbin = self.mbin
            pbin = self.pbin

        # initialize the array
        persistent_sys = np.zeros((N_lmxb, 3))

        # initialize the iteration count
        iteration = 1

        # re-zero the index
        persistent_index = 0

        # Start the loop
        while True:

            # Randomly select a mass, period and age
            random_mass = np.random.uniform(min_mass, max_mass)
            random_period = np.random.uniform(min_period, max_period)
            random_age = np.random.uniform(min_age, max_age)

            # find the distance from our random number to the bin edges
            m_abs_diff = abs(mbin - random_mass)
            p_abs_diff = abs(pbin - random_period)

            # find the bin closest to our random mass/period
            m_ind = np.where(min(m_abs_diff) == abs(m_abs_diff))[0]
            p_ind = np.where(min(p_abs_diff) == abs(p_abs_diff))[0]

            # find the index of that bin
            progen_ind = list(set(m_ind) & set(p_ind))[0]

            # Iterate the third column by one to denote we've randomly
            # generated a system with that progenitor pair
            density_table[progen_ind, -1] += 1

            # Grab that systems information
            temp_system = self.persistent_table[progen_ind]

            # grab that systems age
            persistent_age = temp_system[-1]

            # If that system is persistent
            if persistent_age > 0:

                # offset the age of that system by some random
                # amount defined above since not all systems
                # will be born at the same time
                offset_age = persistent_age + random_age

                # if the offset is within our limit
                if offset_age < age_lim:

                    # save the system and move to next spot in array
                    persistent_sys[persistent_index] = temp_system
                    persistent_index += 1
            iteration += 1

            # Once we've found enough persistent systems, leave loop
            if persistent_index == N_lmxb:
                break

        return density_table, persistent_sys, iteration

    def gen_avg_lmxb_density(self, N_iters=100, N_lmxb=14,
                             min_age=0, max_age=1e9,
                             age_lim=5e8):
        """
        This code runs through N_iters number of iterations through
        gen_lmxb_density to determine an average number of
        iterations necessary to produce N_lmxbs.

        N_iters : int
            An integer number of iterations we want to loop through
            gen_lmxb_density
        N_lmxb : int
            An integer number of persistent LMXBs we want to generate
        min_age : float
            The lower limit of our age array that we will randomly select from
        max_age : float
            The upper limit of our age array that we'll randomly select from
        age_lim : float
            The limit for the age offset we want to handle when randomly
            selecting a system

        """

        # Create an array equal to the length of our number of iterations
        num_its = np.zeros(N_iters)

        # start looping through the iterations
        for iteration in range(N_iters):

            # every mod 100 iterations output a string to show progress
            if iteration % 100 == 0:
                print("Iterations Completed: " + str(iteration))

            # run through the gen_lmxb_density code
            temp_den, temp_sys, temp_it = self.gen_lmxb_density(N_lmxb,
                                                                min_age,
                                                                max_age,
                                                                age_lim)
            # if its the first iteration then create the arrays
            # using the above output
            if iteration == 0:
                avg_den_tab = temp_den
                per_sys = temp_sys

            # otherwise append to the existing array
            else:
                avg_den_tab[:, -1] = avg_den_tab[:, -1] + temp_den[:, -1]
                per_sys = np.vstack((per_sys, temp_sys))

            # save the number of iterations needed
            num_its[iteration] = temp_it

        # get the bin sizes
        self.get_bin_sizes(self.periods, self.masses)
        x_bins = self.dp
        y_bins = self.dm

        # calculate the bin size
        bin_size = x_bins * y_bins

        # calculate the average density of the iterations
        avg_den_tab[:, -1] = avg_den_tab[:, -1] / bin_size
        avg_den_tab[:, -1] = avg_den_tab[:, -1] / max(avg_den_tab[:, -1])

        # store the values internally
        self.avg_density = avg_den_tab
        self.pers_sys = per_sys
        self.num_its = num_its
        # return avg_den_tab, per_sys

    def plot_density(self, alpha=1.0, avg_lmxb_file="",
                     max_color=0.1, min_color=0, num_color=11,
                     color_map='viridis', colorbar=True,
                     xvar=r'$\log_{10}$ (Period/days)',
                     yvar=r'Mass ($M_\odot$)', title="",
                     cb_label=r"Normalized Counts",
                     solo_fig=True):
        """
        A routine that produces a density plot using the density file given
        with the color showing different properties based on user choice.

        inputs
        ------
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
        max_color : float
            Defines the maximum value in the colorbar, default is 1
        min_color : float
            Defines the minimum value in the colorbar, default is 0
        num_color : integer
            Defines the number of breakpoints in the colorbar, the smaller
            the number the more discretized the colorbar appears. default is
            25
        color_map : string
            The colormap used in the colorbar, default is viridis.
            See Matplotlib colorbar documentation for more choices:
            https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        colorbar : boolean
            True to display a colorbar, true by default
        xvar : string
            Label for the x-axis, default label is log10(period)
        yvar : string
            Label for the y-axis, default label is Mass
        title : string
            Title of the plot produced
        cb_label : string
            Label associated with the colorbar
        solo_fig : boolean
            True or false to determine if we are only plotting a single density
            file for the figure. The code automatically clears the figure each
            time it is run. If we want to overplot data on top of each other
            set this flag to False.
        """

        # If we're only plotting this data alone then clear the figure
        if solo_fig:
            plt.close('all')
            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(1, 1, 1)

        try:
            self.avg_density
        except AttributeError:
            if avg_lmxb_file:
                self.load_avg_lmxb_density(avg_lmxb_file)
            else:
                self.gen_avg_lmxb_density()
        # Grab the density data
        density_data = self.avg_density
        xvalues = density_data[:, 0]
        # xmin = min(xvalues)
        # xmax = max(xvalues)

        yvalues = density_data[:, 1]
        # ymin = min(yvalues)
        # ymax = max(yvalues)

        xnum = len(np.unique(xvalues))
        ynum = len(np.unique(yvalues))

        rect_color = density_data[:, -1]

        # Define the colormap
        cmap = plt.get_cmap(color_map)
        cNorm = matplotlib.colors.Normalize(vmin=min_color,
                                            vmax=max_color)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        # Grab the bin properties
        # mass = self.masses
        # period = self.periods

        self.get_bin_sizes(self.periods, self.masses)
        x_bins = self.dp
        y_bins = self.dm
        x_corner = self.pbin
        y_corner = self.mbin

        # If the bin color is outside our colorbar range then ignore it
        color_check = np.where((rect_color > -1))

        # Loop through the indicies where the color is in the appropriate range
        for bin_index in color_check[0]:
            bin_color = rect_color[bin_index]
            xedge = x_corner[bin_index]
            yedge = y_corner[bin_index]

            # Convert the color value to the 4 channel RGBA value
            scaled_color = scalarMap.to_rgba(bin_color)

            # Create the rectangle
            rect = patches.Rectangle((xedge, yedge),
                                     x_bins[bin_index], y_bins[bin_index],
                                     angle=0, edgecolor='None', alpha=alpha,
                                     facecolor=scaled_color)
            # Draw the rectangle
            plt.gca().add_patch(rect)
            # plt.scatter(yedge, xedge, c=bin_color)
        plt.draw()

        if colorbar:
            # If we want a colorbar we need to give the plotter something
            # to compare the color to. We create an arbitrary image to
            # anchor to the colorbar
            z = [[-100 for i in range(xnum)] for j in range(ynum)]
            levels = matplotlib.ticker.MaxNLocator(nbins=num_color).\
                tick_values(min_color, max_color)
            im = plt.contourf(z, levels=levels, cmap=cmap)

            # Create the colorbar
            CB = plt.colorbar(im)
            CB.set_label(cb_label, fontsize=18)
            CB.ax.tick_params(labelsize=16)

        # Set the plots limits
        min_xlim = min(x_corner)
        min_ylim = min(y_corner)
        max_xlim = max(x_corner) + max(x_bins)
        max_ylim = max(y_corner) + max(y_bins)
        plt.axis([min_xlim, max_xlim, min_ylim, max_ylim])

        # If it is a solo figure then draw the rest of the relevant properties
        if solo_fig:
            plt.xlabel(xvar, fontsize=18)
            plt.ylabel(yvar, fontsize=18)
            plt.title(title, fontsize=22)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)
        plt.savefig("random_progen.pdf")

    def compare_to_obs(self, obs_sys_file="ext_init_table.dat", N_lmxb=14):
        """
        This code loops through the previously generated random simulation
        table and compares the table to the observed LMXBs of interest

        inputs
        ------
        obs_sys_file : string
            The path to the table of observed LMXBs we're interested in
        N_lmxb : int
            The number of observed LMXBs that are in the table
        """
        # Load in the observed rate table given user input
        obs_data = np.loadtxt(obs_sys_file, delimiter=',')

        # split the data into mass, period and rates
        obs_mass = obs_data[:, 0]
        obs_period = obs_data[:, 2]
        obs_progen = np.column_stack((obs_period, obs_mass))
        obs_inits = obs_data[:, 7:]

        # split the random systems into sets
        split_per_sys = np.split(self.pers_sys, len(self.pers_sys) / N_lmxb)

        # create an empty array to populate
        num_obs = np.zeros((len(split_per_sys), 5))

        # generate an empty list to populate
        list_obs_inds = []

        # loop through the different observed LMXBs
        for iteration in range(len(split_per_sys)):
            sub_per_sys = split_per_sys[iteration][:, :-1]
            for system_ind in range(len(sub_per_sys)):

                # Check an individual system
                progen_system = sub_per_sys[system_ind]

                # see if any of our random systems match an observed system
                matches = np.all(progen_system == obs_progen, axis=1)
                ind_matches = np.where(matches)[0]

                # if it matches then bin it into the right category
                if ind_matches:
                    # print(iteration, ind_matches)
                    num_obs[iteration][0] += len(ind_matches)
                    list_obs_inds.append(ind_matches[0])

                    lmxb_list = obs_inits[ind_matches[0]]
                    lmxb_cat = np.where(lmxb_list != 0)[0]

                    ucxb_chk = set(lmxb_cat).issubset(range(0, 7))

                    short_chk = set(lmxb_cat).issubset(range(7, 11))

                    med_chk = set(lmxb_cat).issubset(range(11, 13))

                    long_chk = set(lmxb_cat).issubset(range(13, 14))

                    if ucxb_chk:
                        num_obs[iteration][1] += len(ind_matches)
                    if short_chk:
                        num_obs[iteration][2] += len(ind_matches)
                    if med_chk:
                        num_obs[iteration][3] += len(ind_matches)
                    if long_chk:
                        num_obs[iteration][4] += len(ind_matches)

        return num_obs

    def save_avg_lmxb_density(self, output_file="average_lmxb_density.dat"):
        try:
            np.savetxt(output_file, self.avg_density,
                       fmt="%.3f, %.3f, %.3f")
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.save_avg_lmxb_density()

    def load_avg_lmxb_density(self, input_file="average_lmxb_density.dat"):
        try:
            self.avg_density = np.loadtxt(input_file, delimiter=',')
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.load_avg_lmxb_density()

    def save_pers_sys(self, output_file="random_persistent_systems.dat"):
        try:
            np.savetxt(output_file, self.pers_sys,
                       fmt="%.3f, %.3f, %.0f")
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.save_avg_lmxb_density()

    def load_pers_sys(self, input_file="random_persistent_systems.dat"):
        try:
            self.pers_sys = np.loadtxt(input_file, delimiter=',')
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.load_avg_lmxb_density()

    def save_num_its(self, output_file="num_random_iterations.dat"):
        try:
            np.savetxt(output_file, self.num_its,
                       fmt="%.0f")
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.save_avg_lmxb_density()

    def load_num_its(self, input_file="num_random_iterations.dat"):
        try:
            self.num_its = np.loadtxt(input_file, delimiter=',')
        except AttributeError:
            self.gen_avg_lmxb_density()
            self.load_avg_lmxb_density()

    def _calc_bin_props(self, array):
        """
        Private routine that calculates the properties of the bins

        Inputs
        ------
        array : numpy array
            A 1xN array of values used to calculate the approximate
            bin sizes to have complete coverage in 1 dimension
        """

        # Find the distance from one data point to the next
        array_diff = np.diff(array)

        # Round the values to the third decimal place
        rounded_diff = np.around(array_diff, decimals=3)

        # Extend the distance array by 1 to match the total number of bins
        rounded_diff = np.append(rounded_diff, rounded_diff[-1])

        num_darray = np.unique(rounded_diff)
        if len(num_darray) == 1:
            dnew = rounded_diff
            lower_total = array - rounded_diff / 2

        else:
            # Our data has two step sizes, a fine and a coarse step size
            # The smaller values have a finer step size while the larger
            # values have a coarser step size
            dfine = rounded_diff[0]
            dcoarse = rounded_diff[-1]

            # The place where the difference between data points changes
            # is where our fine and coarse data begin/end
            split_cond = np.where(rounded_diff != dfine)[0][1]
            fine_vals, coarse_vals = np.split(array, [split_cond])

            # Calculate the boundaries of the bins with fine steps
            lower_fine = fine_vals - dfine / 2
            upper_fine = fine_vals + dfine / 2

            # Calculate the boundaries of the bins with coarse steps
            lower_coarse = coarse_vals - dcoarse / 2
            upper_coarse = coarse_vals + dcoarse / 2

            # This section is to avoid overlap or gaps at the boundary
            # between the coarse and fine step sizes
            if lower_coarse[0] > upper_fine[-1]:
                upper_fine[-1] = lower_coarse[0]
            elif lower_coarse[0] < upper_fine[-1]:
                lower_coarse[0] = upper_fine[-1]

            # Combine all of the data back together
            lower_total = np.concatenate((lower_fine, lower_coarse))
            upper_total = np.concatenate((upper_fine, upper_coarse))
            dnew = upper_total - lower_total

        return dnew, lower_total

    def _read_persistent_file(self, filename):

        persist_file = open(filename, 'r')
        persist_file.seek(0)

        # The data has a header section and a main body, we need
        # to properly format the output to work with it
        header_list = []
        persist_age = []

        # iterate through all of the lines in the data
        for line in persist_file:

            # take the line of data and split it into a list of floats
            temp_list = list(map(float, line.split()))

            # if the line contains more than a single number then it
            # is not part of the header
            if len(temp_list) == 1:
                header_list.append(temp_list[0])
            else:
                persist_age.append(temp_list)

        # convert the lists to numpy arrays
        header = np.asarray(header_list)
        persist_array = np.asarray(persist_age)

        persist_file.close()

        return header, persist_array
