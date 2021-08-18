import os
import copy
import numpy as np
import mesa_read as ms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import glob
import val_find


class progenitor_file():
    """
    Takes an input progenitor file and shifts the progenitor mass to plot
    the simulations

    Inputs
    ------
    dir_name : string
        Directory where the progenitor data is located, default
        is current working directory
    file_name : string
        the name of the progenitor file to be read in
    """
    def __init__(self, dir_name='./', file_name=''):
        self.dir_name = dir_name
        self.file_name = file_name

        # Combine the directory name and file name to produce
        # a complete path to the data file
        progen_file = dir_name + '/' + file_name
        self.progen_file = progen_file

        # Read in the file
        total_data = self._read_progen_file(progen_file)

        # Grab the data that we're going to use
        progen_data = total_data[:, :-1]

        # Store the relevant properties we're going to work with
        self.progen_data = progen_data
        self._mass_and_period_dirs()
        self._set_observed_props()

    def add_progen_file(self, dir_name='./', file_name=''):
        """
        Adds another progenitor file to calculate rates from in code

        Inputs
        ------
        dir_name : string
            Directory where the progenitor data is located, default
            is current working directory
        file_name : string
            The name of the additional file we want to read information from
        """

        # Check to see if we have information loaded already
        if len(self.progen_data):

            # Check to make sure the input is a string
            if type(file_name) == str:

                # Read in the file
                path_to_file = dir_name + '/' + file_name
                temp_total_data = self._read_progen_file(path_to_file)
                temp_progen_data = temp_total_data[:, :-1]

                # Combine the data into a large array
                temp_combined_data = np.concatenate((self.progen_data,
                                                     temp_progen_data), axis=0)

                # filter for only the unique data to avoid double counting
                temp_unique = self._filter_unique(temp_combined_data)

                # store the progenitor data
                self.progen_data = temp_unique

            # If the input is a list instead, loop through the list and rerun
            elif type(file_name) == list:
                for max_file in file_name:
                    self.add_progen_file(max_file)

        # If we dont have a file read in, read in the initial file first
        else:
            print("Reading in initial file...")
            self._read_progen_file()

    def set_path_to_sims(self, path):
        """
        Defines the path to our simulated files

        Inputs
        ------
        path : string
            The name of the file we're going to determine the path to

        """
        full_path = glob.glob("period*/group*/" + path, recursive=True)[0]
        return full_path + '/LOGS'

    def gen_age_persistent(self, directory="default_combined_dc/",
                           direction="lower"):
        """
        Outputs the amount of time the non observed systems spend as a
        persistent system

        Inputs
        ------
        directory : string
            path to directory where the matching duty cycle files are
        direction : string
            Defines if we want to get progenitors that are one step
            higher (upper) or smaller (lower) in mass. Default is upper
        """

        # shift our progenitor file masses
        shifted_data = self._shift_mass(direction=direction)

        # Ensure that our shifted data doesn't overlap with
        # our data prior to shifting
        unique_data = self._return_limits(shifted_data)

        # Once we have the unique data, grab the masses and peiods
        unique_mass = unique_data[:, 0]
        unique_period = unique_data[:, 2]

        # create an array to store the amount of time a simulation spends
        # in a persistent state
        persistent_array = np.zeros(len(unique_mass))

        # Grab the index of the progenitor file we're working with
        file_index = self.file_name.split("_")

        # Wrap the index in wild cards to make it easier to search
        file_to_search = "*" + file_index + "*"

        # Search the appropriate directory for the matching duty cycle file
        matching_files = glob.glob(directory + file_to_search)

        # read in the duty cycle data skipping the first few rows of data
        # as these rows contain metadata
        dc_data = np.loadtxt(matching_files[0], skiprows=8)

        # grab the period and masses inside the file to compare
        period_vals = dc_data[:, 0]
        mass_vals = dc_data[:, 1]

        # Loop through our masses and periods of interest to find matches
        # in the dc data to determine how much time a given system spends
        # in a persistent state
        for system_index in range(len(unique_mass)):

            # Get the mass and period we want to match
            sys_mass = unique_mass[system_index]
            sys_period = unique_period[system_index]

            # check for indices where the mass and period match
            mass_index = np.where(mass_vals == sys_mass)[0]
            period_index = np.where(period_vals == sys_period)[0]

            # Use python sets to find unique overlapping indices
            common_index = list(set(mass_index) & set(period_index))

            # Using the common index, grab the amount of time the system
            # spends appearing persistent
            persistent_array[system_index] = dc_data[common_index[0]][-2]

        # put the data together into a numpy array
        output_array = np.array([unique_period, unique_mass, persistent_array])

        # Save the array to a file
        np.savetxt("output_array.dat", output_array.T,
                   fmt="%1.3f, %1.3f, %1.0f")

    """
    ################################
    Plotting Functions
    ################################
    """

    def plot_many_sims(self, mass="", period="", direction="upper",
                       frequency=3, pt_freq=10, title="",
                       yvar=r'$\log_{10}$ (Period/days)',
                       xvar=r'Mass Ratio',
                       cb_label=r"$\log_{10}(M_\odot \rm yr^{-1})$",
                       min_color=-10, max_color=-6, num_color=9,
                       color_map='viridis', colorbar=True,
                       draw_obs=True, init_sys=0, fin_sys=14,
                       rasterize=True, define_lims=True,
                       obs_file="obs_bins.dat"):
        """
        A routine that loops through a list of masses and periods to plot
        the evolution of a given binary system. The colour of the track will
        be the mass transfer rate of the system at that point.

        inputs
        ------
        mass : list
            A list of the progenitor masses we're going to grab simulated
            data for
        period: list
            A list of the progenitor periods we're going to grab simulated
            data for
        direction : string
            Defines if we want to get progenitors that are one step
            higher (upper) or smaller (lower) in mass. Default is upper
        frequency : integer
            Defines how frequently we plot the simulations, the default is 3
            which means we plot every third simulation
        pt_freq : integer
            Defines how frequently we plot the data, the default is 10 so our
            plot will plot every tenth data point
        title : string
            Title of the plot produced
        cb_label : string
            Label associated with the colorbar
        min_color : float
            The lower limit of our colorbar used to show mass transfer rate,
            default is -10
        max_color : float
            The lower limit of our colorbar used to show mass transfer rate,
            default is -6
        num_color : int
            The number of colors between our max and min value, default is 9
        color_map : string
            The colormap used in the colorbar, default is viridis.
            See Matplotlib colorbar documentation for more choices:
            https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        colorbar : boolean
            True to display a colorbar, true by default
        draw_obs : boolean
            A boolean that defines if we want to draw the bins for our observed
            LMXBS. Default is True
        init_sys : int
            The index of the first observed LMXB we want to include on the
            figure. Default is 0
        fin_sys : int
            The index of the last observed LMXB we want to include on the
            figure. Default is 14
        define_lims : boolean
            This option set whether or not we want values below our range to
            be black or not
        """

        # Make sure the figure specifications are what we want, clear the
        # figure and ensure its the right size
        plt.close('all')
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 1, 1)

        # Define the colormap
        nbins = len(np.linspace(min_color, max_color, num_color)) - 1
        cmap = copy.copy(matplotlib.cm.get_cmap(color_map, nbins))

        if define_lims:
            cmap.set_under('black')

        # If we want to draw the bins for the observed LMXBs then enter loop
        if draw_obs:
            self._set_observed_props(obs_file)
            # Grab the observed periods and mass ratios
            obs_period = self.obs_period
            obs_mass_ratio = self.obs_mass_ratio

            # Grab the periods and mass ratios of the LMXBs we want to plot
            p_range = obs_period[init_sys:fin_sys + 1]
            q_range = obs_mass_ratio[init_sys:fin_sys + 1]

            # Loop through the LMXBs
            for unique_sys in range(len(p_range)):

                # Get the specific mass and period of a given system
                unique_p = p_range[unique_sys]
                unique_q = q_range[unique_sys]

                # Convert the period to log(days)
                p_list = np.log10(10**np.array(unique_p) / 24.)

                # Define the parameters of the box we're going to draw
                xy = (unique_q[0], p_list[0])
                xdist = abs(unique_q[1] - unique_q[0])
                ydist = abs(p_list[1] - p_list[0])

                # Create the box in red with no face color so it only
                # creates and outline
                rect = patches.Rectangle(xy, xdist, ydist,
                                         linewidth=1, edgecolor='r',
                                         facecolor='none')

                # Draw the box on the figure
                plt.gca().add_patch(rect)

        # If the user gives a specific list of masses and periods,
        # we assume the usre want to plot those masses and periods
        # so we don't shift the data
        if len(mass):

            # Grab the masses and period at a specific frequency
            mass = mass[::frequency]
            period = period[::frequency]

        # If the user doesn't provide a specific list of masses and
        # periods then we generate them from the files read in initially
        else:
            # Shift our masses either upwards or downward
            shifted_data = self._shift_mass(direction=direction)

            # Ensure that our shifted data doesn't overlap with
            # our data prior to shifting
            unique_data = self._return_limits(shifted_data)

            # Once we have the unique data, grab the masses and peiods
            unique_mass = unique_data[:, 0]
            unique_period = unique_data[:, 2]

            # Grab the masses and period at a specific frequency
            mass = unique_mass[::frequency]
            period = unique_period[::frequency]

        # Now that we've filtered our data, time to loop through them
        # all and plot each simulated set of masses and period
        for sim_ind in range(len(mass)):

            # Grab the specific mass and period
            sim_mass = mass[sim_ind]
            sim_period = period[sim_ind]

            # Get the path to the simulation
            sim_dir = self._gen_path_to_sim(sim_mass, sim_period)

            # Set the path to the simulation
            full_sim_dir = self.set_path_to_sims(sim_dir)

            # For debugging, print out the path to simulation
            # print(full_sim_dir)

            # Try to open the simulation, some combinations of mass
            # and period fail immediately so there wont be a simulated
            # data file produced, this should only result in an OSError.
            # Any other error is a problem and needs to be fixed.
            try:
                # Read in the simulated data with mesa_read
                m1 = ms.history_data(full_sim_dir)

                # Grab the mass transfer rate, period and masses at the
                # appropriate frequency
                mt = m1.get('lg_mtransfer_rate')[::pt_freq]
                period_days = m1.get('period_days')[::pt_freq]
                log_period = np.log10(period_days)

                mass_1 = m1.get('star_1_mass')[::pt_freq]
                mass_2 = m1.get('star_2_mass')[::pt_freq]
                mass_ratio = mass_1 / mass_2

                # Plot the simulated data
                plt.scatter(mass_ratio, log_period, c=mt,
                            vmin=min_color, vmax=max_color,
                            cmap=cmap, edgecolors='face',
                            marker=',', s=1, rasterized=rasterize)
            except OSError:
                pass

        # set the x and y axis label
        plt.xlabel(xvar, fontsize=18)
        plt.ylabel(yvar, fontsize=18)

        # Set the title of the plot
        plt.title(title, fontsize=22)

        # Create the colorbar
        try:
            cb = plt.colorbar()
            cb.set_label(cb_label, fontsize=18)
            cb.ax.tick_params(labelsize=16)
        except RuntimeError:
            pass

        # Increase the tick size of the x and y axis
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Reduce the margins of the figure
        plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)

    """
    ################################
    Private Functions
    ################################
    """

    def _read_progen_file(self, file_name=""):
        """
        Private routine that reads in the progenitor data
        Inputs
        ------
        file_name : string
            The name of the file to be read

        """

        # Make sure the file exists
        if not file_name:
            file_name = self.progen_file

        # If the file doesn't exist, just grab the first max.dat file
        # that glob finds
        if not os.path.isfile(file_name):
            print("Invalid file name, automatically grabbing a file")
            max_files = glob.glob("*max.dat")
            max_files.sort()
            self.progen_file = max_files[0]
            file_name = self.progen_file

        # Load in the data
        progen_data = np.loadtxt(file_name, delimiter=',')
        return progen_data

    def _mass_and_period_dirs(self):
        """
        Private routine that converts a given list of mass and period values
        into paths to appropriate matching simulations
        """

        # Find the path to ALL of the simulations
        path_to_sims = glob.glob("period*/group*/*M/*d", recursive=True)

        # Initialize lists for ALL mass and period directories
        total_mass_dirs = []
        total_period_dirs = []

        # Split the paths to find the mass and period values
        for sub_path in path_to_sims:
            split_path = sub_path.split("/")
            total_mass_dirs.append(split_path[2])
            total_period_dirs.append(split_path[-1])

        # Get the unique masses then sort
        mass_dirs = list(set(total_mass_dirs))
        mass_dirs.sort()

        # Get the unique periods then sort
        period_dirs = list(set(total_period_dirs))
        period_dirs.sort(key=val_find.natural_keys)

        # Initialze lists for specific mass and perido values
        mass_vals = []
        pers_vals = []

        # Pull the unit portion of the directory name and convert to array
        for mass in mass_dirs:
            mass_vals.append(float(mass[0:-1]))
        mass_vals = np.array(mass_vals)

        # Convert the directory strings to floats and convert to log values
        for pers in period_dirs:
            split_pers = pers.split("_")
            per_to_append = float(split_pers[0][0:-1])
            pers_vals.append([per_to_append,
                             round(np.log10(per_to_append / 1.00), 2)])

        # Sort the period values then convert to numpy array
        pers_vals.sort()
        pers_vals = np.array(pers_vals)

        # Store the mass and period directories and values
        self.mass_dirs = mass_dirs
        self.period_dirs = period_dirs

        self.mass_vals = mass_vals
        self.pers_vals = pers_vals

    def _set_observed_props(self, bin_file="obs_bins.dat"):
        """
        A private routine that sets the properties of the observed LMXBs
        """

        # Load in the file containing the observed properties
        obs_data = np.loadtxt(bin_file, delimiter=',', dtype=str)

        # Grab the names
        obs_names = obs_data[:, 0]

        # Grab the numerical properties of the LMXBs
        obs_props = obs_data[:, 1:].astype(float)

        # Splot the properties into individual lists
        obs_split = np.split(obs_props, 5, axis=1)
        obs_p, obs_q, obs_mt, obs_shift_mt, obs_temp = obs_split

        # Store the observed properties
        # Name
        self.obs_nams = obs_names

        # Period
        self.obs_period = obs_p

        # Mass Ratio
        self.obs_mass_ratio = obs_q

        # Mass transfer rate
        self.obs_mt = obs_mt

        # Shifted mass transfer rate
        self.obs_shifted_mt = obs_shift_mt

        # Effective temperature
        self.obs_temperature = obs_temp

    def _filter_unique(self, data):
        """
        A private routine that filters the input for unique values
        """
        try:
            filtered_list = [list(x) for x in set(tuple(x) for x in data)]
            filtered_array = np.array(filtered_list)
            return filtered_array
        except NameError:
            print("No list given to filter")

    def _shift_mass(self, direction):
        """
        Shifts an inputted list in a pre-defined direction,
        if "lower" then we shift the list one step down, if
        "upper" then we shift the list one step up.
        """

        # Make sure that we've read in progenitor data already
        if len(self.progen_data):

            # Get the mass, and mass step from our data
            progen_data = self.progen_data
            mass = progen_data[:, 0]
            dm = progen_data[:, 1]

            # For a specific step we need to make a slight
            # change, as this step size of 0.075 is the gap
            # for drawing, the previous/subsequent progen mass
            # is 0.05 in reality
            dm[np.where(dm == 0.075)] = 0.05

            # Depending on the input we either shift up or down
            if direction == "lower":
                print("shifting mass upwards")
                shifted_mass = np.round(mass - dm, 2)
            elif direction == "upper":
                print("shifting mass downwards")
                shifted_mass = np.round(mass + dm, 2)
            else:
                print("not shifting the mass")
                shifted_mass = np.round(mass, 2)

            # Append our results to an output array
            output_data = np.copy(progen_data)
            output_data[:, 0] = shifted_mass

        # If progenitor data hasn't been read in then read it in
        # first and retry code
        else:
            self._read_progen_file()
            self._shift_mass()
        return output_data

    def _return_limits(self, data):
        """
        A private routine that ensures that the data doesn't overlap
        with another array
        """
        progen_data = self.progen_data
        temp_data = np.copy(data)
        output_data = []

        for row in temp_data:
            if not any((row == x).all() for x in progen_data):
                output_data.append(row)
        if output_data:
            return np.array(output_data)
        else:
            return progen_data

    def _gen_path_to_sim(self, mass, period):
        """
        A private routine that combines strings together into a single
        path to a simulations output data
        """

        # print(mass)
        # print(period)

        # Gets the index where the mass and period match with a directory
        mass_ind = np.where(self.mass_vals == np.round(mass, 2))[0][0]
        pers_ind = np.where(self.pers_vals[:, 1] == np.round(period, 2))[0][0]
        # print(mass_ind)
        # print(pers_ind)

        # Gets the directory that matches with our mass and period
        mass_dir = self.mass_dirs[mass_ind]
        period_dir = self.period_dirs[pers_ind]

        # print(mass_dir)
        # print(period_dir)

        # combines together into a single path
        path_to_sim = mass_dir + '/' + period_dir
        return path_to_sim
