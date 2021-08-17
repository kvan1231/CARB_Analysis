import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import mesa_read as ms
import pandas as pd
import val_find


class density_profile():
    """
    Read in some density data produced from MESA simulations

    Inputs
    ------
    dir_name : string
        Directory where the density data is located, default
        is current working directory
    file_name : string
        the name of the density file to be read in
    """
    def __init__(self, dir_name="./", file_name=""):
        self.dir_name = dir_name
        self.file_name = file_name

        # Combine the directory name and file name to produce
        # a complete path to the data file
        density_file = dir_name + '/' + file_name

        # Check to make sure the file actually exists
        if not os.path.exists(density_file):
            raise IOError(density_file + " does not exist")
        else:
            # Read in the data
            header, density_data = self._read_density()

        # Store the relevant properties to be used
        self.header = header
        self.density_data = density_data
        self.xvalues = density_data[:, 0]
        self.yvalues = density_data[:, 1]
        self.color_vals = density_data[:, 2:]

    def get_density(self):
        """
        Return the read in density data file
        """

        density_data = self.density_data
        return density_data

    def get_bin_sizes(self, bin_file=""):
        """
        Find the corner of each bin and the size of the bins to
        be plotted. The x and y values of the density file are
        the initial conditions of the system and are not necessarily
        the limits of the plotted bin.

        Inputs
        ------
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
        """

        # Check to make sure the file actually exists
        if not os.path.exists(bin_file):

            # If the file doesn't exist then we need to calculate the values
            print("no bin size file found, calculating...")

            # Shorten our x and y columns to only have the unique values
            unique_xvals = np.unique(self.xvalues)
            unique_yvals = np.unique(self.yvalues)

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

        else:
            # If the file exists then just open the file and return the values
            print("bin size file given, opening " + bin_file)
            x_bins, y_bins, x_corner, y_corner = self._read_bin_props(bin_file)

        return x_bins, y_bins, x_corner, y_corner

    def get_crit_data(self, crit_file=""):
        """
        A routine that generates the critial mass transfer rate value that
        separates persistent and transient binaries using a disc instability
        model. If a file exists that contains the values it is read in, if
        not then the code will calculate the values.

        inputs
        ------
        crit_file : string
            The path to the file containing the data
        """

        # Check to make sure the file actually exists
        if not os.path.exists(crit_file):

            # If the file doesn't exist then we need to calculate the values
            print("no crit file found, calculating...")

            # Grab the unique period values from our density file
            logP = np.unique(self.xvalues)

            # Convert the log values to linear values
            linear_p = 10 ** logP

            # Calculate the various mass transfer values
            mc_niu, mc_nil, mc_iu, mc_il = self._calc_mdot_crit(linear_p)

        else:
            # logP is the period of the binary system
            # mc denotes the critical mass transfer rate
            # ni denotes that the model used assumes no irradiation
            # i denotes that the model uses irradiation
            # l and u denote the lower and upper limits of the model
            crit_data = self._read_crit_file(crit_file)
            logP, mc_niu, mc_nil, mc_iu, mc_il = crit_data

        # Calculate an average mass transfer value for each case
        mc_ni_avg = (np.array(mc_niu) + np.array(mc_nil)) / 2
        mc_i_avg = (np.array(mc_iu) + np.array(mc_il)) / 2

        return logP, mc_ni_avg, mc_i_avg

    def get_min_period(self, min_period_file="min_mass.txt"):
        """
        A routine that generates the minimum period that results in mass
        transfer. If a file exists that contains these values then it is read
        in, if not then the code will try to generate the appropriate file
        containing the periods.

        inputs
        ------
        min_period_file : string
            The path to the file containing the minimum periods
        """

        # Check to make sure the file actually exists:
        if not os.path.exists(min_period_file):

            # If the file doesn't exist then we need to generate it
            print("minimum period file not found, generating...")
            min_data = self._gen_min_period()

        else:
            # read in the data
            min_data = self._read_min_period(min_period_file)

        # split the data into mass and period
        min_mass, min_period, min_period_non_init = min_data
        return min_mass, min_period, min_period_non_init

    """
    ################################
    Private Functions
    ################################
    """

    def _read_density(self):

        """
        Private routine that reads in the density data
        """

        # open the file using the directory and file name provided
        file_name = self.dir_name + self.file_name
        density_file = open(file_name, "r")
        density_file.seek(0)

        # The data has a header section and a main body, we need
        # to properly format the output to work with it
        header_list = []
        density_list = []

        # iterate through all of the lines in the data
        for line in density_file:

            # take the line of data and split it into a list of floats
            temp_list = list(map(float, line.split()))

            # if the line contains more than a single number then it
            # is not part of the header
            if len(temp_list) == 1:
                header_list.append(temp_list[0])
            else:
                density_list.append(temp_list)

        # convert the lists to numpy arrays
        header = np.asarray(header_list)
        density_data = np.asarray(density_list)

        density_file.close()

        return header, density_data

    def _read_bin_props(self, file_name):
        """
        Private routine that reads in the properties of the bins
        """
        bin_sizes = np.loadtxt(file_name, delimiter=',')

        # The bin data includes columns we dont need so we'll only
        # grab the bins of interest
        x_bins = bin_sizes[:, 4]
        y_bins = bin_sizes[:, 7]
        x_corner = bin_sizes[:, 2]
        y_corner = bin_sizes[:, 5]

        return x_bins, y_bins, x_corner, y_corner

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

    def _read_crit_file(self, crit_file):
        """
        This is a private routine that grabs the critical mass tranfer
        values for a given period using a DIM model

        inputs
        ------
        crit_file : string
            The name of the file that contains the critical mass transfer
            values
        """
        crit_data = np.loadtxt(crit_file)
        return crit_data.T

    def _calc_mdot_crit(self, p_days):
        '''
        Calculate a critical mass transfer rate for irradiated accretion discs.
        https://ui.adsabs.harvard.edu/abs/2012MNRAS.424.1991C/abstract

        inputs
        ----------
        p_days : array
            orbital period of the binary system in days

        Returns
        -------
        mdot_crit : float
            the critical mass transfer rate for the given inputs
            in solar masses per year
        '''
        # convert orbital period from days to hours
        p_hr = p_days * 24

        # Use critical mass transfer rate from Coriat et al. 2012
        # Assume C=2e-5, converted the original equation from g/s to M_sun/yr.
        # mdot_crit = 3.15e-11 * (m1 ** (0.5)) / (m2 ** (0.2)) * p_h ** 1.4

        # Use critical mass transfer rate from Dubus et al. 2012
        # Assume alpha = 0.1, and C = 1e-3
        # non irradiated denoted by ni, irradiated denoted by i

        b_ni = 1.76
        b_i = 1.59

        # upper and lower limits on k when non irradiated
        k_u_ni = 3.5e16
        k_l_ni = 1.7e16

        # convert grams per second to solar masses per year
        mass_conv = 1.6e-26

        mdot_crit_u_ni = np.log10((k_u_ni * p_hr ** b_ni) * mass_conv)
        mdot_crit_l_ni = np.log10((k_l_ni * p_hr ** b_ni) * mass_conv)

        # upper and lower limits on k when irradiated
        k_u_i = 3.8e15
        k_l_i = 2.0e15

        mdot_crit_u_i = np.log10((k_u_i * p_hr ** b_i) * mass_conv)
        mdot_crit_l_i = np.log10((k_l_i * p_hr ** b_i) * mass_conv)

        return mdot_crit_u_ni, mdot_crit_l_ni, mdot_crit_u_i, mdot_crit_l_i

    def _read_min_period(self, min_period_file):
        """
        This is a private routine that grabs the minimum period for the onset
        of mass transfer at a given mass.

        inputs
        ------
        min_period_file : string
            The name of the file that contains the critical mass transfer
            values
        """
        min_data = np.loadtxt(min_period_file, delimiter=',')
        return min_data.T

    def _gen_min_period(self):
        """
        This is a private routine that finds the minimum period for the onset
        of mass transfer at a given mass and saves it to a file.

        """

        # Grab the mass and period values
        masses = np.array(list(set(self.yvalues)))
        masses.sort()
        min_pers = np.zeros(len(masses))
        min_pers_non_init = np.zeros(len(masses))

        # loop through all of our mass values using indices
        # we need the index to populate the minimum period array
        for mass_ind in range(len(masses)):

            # get the appropriate mass value
            mass = masses[mass_ind]

            # using glob, generate the path to all simulations with that mass
            full_path = glob.glob("period*/group*/" + str(mass) + '*/*')

            # sort the paths using a natural key, or in a human readable way
            full_path.sort(key=val_find.natural_keys)

            # split all of the paths into two subsets for reordering
            fine_dirs = [s for s in full_path if "fine" in s]
            coarse_dirs = [s for s in full_path if "coarse" in s]

            # reorder the two subset of paths
            reordered_paths = fine_dirs + coarse_dirs

            min_period_check = False

            # loop through all of the paths
            for sim_path in reordered_paths:
                # wrap our data reading inside a try-except statement
                # we do not check if the file exists, if it doesnt it should
                # return an OSError which is expected
                try:
                    # read in the data of a given simulation
                    m1 = ms.history_data(sim_path + '/LOGS')

                    # grab the mass transfer rate to check if RLOF occurs
                    mt = m1.get('lg_mtransfer_rate')

                    # find where the mass transfer exceeds -12, this is a
                    # strict lower limit where anything less than this is
                    # effectively no mass transfer
                    rlof_check = np.where(mt > -12)[0]

                    # if mass transfer occurs in this system, enter here
                    if len(rlof_check):
                        # Output to terminal that the min was found
                        # print("MIN PERIOD FOUND")

                        # Output to terminal what that simulation was
                        print(sim_path)

                        # Grab the period of the simulation
                        period_str = sim_path.split("M/")[-1]

                        # Convert that period to a float
                        period_float = float(period_str[:-1])

                        # If we haven't saved an initial minimum period yet,
                        # save it to our minimum period array
                        if not min_period_check:
                            min_pers[mass_ind] = period_float
                            min_period_check = True

                        # If the simulation hasn't undergone RLOF at the first
                        # step save it to minimum period non intial mass
                        # transfer array
                        if 0 not in rlof_check:
                            min_pers_non_init[mass_ind - 1] = period_float

                            # break the loop, we no longer need to look at
                            # longer periods as the minimum period was found
                            break
                except OSError:
                    pass

        # create an output array with our masses and minimum periods
        output_array = np.array([masses, min_pers, min_pers_non_init]).T

        # save the mass and period to a file
        np.savetxt("min_mass.txt", output_array, fmt="%1.3f,  %1.3f,  %1.3f")
        # return masses, min_pers, min_pers_non_init

    def _gen_max_period(self):
        """
        This is a private routine that finds the minimum period for the onset
        of mass transfer at a given mass and saves it to a file.

        """

        # Grab the mass and period values
        masses = np.array(list(set(self.yvalues)))
        masses.sort()
        max_pers = np.zeros(len(masses))

        # loop through all of our mass values using indices
        # we need the index to populate the minimum period array
        for mass_ind in range(len(masses)):

            # get the appropriate mass value
            mass = masses[mass_ind]

            # using glob, generate the path to all simulations with that mass
            full_path = glob.glob("period*/group*/" + str(mass) + '*/*')

            # sort the paths using a natural key, or in a human readable way
            full_path.sort(key=val_find.natural_keys)

            # split all of the paths into two subsets for reordering
            fine_dirs = [s for s in full_path if "fine" in s]
            coarse_dirs = [s for s in full_path if "coarse" in s]

            # reorder the two subset of paths
            reordered_paths = fine_dirs + coarse_dirs
            reordered_paths = reordered_paths[::-1]

            # loop through all of the paths
            for sim_path in reordered_paths:
                # wrap our data reading inside a try-except statement
                # we do not check if the file exists, if it doesnt it should
                # return an OSError which is expected
                try:
                    # read in the data of a given simulation
                    m1 = ms.history_data(sim_path + '/LOGS')

                    # grab the mass transfer rate to check if RLOF occurs
                    mt = m1.get('lg_mtransfer_rate')

                    # find where the mass transfer exceeds -12, this is a
                    # strict lower limit where anything less than this is
                    # effectively no mass transfer
                    rlof_check = np.where(mt > -12)[0]

                    # if mass transfer occurs in this system, enter here
                    if len(rlof_check):
                        # Output to terminal that the min was found
                        # print("MIN PERIOD FOUND")

                        # Output to terminal what that simulation was
                        print(sim_path)

                        # Grab the period of the simulation
                        period_str = sim_path.split("M/")[-1]

                        # Convert that period to a float
                        period_float = float(period_str[:-1])
                        max_pers[mass_ind] = period_float

                        break
                except OSError:
                    pass

        # create an output array with our masses and minimum periods
        output_array = np.array([masses, max_pers]).T

        # save the mass and period to a file
        np.savetxt("max_mass.txt", output_array, fmt="%1.3f,  %1.3f")
        # return masses, min_pers, min_pers_non_init

    """
    ################################
    Plotting Functions
    ################################
    """

    def plot_density(self, bin_file="",
                     zvalue="dc", zscale='log10', alpha=1.0,
                     max_color=0.0, min_color=-12, num_color=25,
                     color_map='viridis', colorbar=True,
                     xvar=r'$\log_{10}$ (Period/days)',
                     yvar=r'Mass ($M_\odot$)', title="",
                     cb_label=r"$\log_{10}(f_{\rm obs})$",
                     draw_min_period=True, solo_fig=True):
        """
        A routine that produces a density plot using the density file given
        with the color showing different properties based on user choice.

        inputs
        ------
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
        zvalue : string
            Defines what property to plot as the color of the bin, default
            is the duty cycle of the progenitor
        zscale : string
            Defines if we want the colorbar to be logarithmic or linear,
            default is base 10 logarithmic
        max_color : float
            Defines the maximum value in the colorbar, default is 0
        min_color : float
            Defines the minimum value in the colorbar, default is -12
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
        draw_min_period : boolean
            True or false to determine if we want to draw a minimum period line
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

        # Grab the density data
        density_data = self.get_density()
        xvalues = density_data[:, 0]
        xmin = min(xvalues)
        xmax = max(xvalues)

        yvalues = density_data[:, 1]
        ymin = min(yvalues)
        ymax = max(yvalues)

        xnum = len(np.unique(xvalues))
        ynum = len(np.unique(yvalues))

        color_vals = density_data[:, 2:]

        # Define the colormap
        cmap = plt.get_cmap(color_map)
        cNorm = matplotlib.colors.Normalize(vmin=min_color,
                                            vmax=max_color)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        # Depending on choice, grab the appropriate property used for color
        if zvalue == 'dc':
            rect_color = color_vals[:, 0]

            # If any simulated systems match with an observed LMXB
            # but dont qualify to be persistent, then it may have
            # a value > 1, set these equal to 0.9
            rect_color[rect_color >= 1.0] = 0.9
        if zvalue == 'oa':
            rect_color = color_vals[:, 1]
        if zvalue == 'da':
            rect_color = color_vals[:, 2]

        # If using log scale then convert values to log base 10
        if zscale == 'log10':
            rect_color = np.log10(rect_color)

            # If any values were originally 0 then they would be
            # -inf once in log form, replace these with a large
            # negative number
            rect_color[rect_color == -np.inf] = -20.

        # Grab the bin properties
        x_bins, y_bins, x_corner, y_corner = self.get_bin_sizes(bin_file)

        # If the bin color is outside our colorbar range then ignore it
        color_check = np.where((rect_color >= min_color) &
                               (rect_color <= max_color))

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
        plt.axis([xmin, xmax, ymin, ymax])

        if draw_min_period:
            y_line, x_line1, x_line2 = self.get_min_period()
            plt.plot(np.log10(x_line1), y_line, 'r--')
            plt.plot(np.log10(x_line2), y_line, 'k--')

        # If it is a solo figure then draw the rest of the relevant properties
        if solo_fig:
            plt.xlabel(xvar, fontsize=18)
            plt.ylabel(yvar, fontsize=18)
            plt.title(title, fontsize=22)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)

    def plot_class(self, bin_file="",
                   zvalue="dc", zscale='log10', alpha=1.0,
                   max_color=0.0, min_color=-12,
                   color='Red', hatch=None,
                   xvar=r'$\log_{10}$ (Period/days)',
                   yvar=r'Mass ($M_\odot$)', title="",
                   draw_min_period=True, solo_fig=True):
        """
        A routine that produces a density plot using the density file given
        with the color showing different properties based on user choice.

        inputs
        ------
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
        zvalue : string
            Defines what property to plot as the color of the bin, default
            is the duty cycle of the progenitor
        zscale : string
            Defines if we want the colorbar to be logarithmic or linear,
            default is base 10 logarithmic
        max_color : float
            Defines the maximum value in the colorbar, default is 0
        min_color : float
            Defines the minimum value in the colorbar, default is -12
        xvar : string
            Label for the x-axis, default label is log10(period)
        yvar : string
            Label for the y-axis, default label is Mass
        title : string
            Title of the plot produced
        draw_min_period : boolean
            True or false to determine if we want to draw a minimum period line
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

        # Grab the density data
        density_data = self.get_density()
        xvalues = density_data[:, 0]
        xmin = min(xvalues)
        xmax = max(xvalues)

        yvalues = density_data[:, 1]
        ymin = min(yvalues)
        ymax = max(yvalues)

        color_vals = density_data[:, 2:]

        # Depending on choice, grab the appropriate property used for color
        if zvalue == 'dc':
            rect_color = color_vals[:, 0]

            # If any simulated systems match with an observed LMXB
            # but dont qualify to be persistent, then it may have
            # a value > 1, set these equal to 0.9
            rect_color[rect_color >= 1.0] = 0.9

        if zvalue == 'oa':
            rect_color = color_vals[:, 1]
        if zvalue == 'da':
            rect_color = color_vals[:, 2]

        # If using log scale then convert values to log base 10
        if zscale == 'log10':
            rect_color = np.log10(rect_color)

            # If any values were originally 0 then they would be
            # -inf once in log form, replace these with a large
            # negative number
            rect_color[rect_color == -np.inf] = -20.

        # Grab the bin properties
        x_bins, y_bins, x_corner, y_corner = self.get_bin_sizes(bin_file)

        # If the bin color is outside our colorbar range then ignore it
        color_check = np.where((rect_color >= min_color) &
                               (rect_color <= max_color))

        # Loop through the indicies where the color is in the appropriate range
        for bin_index in color_check[0]:
            xedge = x_corner[bin_index]
            yedge = y_corner[bin_index]

            # Create the rectangle
            rect = patches.Rectangle((xedge, yedge),
                                     x_bins[bin_index], y_bins[bin_index],
                                     angle=0, edgecolor='None', alpha=alpha,
                                     facecolor=color, hatch=hatch)
            # Draw the rectangle
            plt.gca().add_patch(rect)
        plt.draw()

        # Set the plots limits
        plt.axis([xmin, xmax, ymin, ymax])

        if draw_min_period:
            y_line, x_line1, x_line2 = self.get_min_period()
            plt.plot(np.log10(x_line1), y_line, 'r--')
            plt.plot(np.log10(x_line2), y_line, 'k--')

        # If it is a solo figure then draw the rest of the relevant properties
        if solo_fig:
            plt.xlabel(xvar, fontsize=18)
            plt.ylabel(yvar, fontsize=18)
            plt.title(title, fontsize=22)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)

    def plot_crit(self, crit_file="", period_hr=False):
        """
        A routine that plots the critical mass transfer rate at a given period
        that separates if a binary system is predicted to be persistent or
        transient using a disc instability model (DIM).
        https://ui.adsabs.harvard.edu/abs/2012MNRAS.424.1991C/abstract

        The critical mass transfer rate lines should be an addition to a plot
        to show the separation between types of binaries and should not be
        plotted alone.

        inputs
        ------
        crit_file : string
            path to the file containing the critical mass transfer rates
        period_hr : boolean
            A flag to determine if we want our period values to be in hours,
            by default set to False so period is in days
        """

        # Get the critical mass transfer values
        logP, mc_ni_avg, mc_i_avg = self.get_crit_data(crit_file)

        # Convert the period from days to hours
        if period_hr:
            logP = np.log10(10**np.array(logP) * 24)

        # Plot the values
        plt.plot(logP, mc_ni_avg, 'k--', lw=5, alpha=0.5)
        plt.plot(logP, mc_i_avg, 'k:', lw=5, alpha=0.5)

    def plot_obs(self,
                 obs_file="/media/kenny/seagate10TB/data_tables/NS_LMXB.csv",
                 marker_size=8, Sco_star=False, period_hr=False,
                 show_edd=True):
        """
        A routine that plots our observed sample of LMXBs. Our sample
        is taken from:
            https://ui.adsabs.harvard.edu/abs/2019ApJ...886L..31V/abstract.

        The observed data points should include errors in the periods and mass
        transfe rates but in some cases the errors are small enough that they
        are not visible under the data point. The observed data points should
        be an addition to a plot to show how the simulated data compares to
        observed systems and should not be plotted alone.

        Inputs
        ------
        obs_file : string
            path to the file containing the observed data which should include
            period and mass transfer rate
        marker_size : float
            The marker size used to plot the data, default is 8
        Sco_star : boolean
            A flag to determine if the user wants to plot the observed system
            Sco X-1 with a special data point. Default is False so the system
            is plotted with the same marker as other points
        period_hr : boolean
            A flag to determine if we want our period values to be in hours,
            by default this is set to False so our period is in units of days
        show_edd : boolean
            A flag to determine if we want to plot a dashed line showing the
            Eddington limit of a Neutron star. Default is True
        """

        # Initialize the scatter plot marker size
        marker_size = marker_size

        # Define the columns used in the pandas file being read in
        cols = ['bin_type',
                'q',
                'period_hours',
                'period_error',
                'mass_transfer',
                'mass_xfer_pos',
                'mass_xfer_neg',
                ]

        # Read in the data using pandas, its a csv file so we use
        # the comma separator
        NS_LMXB = pd.read_csv(obs_file, sep=',', usecols=cols)

        # Split the read in data into columns defined by cols
        bin_type_tot = NS_LMXB.bin_type.values
        period_tot = NS_LMXB.period_hours.values
        period_error_tot = NS_LMXB.period_error.values
        mass_transfer_tot = NS_LMXB.mass_transfer.values
        mass_xfer_pos_tot = NS_LMXB.mass_xfer_pos.values
        mass_xfer_neg_tot = NS_LMXB.mass_xfer_neg.values
        q_tot = NS_LMXB.q.values

        # Remove any systems that dont have a value for mass ratio
        # or are labelled as persistent
        mass_ratio_chk = np.where(q_tot != '--')[0]
        pers_sys_chk = np.where(bin_type_tot != 'T')[0]
        valid_inds = list(set(mass_ratio_chk) & set(pers_sys_chk))
        valid_inds.sort()

        bin_type = bin_type_tot[valid_inds]
        period = period_tot[valid_inds]
        period_error = period_error_tot[valid_inds]
        mass_transfer = mass_transfer_tot[valid_inds]
        mass_xfer_pos = mass_xfer_pos_tot[valid_inds]
        mass_xfer_neg = mass_xfer_neg_tot[valid_inds]

        # loop through the observed LMXBS
        for systen_index in range(len(period)):

            # Set the error bars for mass transfer
            mdot_up = float(mass_xfer_pos[systen_index])
            mdot_lo = float(mass_xfer_neg[systen_index])

            # Replace fields where we dont have error with 0
            if period_error[systen_index] == '--':
                period_error[systen_index] = 0.0

            # Check if our period is in units of hours or days and convert
            # if necessary
            if not period_hr:
                d_period = float(period_error[systen_index])
                period_val = float(period[systen_index])
            else:
                d_period = float(period_error[systen_index]) * 24
                period_val = float(period[systen_index]) * 24

            # Define the mass transfer rate
            mdot = float(mass_transfer[systen_index])

            # Only grab data that is defined as persistent
            if bin_type[systen_index] == "P":

                # Draw the error in mass transfer rate
                plt.vlines(np.log10(period_val), np.log10(mdot - mdot_lo),
                           np.log10(mdot + mdot_up), zorder=20,
                           colors='k')

                # Draw the error in period
                plt.hlines(np.log10(mdot), np.log10(period_val - d_period),
                           np.log10(period_val + d_period), zorder=20,
                           colors='k')

                # Draw the data point
                plt.scatter(np.log10(period_val), np.log10(mdot),
                            c='w', s=marker_size,
                            marker="^", edgecolors='k',
                            vmin=0, vmax=0.2, zorder=20)

            # Sco X-1 has a special label, if we encounter this label and
            # the user doesn't want to use a special marker for Sco X-1
            # then enter here
            elif bin_type[systen_index] == "S" and not Sco_star:

                # Draw the error in mass transfer rate
                plt.vlines(np.log10(period_val), np.log10(mdot - mdot_lo),
                           np.log10(mdot + mdot_up), zorder=20,
                           colors='k')

                # Draw the error in period
                plt.hlines(np.log10(mdot), np.log10(period_val - d_period),
                           np.log10(period_val + d_period), zorder=20,
                           colors='k')

                # Draw the data point
                plt.scatter(np.log10(period_val), np.log10(mdot),
                            c='w', s=marker_size,
                            marker="^", edgecolors='k',
                            vmin=0, vmax=0.2, zorder=20)

            # If the user wants a special marker for Sco X-1 then enter
            # here
            if Sco_star:
                if bin_type[systen_index] == "S":

                    # Draw the error in mass transfer rate
                    plt.vlines(np.log10(period_val),
                               np.log10(mdot - mdot_lo),
                               np.log10(mdot + mdot_up),
                               zorder=20, colors='k')

                    # Draw the error in period
                    plt.hlines(np.log10(mdot),
                               np.log10(period_val - d_period),
                               np.log10(period_val + d_period),
                               zorder=20, colors='k')

                    # Draw the data point
                    plt.scatter(np.log10(period_val), np.log10(mdot),
                                c='y', s=marker_size * 2,
                                marker="*", edgecolors='k', zorder=20)

        # Check if we want to plot the line showing the Eddington limit
        if show_edd:

            # If the period is in hours then define the limits here
            if period_hr:
                min_period = -2.5
                max_period = 4.0
                text_pos = -1.9
                UCXB_per = np.log10(80.0 / (60.0 * 24))

            # Define the range in period  we want our line to span
            # If the period is in days then enter here
            else:
                min_period = np.log10(10**-2.5 * 24)
                max_period = np.log10(10**4.0 * 24)
                text_pos = np.log10(10**-1.9 * 24)
                UCXB_per = np.log10(80.0 / 60.0)

            # Generate a line that spans our period range
            edd_periods = np.linspace(min_period, max_period, 1000)

            # calculated taking Medd = 4 pi c R/K
            # R = 11.5km, K = 0.2*(1+X), with X = 0.7
            mt_edd = np.log10(2.021E-8 * np.ones(len(edd_periods)))

            # Draw the text above the line
            plt.text(text_pos, -7.6, 'Eddington Limit', color='r')

            # Plot the line
            plt.plot(edd_periods, mt_edd, 'r--')

        # Draw a hased area showing where ultra-compact systems reside
        plt.axvspan(-2.5, UCXB_per, facecolor=None, fill=False, hatch='/',
                    lw=0.5, alpha=1.0, ls='dashed')
