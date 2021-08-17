import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import mesa_read as ms


class rate_table():
    """
    Handle the calculated values from NDensity.py to generate
    formation rate predictions for the progenitors

    Inputs
    ------
    dir_name : string
        Directory where the observed ratio data is located,
        default is current working directory
    file_suffix : string
        A common suffix for all of the observed ratio files that
        we will be working with. Default is "*dc.dat". Remember
        to include the wild card * in your file_suffix.
    """
    def __init__(self, dir_name='./', file_suffix="*dc.dat"):
        self.dir_name = dir_name
        self.file_suffix = file_suffix

        # Combine the directory name and the suffix to produce
        # a string that glob can use to search for files
        path_to_files = dir_name + file_suffix
        file_list = glob.glob(path_to_files)

        # Sort the files to ensure they're in order
        file_list.sort()

        # Store the list of files we will be working with
        self.file_list = file_list

    def get_file_list(self):
        """
        Returns the list of observed ratio files
        """
        return self.file_list

    def set_col_names(self, col_names=""):
        """
        Sets the column names to be used in a pandas dataframe

        Inputs
        ------
        col_names : list
            A list of strings that respresent the column names in the
            pandas dataframe

        The default column names represent the following
            mass     : the initial mass of that cell
            dm       : the width in mass of that cell
            period   : the initial period of that cell
            dp       : the width in period of that cell
            det_time : the amount of time a system is persistent
            obs_time : the amount of time a system matches with an obs system
            ratio    : the ratio between det_time and obs_time
            oa<n>    : the observed time for that specific obs system
        """

        # If a list of column names is given use that instead
        if col_names:
            self.col_names = col_names

        # Otherwise just use the default column names
        else:
            print("Using default col_names")
            col_names = ["mass", "dm", "period", "dp",
                         "det_time", "obs_time", "ratio",
                         "oa00", "oa01", "oa02", "oa03", "oa04", "oa05",
                         "oa06", "oa07", "oa08", "oa09", "oa10", "oa11",
                         "oa12", "oa13", "combined_UXCB"]
            self.col_names = col_names

    def set_sys_names(self, sys_names=""):
        """
        Defines the names of the different observed LMXBs

        Inputs
        ------
        sys_names : list
            A list of strings that are the names of the observed LMXBs
        """

        # If a list of system names is given use that instead
        if sys_names:
            self.sys_names = sys_names

        # Otherwise just use the default system names
        else:
            print("Using default sys_names")
            sys_names = ["4U 0513-40",
                         "2S 0918-549",
                         "4U 1543-624",
                         "4U 1850-087",
                         "M15 X-2",
                         "4U 1626-67",
                         "4U 1916-053",
                         "4U 1636-536",
                         "GX 9+9",
                         "4U 1735-444",
                         "2A 1822-371",
                         "Sco X-1",
                         "GX 349+2",
                         "Cyg X-2",
                         "Combined UCXB"]
            self.sys_names = sys_names

    def get_bin_sizes(self, period, mass, bin_file=""):
        """
        Find the corner of each bin and the size of the bins to
        be plotted. The x and y values of the density file are
        the initial conditions of the system and are not necessarily
        the limits of the plotted bin.

        Inputs
        ------
        period : array
            The period values that we are looking for the bin sizes of.
            If a bin_file is provided this array is not necessary.
        mass : array
            The mass values that we are looking for the bin sizes of.
            If a bin_file is provided this array is not necessary.
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
        """

        # Check to make sure the file actually exists
        if not os.path.exists(bin_file):

            # If the file doesn't exist then we need to calculate the values
            print("no bin size file found, calculating...")

            # Shorten our x and y columns to only have the unique values
            unique_xvals = np.unique(period)
            unique_yvals = np.unique(mass)

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

    def table_gen(self, bin_file=""):
        """
        Generate a table of data from the observed ratio files grabbed during
        initialization

        Inputs
        ------
        bin_file : string
            The path to the file containing the sizes of the bins. If no
            file is provided the code will calculate the bin sizes.
        """

        print("generating table...")

        # Grab the list of observed ratio files we are going to loop through
        file_list = self.file_list

        # Loop through all of the files
        num_files = len(file_list)
        for file_index in range(num_files):

            # Read in the data with our private read function
            file_name = file_list[file_index]
            print(file_name)
            _, file_data = self._read_dc(file_name)

            # If this is the first loop, then generate the output array
            if file_index == 0:
                results = file_data

            # All subsequent loops will only append to the output array
            else:

                # Generate a new array that is one column wider than before
                columns = len(results[0])
                rows = len(results)
                temp_results = np.zeros((rows, columns + 1))

                # Populate the n-1 columns with the previous results
                temp_results[:, 0:columns] = results

                # Append the new results to the last column
                temp_results[:, -1] = file_data[:, 3]

                # Replace the old output array with the new wider one
                results = temp_results

        # If progenitor does not produce an observed LMXB then a value of -20
        # will be found in that row, we want to find all the non-detections
        non_detect_ind = np.where(results < -19.5)

        # Set these values be equal to 0
        results[non_detect_ind] = 0

        # Find all of the rows where any observed LMXB is detected
        # This is done by summing the values in the row excluding
        # the mass/period inital properties, if the value is non-zero
        # then that progenitor mass/period produces at least one observed
        # LMXB.
        detection_inds = np.where(np.sum(results[:, 3:], axis=1) > 0)[0]
        trimmed_results = results[detection_inds]

        # Find where the trimmed array is 0 and flip it back to -20
        # We are going to convert from log back to linear values to find
        # the total observed age
        flipped_inds = np.where(trimmed_results == 0)
        trimmed_results[flipped_inds] = -20.0

        # Sum the observed times found
        summed_oa = np.sum(10**trimmed_results[:, 3:], axis=1)

        # Take the ratio of the observed and detectable ages
        ratio = summed_oa / 10**trimmed_results[:, 2]

        # Grab the mass and period
        mass = results[:, 0]
        period = results[:, 1]

        # Calculate the bin sizes
        dp, dm, _, _ = self.get_bin_sizes(period, mass, bin_file)

        # Create the array that holds our final results
        complete_results = np.zeros((len(trimmed_results),
                                     len(trimmed_results[0]) + 4))

        # Populate the array
        complete_results[:, 0] = mass[detection_inds]        # mass
        complete_results[:, 1] = dm[detection_inds]          # dm
        complete_results[:, 2] = period[detection_inds]      # Period
        complete_results[:, 3] = dp[detection_inds]          # dp
        complete_results[:, 4] = trimmed_results[:, 2]       # detectable time
        complete_results[:, 5] = np.log10(summed_oa)         # total obs time
        complete_results[:, 6] = np.log10(ratio)             # obs / detect
        complete_results[:, 7:] = trimmed_results[:, 3:]     # ind obs time

        # Swap the -20 values back to 0
        flip_back_ind = np.where(complete_results < -19.5)
        complete_results[flip_back_ind] = 0

        # Store the entire array
        self.complete_results = complete_results

    def calc_rate_table(self):
        """
        A function used to calculate a production rate range of a progenitor
        We grab a previously generated rate table to perform the calculation.
        The rate is calculated by assuming a single progenitor system
        is responsible for producing the observed LMXB.

        Determine a formation rate using the formula
            N_obs = SUM_ij ( df_ij / dA_ij * Delta A * t_obs,ij )
                Where N_obs         number of observed systems
                      df_ij / dA_ij number of progenitors formed in a
                                    given bin ij
                      Delta A       total size of region that produces thh
                                    observed LMXB
                      t_obs, ij     time spent in the observed bin
            Using N_obs = 1 we can find the maximum rate necessary for a given
            progenitor bin by assuming that seed mass and period combination
            is responsible for the observed binary. Or in other words we assume
            a given ij bin is the actual progenitor combination. We can then
            rearrange the formula to be

            df_ij / dA_ij = N_obs / (Delta A * t_obs,ij)

            The progenitors formed in a given ij bin is given by the above
            equation per year. To find the value per Gyr, we need to multply
            this number by 1e9
        """

        # Try to find the table we will be using to calculate the rates
        try:
            init_table = self.complete_results

        # If the table has not been generated then generate it now
        except AttributeError:
            print("table not generated...")
            self.table_gen()
            init_table = self.complete_results

        # Initialize the column names
        self.set_col_names()

        # Convert the table from a numpy array to a pandas dataframe
        pd_table = pd.DataFrame(init_table, columns=self.col_names)

        # Generate an additional column that represents the first 5 UCXBs
        # Combine together the columns
        # pd_table['Combined_UCXB'] = (10**pd_table.iloc[:, 7:12]).sum(axis=1)

        # Convert the column back to log
        # pd_table['Combined_UCXB'] = np.log10(pd_table['Combined_UCXB'])

        # Due to sum we have some values where it should be 0 that are non-zero
        # pd_table['Combined_UCXB'][]

        # Grab the observed times
        oa_table = pd_table[pd_table.columns[7:]].values

        # Calculate the area of each progenitor binary cell
        dA = pd_table.dm.values * pd_table.dp.values

        # Initialize the length and width of our output array
        table_len = len(dA)
        table_width = len(oa_table[0])

        # Determine where our initial rate table has zeros
        convert_ind = np.where(oa_table == 0)

        # If the observed time is 0 we instead set it to inf as
        # we used observed time as a denominator
        oa_table[convert_ind] = np.inf

        # Generate the table with zeros
        rate_table_range = np.zeros((table_len, table_width))

        # Loop through the columns and calculate the max progenitor number
        for column_index in range(table_width - 1):

            # Determine the total area of the progenitors that produce the
            # observed LMXB
            # detect_ind = np.where(oa_table[:, column_index] != np.inf)
            # detect_area = sum(dA[detect_ind])

            # Calculate the rate per Gyr
            # rate = dA * 1.e9 / (10**oa_table[:, column_index] * detect_area)
            rate = 1.e9 / (10**oa_table[:, column_index])
            oa_table[:, column_index] = np.log10(np.maximum(rate, 1.e-20))

        # combined_detect_ind = np.where(oa_table[:, -1] != np.inf)
        # combined_detect_area = sum(dA[combined_detect_ind])

        # combined_rate = 5 * dA * 1.e9 / (10**oa_table[:, -1] * combined_detect_area)
        combined_rate = 5 * 1.e9 / (10**oa_table[:, -1])
        oa_table[:, -1] = np.log10(np.maximum(combined_rate, 1.e-20))

        # Initialize a new list with zeros to store optimistic values
        optimistic_rate = np.zeros(table_len)
        # Initialize a new list with zeros to store pessimistic values
        pessimistic_rate = np.zeros(table_len)

        # Loop through and grab the largest progenitor number in each row
        for rate_index in range(len(optimistic_rate)):
            oa_row = oa_table[rate_index]
            detect_col = np.where(oa_row > -19.5)
            oa_detect = oa_row[detect_col]

            optimistic_rate[rate_index] = max(oa_detect)
            pessimistic_rate[rate_index] = min(oa_detect)

        # Initialize an array to contain the results
        rate_table_range = np.zeros((table_len, table_width + 6))
        rate_table_range[:, 0] = pd_table.mass.values     # Mass values
        rate_table_range[:, 1] = pd_table.dm.values       # Mass bin
        rate_table_range[:, 2] = pd_table.period.values   # Period values
        rate_table_range[:, 3] = pd_table.dp.values       # Period bin
        rate_table_range[:, 4] = optimistic_rate          # Optimistic rate
        rate_table_range[:, 5] = pessimistic_rate         # Pessimistic rate
        rate_table_range[:, 6:] = oa_table                # Individual values

        # Flip our -20 values back to 0
        flip_ind = np.where(rate_table_range <= -19.5)
        rate_table_range[flip_ind] = 0

        # Store the max progenitor table
        self.rate_table_range = rate_table_range

    def gen_rate_short(self):
        """
        Calculates the progenitor numbers for a given observed LMXB
        """

        # Try to grab the rate_table_range information
        try:
            rate_table_range = self.rate_table_range

        # If the table cannot be found, generate one
        except AttributeError:
            self.calc_rate_table()
            rate_table_range = self.rate_table_range

        # split the table into progenitor properties and rates
        progen_props = rate_table_range[:, 0:4]
        rate_array = rate_table_range[:, 6:]

        # Initialize the system names
        self.set_sys_names()
        sys_names = self.sys_names
        num_sys = len(sys_names)

        # loop through the systems
        for sys_index in range(num_sys):

            # Grab the specific systems name
            sys_name = sys_names[sys_index]

            # Grab the rate column for that system
            rate_column = rate_array[:, sys_index]

            # Determine where the rate is non-zero
            detectable_inds = np.where(rate_column != 0)

            # Find the progenitor properties and the rates
            # corresponding to non-zero rates
            output_props = progen_props[detectable_inds]
            output_rates = rate_column[detectable_inds]

            # Generate the output array
            # The array must be as long as the filtered progenitor properties
            # and the width must be equal to the progen props + 1
            output_len = len(output_props)
            output_width = len(output_props[0]) + 1
            output_array = np.zeros((output_len, output_width))

            # Populate the output array
            output_array[:, :-1] = output_props
            output_array[:, -1] = output_rates

            # Generate a filename for the output
            prefix = str(sys_index).zfill(3)
            sys_str = sys_name.replace(" ", "_")
            output_name = prefix + '_' + sys_str + '_max.dat'

            # Save the array to the given output
            np.savetxt(output_name, output_array,
                       fmt="%1.2f, %1.3f, %1.2f, %1.2f, %1.6f")

    def convert_from_dc(self, max_file_suffix="*max.dat"):
        """
        This function finds all of the observed ratio files and generates a
        rate file name to parse. Using these two files we generate a combined
        file that is in the appropriate format for our plotter.

        Inputs
        ------
        max_file_suffix : string
            We assume that the rate file has a similar naming convention as our
            observed ratio files. We scrape the prefix off the dc file and use
            the suffix given to find the appropriate rate files. The use of a
            wildcard is strongly recommended. Default is "*max.dat".

        """

        # Start looping through all of our observed ratio files
        for dc_file in self.file_list:

            print("Converting " + dc_file)

            # Scrape the prefix from the dc file
            file_prefix = dc_file.split("_")[1]

            # Find the appropriate rate file using the prefix and suffix
            max_file = glob.glob(file_prefix + max_file_suffix)[0]

            print("Using " + max_file)
            # Read in the observed ratio file
            dc_header, dc_data = self._read_dc(dc_file)

            # Read in the rate file
            rate_data = self.load_max_short(max_file)[:, [0, 2, -1]]

            # Split the observed ratio file into mass and period
            dc_mass = dc_data[:, 0]
            dc_period = dc_data[:, 1]

            # Split the rate file into the mass, period and rate values
            rate_mass = rate_data[:, 0]
            rate_period = rate_data[:, 1]
            rate_max = rate_data[:, -1]

            # Generate the output array
            # The output array has length of dc file and the width of
            # the rate file
            output_len = len(dc_data)
            output_width = len(rate_data[0])
            output_array = np.zeros((output_len, output_width))

            # Populate the first two columns of the output with
            # the progenitor properties
            output_array[:, :2] = dc_data[:, :2]

            # Set the rate array to be -20 by default as this is a log value
            output_array[:, -1] = -20

            # Loop through the rows in the rate data
            for row_index in range(len(rate_data)):

                # Grab the appropriate mass, period and rate value from the row
                row_mass = rate_mass[row_index]
                row_period = rate_period[row_index]
                row_rate = rate_max[row_index]

                # Determine where the mass and period overlaps with the dc file
                mass_overlap = np.where(row_mass == dc_mass)
                period_overlap = np.where(row_period == dc_period)

                # Both the mass and period must overlap at the same time
                output_index = np.intersect1d(mass_overlap, period_overlap)

                # Replace the -20 value with the calculated rate
                output_array[output_index, -1] = row_rate

            # Open the file to write to
            output_name = max_file.split(".")[0] + "_to_plot.dat"
            output = open(output_name, 'w')
            print("Writing results to " + output_name + "\n")

            # Write the results to file
            for row in range(len(output_array)):
                mass_value = output_array[row, 0]
                period_value = output_array[row, 1]
                rate_value = output_array[row, 2]
                per_str = str(format(period_value, '.3f')).ljust(10)
                mass_str = str(format(mass_value, '.3f')).ljust(10)
                rate_str = str(format(rate_value, '.3f')).ljust(10)
                output_line = per_str + mass_str + rate_str + '\n'
                output.write(output_line)

            output.close()

    def gen_progen_ranges(self, output_name="rate_ranges.dat"):
        """
        Returns the parameter space ranges for the progenitors

        Inputs
        ------
            output_name : string
                The name of the file to save the ranges to.
                Default is rate_ranges.dat
        """

        # Try to grab the rate_table_range information
        try:
            rate_table_range = self.rate_table_range

        # If the table cannot be found, generate one
        except AttributeError:
            self.calc_rate_table()
            rate_table_range = self.rate_table_range

        # split the table into progenitor properties and rates
        progen_mass = rate_table_range[:, 0]
        progen_period = rate_table_range[:, 2]
        rate_array = rate_table_range[:, 5:]

        # Initialize the system names
        self.set_sys_names()
        sys_names = self.sys_names
        num_sys = len(sys_names)

        # Generate the output array to contain the mass and period ranges
        output_ranges = np.zeros((num_sys, 4))

        # loop through the systems
        for sys_index in range(num_sys):

            # Grab the rate column for that system
            rate_column = rate_array[:, sys_index]

            # Determine where the rate is non-zero
            detectable_inds = np.where(rate_column != 0)

            # Find the progenitor properties and the rates
            # corresponding to non-zero rates
            output_mass = progen_mass[detectable_inds]
            output_period = progen_period[detectable_inds]

            # Find the min/max values of the progenitor properties
            min_mass = min(output_mass)
            max_mass = max(output_mass)

            min_period = min(output_period)
            max_period = max(output_period)

            # Combine the outputs into a list
            range_array = np.array([min_mass, max_mass,
                                    min_period, max_period])

            # Populate the output array
            output_ranges[sys_index] = range_array

        # Save the array to the given output
        np.savetxt(output_name, output_ranges,
                   fmt="%1.2f, %1.2f, %1.2f, %1.2f")

    def gen_bin_prop_table(self, bin_file="", output_name="dc_summary.dat"):
        """
        Generate a table of data from the observed ratio files grabbed during
        initialization. The table will contain the observed binary period
        range, mass ratio range, mass transfer rate range, maximum time
        spent in a given bin and fraction of parameter space the progenitors
        span.

        Inputs
        ------
        bin_file : string
            The path to the file containing the sizes of the bins. If no
            file is provided the code will calculate the bin sizes.
        output_name : string
            The name of the output file, deafult is dc_summary.dat
        """

        # Grab all of the dc files
        dc_files = self.get_file_list()

        # Grab all of the init files
        init_files = glob.glob("*init.dat")
        init_files.sort()

        # Check if the init files are all there, we need one init file per
        # dc file
        print("Checking to see if all init files generated")
        if len(init_files) != len(dc_files):

            print("Missing some init files")
            for dc_file in dc_files:

                # Check the files by grabbing the observed system name
                split_at_combined = dc_file.split("combined_")[-1]
                split_at_dc = split_at_combined.split("_dc.dat")[0]

                print("Checking if init file for " + split_at_dc)
                if not any(split_at_dc in init for init in init_files):

                    # If the init file doesnt exist then generate it
                    print("Generating init file for " + split_at_dc)
                    self._gen_init(dc_file)
                else:
                    print("init file found for " + split_at_dc)

        # Grab a dc file to generate the bin sizes
        dc_file = dc_files[0]
        dc_header, dc_data = self._read_dc(dc_file)
        mass = dc_data.T[0]
        period = dc_data.T[1]

        # Get the bin sizes
        dp, dm, _, _ = self.get_bin_sizes(period, mass, bin_file="")

        # Calculate our total parameter space
        bin_areas = dp * dm
        tot_area = sum(bin_areas)

        # Open the file we're writing our results to
        f2 = open(output_name, 'w')

        # Start looping through the files of interest
        for init_file in init_files:

            # Open the init file
            system_name_no_prefix = init_file.split("combined_")[-1]
            system_name = system_name_no_prefix.split("_init.dat")[0]
            print(system_name)
            f = open(init_file, "r")

            # Initialize the lists we're going to append to
            period = []
            mass = []
            tau = []
            frac = []

            p_bin = []
            m_bin = []

            # Run through all the lines in the init file
            for line in f:
                # Strip the lines of white space and split them at the commas
                linewithoutslashn = line.strip()
                linewithoutcomma = linewithoutslashn.replace(",", " ")
                columns = linewithoutcomma.split()

                # Grab the appropriate values from the file
                calc_p = np.float(columns[0])
                calc_dp = np.float(columns[1])
                calc_m = np.float(columns[2])
                calc_dm = np.float(columns[3])
                calc_den = 10**np.float(columns[4])
                calc_frac = np.float(columns[5])

                # Append the values to the output lists
                period.append(calc_p)
                p_bin.append(calc_dp)
                mass.append(calc_m)
                m_bin.append(calc_dm)
                tau.append(calc_den)
                frac.append(calc_frac)

            # Convert the lists to numpy arrays
            period = np.array(period)
            mass = np.array(mass)
            tau_max = np.array(tau)
            frac_array = np.array(frac)

            dp_array = np.array(p_bin)
            dm_array = np.array(m_bin)
            bin_size_array = dp_array * dm_array
            progen_area = sum(bin_size_array)
            frac_area = progen_area / tot_area

            f.close()

            # If the time spent is non-zero then write the results out
            if len(tau_max) != 0:
                print(max(tau_max))
                print(len(np.where(tau_max)[0]))
                print(sum(bin_size_array))
                print(tot_area)
                print(frac_area)
                max_ind = np.where(max(tau_max) == tau_max)[0]
                max_frac = frac_array[max_ind][0]
                print(max_frac)
                # print(sum(tau_max) / np.where(tau_max)[0])

                f2.write(system_name.ljust(20) +
                         str(format(max(tau_max), "10.2e")).ljust(15) +
                         str(len(np.where(tau_max)[0])).ljust(8) +
                         str(format(frac_area, "10.2e")).ljust(15) +
                         str(format(max_frac, "10.2e")).ljust(10) +
                         '    \n')

            # Otherwise output 0 in the appropriate rows
            else:
                f2.write(system_name.ljust(20) +
                         str(format(0.0, "10.2e")).ljust(15) +
                         str(0.0).ljust(8) +
                         str(format(0.0, "10.2e")).ljust(15) +
                         str(format(0.0, "10.2e")).ljust(10) + '    \n')

        f2.close()

    def gen_rate_summary(self, output_name="rate_summary.dat"):
        """
        Go through all of the max.dat file and summarizes the data
        into a shortened array that contains the name of the LMXB its
        period, mass and mass transfer rate

        Inputs
        ------
        output_name : string
            The name of the file to save the summarized data to. Default
            is rate_summary.dat
        """
        # Open/create the file we're writing to
        output_file = open(output_name, 'w')

        # Grab all of the files we're going to summarize
        max_files = glob.glob("default_max/*_max.dat")

        # Sort the files to ensure they're coming out in the right order
        max_files.sort()

        # Loop through the files
        for system_max in max_files:

            # Strip off the suffixes for the system name
            name_temp = system_max.split("_max.dat")[0]

            # Load in the max data file
            max_array = self.load_max_short(system_max)

            # Grab the mass, period and rates for the file
            mass, dm, period, dp, rate = max_array.T

            # Calculate the area of each bin and total area
            area = dm * dp
            tot_area = sum(area)

            # Find where the rate is max/min in that file
            opt_rate_ind = np.where(min(rate) == rate)[0]

            # Grab the progenitor properties for the optimistic value
            temp_opt_mass = mass[opt_rate_ind][0]
            temp_opt_period = period[opt_rate_ind][0]
            temp_opt_rate = np.around(10**rate[opt_rate_ind][0], 0)

            # Find the average rate
            temp_avg_rate = np.around(sum(10**rate * area) / tot_area, 0)

            # Write the name, the progenitor period, mass and rate
            output_file.write(name_temp.ljust(40) +
                              str(temp_opt_period).ljust(15) +
                              str(temp_opt_mass).ljust(15) +
                              str(temp_opt_rate).ljust(15) +
                              str(temp_avg_rate).ljust(10) +
                              '    \n')

        # Close the file
        output_file.close()

    def convert_rate_to_plot(self, output_name="rate_to_plot"):
        """
        Convert the rate file into a format that our plotting function can
        more easily work with. The difference is that the plotter needs padded
        data with 0s to know what to do there

        Inputs
        ------
        output_name : string
            The name of the output file
        """
        dc_file = self.get_file_list()[0]
        dc_header, dc_data = self._read_dc(dc_file)

        output_array = np.zeros((len(dc_data), 3))

        output_mass = dc_data[:, 0]
        output_period = dc_data[:, 1]

        output_array[:, 0] = output_mass
        output_array[:, 1] = output_period

        try:
            rate_table_range = self.rate_table_range

        # If the table has not been generated then generate it now
        except AttributeError:
            print("table not generated...")
            self.calc_rate_table()
            rate_table_range = self.rate_table_range

        for progen_line in range(len(rate_table_range)):
            progen_mass = rate_table_range[progen_line, 0]
            progen_period = rate_table_range[progen_line, 2]
            progen_rate = rate_table_range[progen_line, 4]

            # Determine where the mass and period overlaps with the dc file
            mass_overlap = np.where(progen_mass == output_mass)
            period_overlap = np.where(progen_period == output_period)

            # Both the mass and period must overlap at the same time
            output_index = np.intersect1d(mass_overlap, period_overlap)

            # Replace the -20 value with the calculated rate
            temp_rate = output_array[output_index, -1]
            if temp_rate == 0:
                output_array[output_index, -1] = progen_rate
            else:
                output_array[output_index, -1] = min(temp_rate, progen_rate)

        non_detects = np.where(output_array[:, -1] == 0)
        output_array[non_detects, -1] = -20

        # Open the file to write to
        output_file = open(output_name, "w")
        print("Writing results to " + output_name + "\n")

        # Write the results to file
        for row in range(len(output_array)):
            mass_value = output_array[row, 0]
            period_value = output_array[row, 1]
            rate_value = output_array[row, 2]
            per_str = str(format(period_value, '.3f')).ljust(10)
            mass_str = str(format(mass_value, '.3f')).ljust(10)
            rate_str = str(format(rate_value, '.3f')).ljust(10)
            output_line = per_str + mass_str + rate_str + '\n'
            output_file.write(output_line)
        output_file.close()

    def gen_random_pmdots(self, N=1000, output_suffix="rand_pmdot.dat"):
        """
        Go through the max.dat files and generates a density file in
        period-mass transfer rate space by grabbing random data points
        from the most optimistic progenitor.

        Inputs
        ------
        N : integer
            Number of random data points to pull from simulation file to
            generate the density file. Default is 1000
        output_suffix : string
            The suffix of the files we are outputting, output string will be
            of the form <numerical index> + <system name> + <output suffix>.
            Default is "rand_pmdot.dat"
        """

        # Grab all of the files we're going to search
        max_files = glob.glob("default_max/*_max.dat")

        # Sort the files to ensure they're coming out in the right order
        max_files.sort()

        print("Generating random pmdot distribution")
        print("N = " + str(N) + " data points to be generated")

        # Loop through the files
        for system_max in max_files:

            print("Finding minimum rate for " + system_max)

            # Strip off the directory and suffixes for the system name
            file_name = system_max.split("/")[1]
            system_name = file_name.split("_max.dat")[0]

            output_name = system_name + "_" + output_suffix

            # Load in the max data file
            max_array = self.load_max_short(system_max)

            # Grab the mass, period and rates for the file
            mass, _, period, _, rate = max_array.T

            # find the index of minimum rate value
            min_rate_index = np.where(rate == min(rate))

            # use the minimum index to grab simulation mass and period
            sim_mass = mass[min_rate_index][0]
            sim_log_period = period[min_rate_index][0]

            # value pulled from file is log(period), convert value
            sim_period = 10**sim_log_period

            # turn the mass and periods into strings to grab the sim
            mass_str = ("%.2f" % sim_mass) + "M"
            period_str = ("%.2f" % sim_period) + "d"
            path_to_sim = mass_str + '/' + period_str

            # use glob to find the complete path the simulation
            full_sim_path = self._set_path_to_sims(path_to_sim)

            print("Looking inside simulation " + path_to_sim)

            # read in the data
            m1 = ms.history_data(full_sim_path)

            # get the age and timestep
            age = m1.get('star_age')
            dt = 10**m1.get('log_dt')

            # get the period
            period_days = m1.get('period_days')
            log_period = np.log10(period_days)

            # calculate the mass ratio
            mass_1 = m1.get('star_1_mass')
            mass_2 = m1.get('star_2_mass')
            mass_ratio = mass_1 / mass_2

            # get the mass transfer rate
            log_mt = m1.get('lg_mtransfer_rate')

            # combine into one array
            sim_array = np.array([age, dt, mass_ratio, log_period, log_mt]).T

            # generate a list of random indices, the possible number of
            # indices must equal to the length of our data or the result
            # will be skewed. Our list of random indices will have N
            # entries corresponding to the number given by user
            rand_inds = np.random.choice(len(sim_array), N)

            # use the random indices to generate random data
            rand_data = sim_array[rand_inds]

            # save the data
            np.savetxt(output_name, rand_data)

    def gen_random_average(self, N=1000, output_suffix="rand_dist.dat",
                           q_bin=[0.01, 0.06], P_bin=[-1.95, -1.90],
                           MT_bin=[-9.0, -8.4], T_bin=[0, 1e9],
                           system_index=0, time_threshold=1e3):
        """
        Go through the max.dat files and generates a density file in
        period-mass transfer rate space by grabbing random data points
        from the most optimistic progenitor.

        Inputs
        ------
        N : integer
            Number of iterations generate the density file. Default is 1000
        output_suffix : string
            The suffix of the files we are outputting, output string will be
            of the form <numerical index> + <system name> + <output suffix>.
            Default is "rand_dist.dat"
        q_bin : list
            The edges of the observed mass ratio bin, default is [0.01, 0.06]
        P_bin : list
            The edges of the observed period bin bin in log days, default is
            [-1.95, -1.90]
        MT_bin : list
            The edges of the observed mass transfer rate bin in Msun/yr,
            default is [-9.0, -8.4]
        T_bin : list
            The edges of the observed effective temperature bin in K, default
            is [0, 1e9]
        system_index : int
            The index of the observed LMXB system we're interested in, default
            is 0
        time_threshold : float
            The minimum amount of time we want to exceed with random points,
            default is 1e3
        """

        # Grab all of the files we're going to search
        max_files = glob.glob("default_max/*_max.dat")

        # Sort the files to ensure they're coming out in the right order
        max_files.sort()
        system_max = max_files[system_index]

        print("Generating physical property array")
        print("Finding random distribution for " + system_max)

        # Strip off the directory and suffixes for the system name
        file_name = system_max.split("/")[1]
        system_name = file_name.split("_max.dat")[0]

        # Load in the max data file
        max_array = self.load_max_short(system_max)

        # Grab the mass, period and rates for the file
        mass, _, period, _, rate = max_array.T

        for sim_index in range(len(mass)):
            sim_mass = mass[sim_index]
            sim_log_period = period[sim_index]

            # value pulled from file is log(period), convert value
            sim_period = 10**sim_log_period

            # turn the mass and periods into strings to grab the sim
            mass_str = ("%.2f" % sim_mass) + "M"
            period_str = ("%.2f" % sim_period) + "d"
            path_to_sim = mass_str + '/' + period_str

            # use glob to find the complete path the simulation
            full_sim_path = self._set_path_to_sims(path_to_sim)

            print("Looking inside simulation " + path_to_sim)

            # read in the data
            m1 = ms.history_data(full_sim_path)

            # get the age and timestep
            # age = m1.get('star_age')
            dt = 10**m1.get('log_dt')

            # get the period
            period_days = m1.get('period_days')
            log_period = np.log10(period_days)

            # calculate the mass ratio
            mass_1 = m1.get('star_1_mass')
            mass_2 = m1.get('star_2_mass')
            mass_ratio = mass_1 / mass_2

            # get the mass transfer rate
            log_mt = m1.get('lg_mtransfer_rate')

            # get the effective temperature
            Teff = 10**m1.get('log_Teff')

            # If this is the first time we're looping through the systems
            # define the temporary arrays
            if sim_index == 0:
                temp_q = mass_ratio
                temp_mt = log_mt
                temp_period = log_period
                temp_Teff = Teff
                temp_dt = dt

            # append the new data to the old temp array
            else:
                temp_q = np.append(temp_q, mass_ratio)
                temp_mt = np.append(temp_mt, log_mt)
                temp_period = np.append(temp_period, log_period)
                temp_Teff = np.append(temp_Teff, Teff)
                temp_dt = np.append(temp_dt, dt)

        # Start the sequence to generate the averages
        print("Generating a random distribution")
        print("N = " + str(N) + " iterations")
        output_name = system_name + "_" + output_suffix

        # Get the min/max edges of our bins
        min_P = min(P_bin)
        max_P = max(P_bin)

        min_q = min(q_bin)
        max_q = max(q_bin)

        min_MT = min(MT_bin)
        max_MT = max(MT_bin)

        min_T = min(T_bin)
        max_T = max(T_bin)

        # generate our output array
        iteration_array = np.zeros(N)

        # Iterate as many times as we have user defined iterations
        for iteration in range(N):

            # track the number of loops
            num_sims = 1
            total_obs_time = 0

            # If we still havent reached a sufficiently long total
            # observed time
            while total_obs_time < 1e3:

                # randomly select a point
                rand_point = np.random.choice(len(temp_q))
                rand_q = temp_q[rand_point]
                rand_mt = temp_mt[rand_point]
                rand_p = temp_period[rand_point]
                rand_T = temp_Teff[rand_point]

                # Check if the simulated model satisfies any of the
                # period (P), mass ratio (q), mass transfer (MT) and
                # temperature (T) condtions
                P_check = np.where(np.logical_and(rand_p >= min_P,
                                                  rand_p <= max_P))[0]

                q_check = np.where(np.logical_and(rand_q >= min_q,
                                                  rand_q <= max_q))[0]

                MT_check = np.where(np.logical_and(rand_mt >= min_MT,
                                                   rand_mt <= max_MT))[0]

                T_check = np.where(np.logical_and(rand_T >= min_T,
                                                  rand_T <= max_T))[0]

                # Find the indices where all four conditions are satisfied
                common_inds = set(P_check) & set(q_check) &\
                    set(MT_check) & set(T_check)

                # Iterate the number of simulations
                num_sims += 1

                # If our random point exists in the observed cuboid then add
                if common_inds:
                    total_obs_time += temp_dt[rand_point]

            # append the number of sims to array
            iteration_array[iteration] = num_sims

        # save the data
        np.savetxt(output_name, iteration_array)

    def gen_random_flat(self, init_mass=1.3, init_per=-0.08,
                        N=1000, output_name="flat_rand_dist.dat",
                        q_bin=[0.01, 0.06], P_bin=[-1.95, -1.90],
                        MT_bin=[-9.0, -8.4], T_bin=[0, 1e9],
                        age_threshold=7e9, max_age=1e10):
        """
        2. randomly select some starting birth point for the star
        between 0-10 Gyr
        3. check if that randomly selected starting birth point results
        in the system reproducing the observed LMXB within 7Gyrs
        4. randomly select a data point from our simulated data with some
        randomly selected starting point between 0-10Gyrs
        5. Check if this random point from step 4 matches with our observed
        LMXB
        6. repeat step 6 until we have a match, record number
        7. repeat until we have two matches, record number


        Inputs
        ------
        N : integer
            Number of iterations generate the density file. Default is 1000
        output_suffix : string
            The suffix of the files we are outputting, output string will be
            of the form <numerical index> + <system name> + <output suffix>.
            Default is "rand_dist.dat"
        q_bin : list
            The edges of the observed mass ratio bin, default is [0.01, 0.06]
        P_bin : list
            The edges of the observed period bin bin in log days, default is
            [-1.95, -1.90]
        MT_bin : list
            The edges of the observed mass transfer rate bin in Msun/yr,
            default is [-9.0, -8.4]
        T_bin : list
            The edges of the observed effective temperature bin in K, default
            is [0, 1e9]
        time_threshold : float
            The minimum amount of time we want to exceed with random points,
            default is 7e9
        """

        # turn the mass and periods into strings to grab the sim
        mass_str = ("%.2f" % init_mass) + "M"
        period_str = ("%.2f" % 10**init_per) + "d"
        path_to_sim = mass_str + '/' + period_str

        # use glob to find the complete path the simulation
        full_sim_path = self._set_path_to_sims(path_to_sim)

        print("Looking inside simulation " + path_to_sim)

        # read in the data
        m1 = ms.history_data(full_sim_path)

        # get the age and timestep
        age = m1.get('star_age')

        # get the period
        period_days = m1.get('period_days')
        log_period = np.log10(period_days)

        # calculate the mass ratio
        mass_1 = m1.get('star_1_mass')
        mass_2 = m1.get('star_2_mass')
        mass_ratio = mass_1 / mass_2

        # get the mass transfer rate
        log_mt = m1.get('lg_mtransfer_rate')

        # get the effective temperature
        Teff = 10**m1.get('log_Teff')

        # Get the min/max edges of our bins
        min_P = min(P_bin)
        max_P = max(P_bin)

        min_q = min(q_bin)
        max_q = max(q_bin)

        min_MT = min(MT_bin)
        max_MT = max(MT_bin)

        min_T = min(T_bin)
        max_T = max(T_bin)

        print("Finding the age for simulation")
        # Find where our simulated system matches with observed LMXB
        P_check = np.where(np.logical_and(log_period >= min_P,
                                          log_period <= max_P))[0]

        q_check = np.where(np.logical_and(mass_ratio >= min_q,
                                          mass_ratio <= max_q))[0]

        MT_check = np.where(np.logical_and(log_mt >= min_MT,
                                           log_mt <= max_MT))[0]

        T_check = np.where(np.logical_and(Teff >= min_T,
                                          Teff <= max_T))[0]

        # Find the indices where all four conditions are satisfied
        common_inds = set(P_check) & set(q_check) &\
            set(MT_check) & set(T_check)

        # Convert the indices back to a list and sort
        common_inds_list = list(common_inds)
        common_inds_list.sort()

        # Calculate the first age point where we are in observed
        # cuboid
        age_lim = age[common_inds_list[0]]

        # generate our output array
        iteration_array = np.zeros(N)
        checkpoints = np.linspace(0, N, 5)

        print("Start iteration")
        # Iterate as many times as we have user defined iterations
        for iteration in range(N):
            if iteration in checkpoints.astype(int):
                print("Iteration Number: " + str(iteration))
            # track the number of loops
            num_sims = 1

            # Loop until we satisfy two specific conditions
            while True:

                # randomly select a point
                rand_point = np.random.choice(len(age))
                rand_q = mass_ratio[rand_point]
                rand_mt = log_mt[rand_point]
                rand_p = log_period[rand_point]
                rand_T = Teff[rand_point]
                # rand_age = age[rand_point]

                # offset our age so that it was born at some random time
                # between 0 and our threshold
                age_offset = np.random.uniform(high=max_age)
                # print(age_offset)

                # shifted_age = rand_age + age_offset + age_lim - age_threshold
                shifted_age = age_threshold - age_offset - age_lim
                # print(shifted_age)

                # Check if the simulated model satisfies any of the
                # period (P), mass ratio (q), mass transfer (MT) and
                # temperature (T) condtions
                P_check = np.where(np.logical_and(rand_p >= min_P,
                                                  rand_p <= max_P))[0]

                q_check = np.where(np.logical_and(rand_q >= min_q,
                                                  rand_q <= max_q))[0]

                MT_check = np.where(np.logical_and(rand_mt >= min_MT,
                                                   rand_mt <= max_MT))[0]

                T_check = np.where(np.logical_and(rand_T >= min_T,
                                                  rand_T <= max_T))[0]

                age_check = np.where(shifted_age >= 0)[0]

                # Find the indices where all four conditions are satisfied
                common_inds = set(P_check) & set(q_check) &\
                    set(MT_check) & set(T_check) & set(age_check)

                # Iterate the number of simulations
                num_sims += 1

                # If our random point exists in the observed cuboid then add
                if common_inds:
                    break

            # append the number of sims to array
            iteration_array[iteration] = num_sims

        print("Average formation rate for " + path_to_sim + ":")
        avg_flat_rate = np.average(iteration_array)
        print(avg_flat_rate)

        # save the data
        print("Finished Iterations, Saving")
        np.savetxt(output_name, iteration_array, fmt='%4.2f')

    def gen_density_from_random(self, input_suffix="rand_pmdot.dat",
                                output_suffix="rand_density.dat",
                                normalize=False,
                                period_range=(-0.5, 3.5), n_period=50,
                                mt_range=(-12, -4), n_mt=50):
        """
        This functionis designed to convert the output of gen_random_pmdots
        to a density file using the given ranges and number of expected
        bins in period and mass transfer rate.

        Inputs
        ------
        input_suffix : string
            The shared suffix of the files we want to generate density
            files from. Default is "rand_pmdot.dat"
        output_suffix : string
            The shared suffix of the files we want to save our generated
            density data to. Default is "rand_density.dat"
        normalize : boolean
            A flag used if the user wishes to normalize the density data.
            Default is false which results in raw time data as density.
        period_range : tuple
            The range of our period axis in log(hours), default is (-0.5, 3.5)
        n_period : integer
            The number of period bins our range spans, default is 50
        mt_range : tuple
            The range of our mass transfer rate axis in log(Msun/yr).
            Default is (-12, -4)
        n_mt : integer
            The number of mass transfer rate bins our range spans.
            Default is 50
        """

        # Define what the min and maxes are for our period and mass transfer
        period_min = float(min(period_range))
        period_max = float(max(period_range))

        mt_min = float(min(mt_range))
        mt_max = float(max(mt_range))

        # Calculate the change in period and mass transfer rate
        dperiod = (period_max - period_min) / n_period
        dmt = (mt_max - mt_min) / n_mt

        # Generate the period and mass transfer bins
        period_axis = np.arange(period_min, period_max, dperiod)
        mt_axis = np.arange(mt_min, mt_max, dmt)

        # Check to make sure our upper edge of our bins is included
        if period_max not in period_axis:
            period_axis = np.append(period_axis, period_max)
        if mt_max not in mt_axis:
            mt_axis = np.append(mt_axis, mt_max)

        # Create the array to hold our combined time density
        pmdot_tot = np.zeros((n_mt, n_period))

        # Grab all of the density files that match with our suffix
        pmdot_files = glob.glob("*" + input_suffix)
        pmdot_files.sort()

        # Loop through the density files
        for random_data in pmdot_files:

            print("Generating density from " + random_data)

            # Read in a given file
            temp_pmdot = np.loadtxt(random_data)

            # Split the data into the appropriate columns
            age, dt, mass_ratio, log_period, log_mt = temp_pmdot.T

            # Convert period to log hours
            period_hours = (10**log_period) * 24
            log_p_h = np.log10(period_hours)

            # Loop through the period bins
            for p_ind in range(len(period_axis) - 1):
                sub_period = (period_axis[p_ind],
                              period_axis[p_ind + 1])
                # print(sub_y)
                sub_p_min = float(min(sub_period))
                sub_p_max = float(max(sub_period))

                for mt_ind in range(len(mt_axis) - 1):
                    sub_mt = (mt_axis[mt_ind],
                              mt_axis[mt_ind + 1])
                    # print(sub_x)
                    sub_mt_min = float(min(sub_mt))
                    sub_mt_max = float(max(sub_mt))

                    p_chk = np.where(np.logical_and(log_p_h >= sub_p_min,
                                                    log_p_h <= sub_p_max))[0]
                    mt_chk = np.where(np.logical_and(log_mt >= sub_mt_min,
                                                     log_mt <= sub_mt_max))[0]

                    p_mt_chk = list(set(p_chk) & set(mt_chk))
                    if len(p_mt_chk) > 0:
                        for p_mt_ind in p_mt_chk:
                            pmdot_tot[mt_ind, p_ind] += dt[p_mt_ind]

            pmdot_max = np.amax(pmdot_tot)
            if normalize:
                normed_pmdot = pmdot_tot / pmdot_max
            else:
                normed_pmdot = pmdot_tot

            # Grab the system name by stripping the suffix off of the
            # randomly generated data file name
            system_name = random_data.split("rand")[0]

            # Generate the output file name
            output_name = system_name + output_suffix

            # Open the file
            f = open(output_name, 'w')

            # Ensure we are at the top of the file
            f.seek(0)

            # Write the results to the text file
            f.write(str(n_period) + '\n')
            f.write(str(n_mt) + '\n')
            f.write(str(period_max) + '\n')
            f.write(str(period_min) + '\n')
            f.write(str(mt_max) + '\n')
            f.write(str(mt_min) + '\n')
            f.write(str(dperiod) + '\n')
            f.write(str(dmt) + '\n')

            # Loop through all combinations of period and mass transfer rate
            for i in range(int(n_period)):
                for j in range(int(n_mt)):
                    per_string = str(format(period_axis[i], '.3f')).ljust(10)
                    mt_string = str(format(mt_axis[j], '.3f')).ljust(10)
                    ptot_string = str(format(normed_pmdot[j, i], '.8f')) + '\n'
                    f.write(per_string + mt_string + ptot_string)

            # Close the file
            f.close()

    def gen_random_rate_range(self, N=1000, output_name="random_range.dat",
                              exclude_cyg=False):
        """
        Go through the max.dat files to sum together one calculated rate per
        observed LMXB to generate a random formation rate.

        Inputs
        ------
        N : integer
            Number of iterations we want to do the calculation
        output_name : string
            The name of the files we are outputting our results to
        exclude_cyg : boolean
            A flag to allow the user to exclude the rates of the system Cyg X-2
            The formation rate of this LMXB dominates the calculation so for
            more insights the user can exclude this system.
        """

        # Grab all of the files we're going to search
        max_files = glob.glob("default_max/*_max.dat")

        # Sort the files to ensure they're coming out in the right order
        max_files.sort()

        # Exclude Cyg X-2 if flagged
        if exclude_cyg:
            max_files = max_files[:-2]
        else:
            max_files = max_files[:-1]

        # create the array to hold our results

        output_rates = np.zeros(N)
        temp_array = np.zeros((N, len(max_files), 5))

        print("Generating random formation total")
        print("N = " + str(N) + " iterations")

        # loop through the systems
        for system_ind in range(len(max_files)):
            system_max = max_files[system_ind]
            print("Pulling a random formation rate from " + system_max)

            # Load in the max data file
            max_array = rt.load_max_short(system_max)

            # loop through the iterations
            for iteration in range(N):

                # randomly select a progenitor
                random_ind = np.random.choice(len(max_array))
                random_rate_data = max_array[random_ind]

                # check if our randomly selected progenitor is already in array
                matching_inds = np.where((temp_array[iteration, :, :-1]
                                         == random_rate_data[:-1]).all(axis=1))
                if len(matching_inds[0]):
                    # print("found match")
                    curr_rate = temp_array[iteration, matching_inds, -1]
                    temp_rate = random_rate_data[-1]
                    temp_array[iteration, matching_inds, -1] = max(curr_rate,
                                                                   temp_rate)
                    random_rate_data = random_rate_data * -np.inf

                temp_array[iteration][system_ind] = random_rate_data



            # Grab the mass, period and rates for the file
            mass, _, period, _, rate = max_array.T

            # generate a list of random indices, the possible number of
            # indices must equal to the length of our data or the result
            # will be skewed. Our list of random indices will have N
            # entries corresponding to the number given by user
            rand_rate = np.random.choice(rate, N)

            # use the random indices to generate random data
            output_rates += 10**rand_rate

        # save the data
        np.savetxt(output_name, output_rates, fmt="%1.5f")

    """
    ################################
    Save Functions
    ################################
    """

    def save_init_table(self, output_name):
        """
        Output the results generated from table_gen
        If table_gen has not been run then we have nothing to save so
        raise an error then run the table_gen code

        Inputs
        ------
            output_name : string
                The name of the save file
        """

        # Try to save the file, if successful then print to terminal
        # that the file successfully saved
        try:
            np.savetxt(output_name, self.complete_results,
                       fmt="%1.2f, %1.3f, %1.2f, %1.3f, %1.4f, %1.4f, %1.4f,\
                            %1.4f, %1.4f, %1.4f, %1.4f, %1.4f, %1.4f, %1.4f,\
                            %1.4f, %1.4f, %1.4f, %1.4f,\
                            %1.4f, %1.4f,\
                            %1.4f,\
                            %1.4f")
            print("saving table as " + output_name)

        # If the table could not be saved then regenerate it and try again
        except AttributeError:
            print("table not generated...")
            self.table_gen()
            self.save_init_table(output_name)

    def save_rate_table_range(self, output_name):
        """
        Output the results generated from calc_rate_table
        If table_gen has not been run then we have nothing to save so
        raise an error then run the table_gen code

        Inputs
        ------
            output_name : string
                The name of the save file
        """

        # Try to save the file, if successful then print to terminal
        # that the file successfully saved
        try:
            np.savetxt(output_name, self.rate_table_range,
                       fmt="%1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f,\
                            %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f,\
                            %1.2f, %1.2f, %1.2f, %1.2f,\
                            %1.2f, %1.2f,\
                            %1.2f,\
                            %1.2f")
            print("saving table as " + output_name)

        # If the table could not be saved then regenerate it and try again
        except AttributeError:
            print("max rate table not generated...")
            self.calc_rate_table()
            self.save_rate_table_range(output_name)

    """
    ################################
    Load Functions
    ################################
    """
    """
    These functions all load in an already generated table.
    Assumed to be of the same format as the tables generated as this code,
    we dont have any checks for this. If we cannot load in a file then we will
    automatically run the code that produces and saves a table.

    Inputs
    ------
        input_name : string
            The name of the file we wish to load, if np.loadtxt fails
            then this becomes the name of the save file
    """

    def load_init_table(self, input_name):
        # Try to load in a table that is comma delimited
        try:
            self.complete_results = np.loadtxt(input_name, delimiter=',')
            print("loading table " + input_name)

        # If a table cannot be found then we move into the save_init_table
        # function which will save any currently generated table or generate
        # a new table
        except IOError:
            print("table not found...")
            self.save_init_table(input_name)

    def load_max_table(self, input_name):
        # Try to load in a table that is comma delimited
        try:
            self.rate_table_range = np.loadtxt(input_name, delimiter=',')
            print("loading table " + input_name)

        # If a table cannot be found then we move into the save_init_table
        # function which will save any currently generated table or generate
        # a new table
        except IOError:
            print("table not found...")
            self.save_rate_table_range(input_name)

    def load_max_short(self, input_name):
        # Try to load in a table that is comma delimited
        try:
            max_short = np.loadtxt(input_name, delimiter=',')
            print("loading table " + input_name)
            return max_short
        except IOError:
            print("file does not exist")

    """
    ################################
    Private Functions
    ################################
    """

    def _read_dc(self, file_name, inc_total_age=False):
        """
        Private routine that reads in the observed ratio data
        Inputs
        ------
        file_name : string
            The name of the file to be read

        """
        # open the file using the name provided
        dc_file = open(file_name, 'r')
        dc_file.seek(0)

        # initialize the header list and the data
        header_list = []
        mass = []
        period = []
        detect_age = []
        observ_age = []
        total_age = []

        # Loop through the entire file
        for line in dc_file:

            # take the line of data and split it into a list of floats
            temp_list = list(map(float, line.split()))

            # If the line contains only a signle number it is in the
            # header
            if len(temp_list) == 1:
                header_list.append(temp_list[0])

            # otherwise it is the main body of the data
            else:
                # strip and split the data into columns
                linewithoutslashn = line.strip()
                columns = linewithoutslashn.split()

                # convert the values into floats
                period_val = np.float(columns[0])
                mass_val = np.float(columns[1])

                # observable means that the simulation is in an observable bin
                # detectable means that the simulation is deemed persistent and
                # thus detectable
                observable_val = np.float(columns[3])
                detectable_val = np.float(columns[4])
                total_age_val = np.float(columns[5])

                # convert values into log, we also set a lower limit
                # to this value to avoid issues for log(0)
                log_observ = np.log10(np.maximum(np.array(observable_val),
                                                 1.e-20))
                log_detect = np.log10(np.maximum(np.array(detectable_val),
                                                 1.e-20))
                log_total_age = np.log10(np.maximum(np.array(total_age_val),
                                                    1.e-20))

                # append all of the values to the appropriate column
                # We ensure that the mass is appropriate by rounding it
                mass.append(round(mass_val, 2))
                period.append(period_val)
                detect_age.append(log_detect)
                observ_age.append(log_observ)
                total_age.append(log_total_age)

                # Transpose the array
                if inc_total_age:
                    results = np.array([mass, period,
                                        detect_age, observ_age, total_age]).T
                else:
                    results = np.array([mass, period,
                                        detect_age, observ_age]).T
        return header_list, results

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

    def _find_init_files(self):
        """
        A function that checks if the init files are all generated. If the
        files aren't generated then generate them here
        """

        # Grab all of the dc files
        dc_files = self.get_file_list()

        # Grab all of the init files
        init_files = glob.glob("*init.dat")
        init_files.sort()

        # If we don't have any init files then just generate all of them
        if not init_files:
            self._gen_init_files()

        else:
            # Check if the init files are all there, we need one init file
            # per dc file
            print("Checking to see if all init files generated")
            if len(init_files) != len(dc_files):

                print("Missing some init files")
                for dc_file in dc_files:

                    # Check the files by grabbing the observed system name
                    split_at_combined = dc_file.split("combined_")[-1]
                    split_at_dc = split_at_combined.split("_dc.dat")[0]

                    print("Checking if init file for " + split_at_dc)
                    if not any(split_at_dc in init for init in init_files):

                        # If the init file doesnt exist then generate it
                        print("Generating init file for " + split_at_dc)
                        self._gen_init(dc_file)
                    else:
                        print("init file found for " + split_at_dc)

    def _gen_init_files(self):
        """
        A short function that generates an init file for each dc file
        """
        dc_files = self.get_file_list()
        for dc_file in dc_files:
            self._gen_init(dc_file)

    def _gen_init(self, file_name):
        """
        A function that generates an init file that contains the following
            initial period
            period bin size
            initial mass
            mass bin size
            amount of time simulated system in observed bin
            fraction of lifetime simulated system in observed bin
        """

        # Read in the dc file and split the data into the relevant columns
        dc_header, dc_data = self._read_dc(file_name, inc_total_age=True)
        mass, period, detectable, observable, total_age = dc_data.T

        # Calculate the bin sizes
        dp, dm, _, _ = self.get_bin_sizes(period, mass, bin_file="")

        # Calculate the fractional amount of time the simulation spends
        # persistent
        fractional_age = 10**detectable / 10**total_age
        # log_frac_age = np.log10(fractional_age)

        # Determine where the simulation enters our observable bins
        observable_inds = np.where(observable > -20)

        # Grab the data where the simulation enters an observedbin
        output_mass = mass[observable_inds]
        output_period = period[observable_inds]
        output_dp = dp[observable_inds]
        output_dm = dm[observable_inds]
        output_observable = observable[observable_inds]
        output_frac = fractional_age[observable_inds]

        # put it all together into a numpy array
        output_array = np.array([output_period, output_dp,
                                 output_mass, output_dm,
                                 output_observable, output_frac]).T

        # Set the output file name as the system name + init.dat
        output_file = file_name.split("dc")[0] + "init.dat"

        # Save the file
        np.savetxt(output_file, output_array,
                   fmt="%1.2f, %1.2f, %1.2f, %1.2f, %1.7f, %1.7f")

    def _set_path_to_sims(self, path):
        """
        Defines the path to our simulated files

        Inputs
        ------
        path : string
            The name of the file we're going to determine the path to

        """
        full_path = glob.glob("period*/group*/" + path, recursive=True)[0]
        return full_path + '/LOGS'

    """
    ################################
    Plotting Functions
    ################################
    """

    def plot_density(self, rate_file, bin_file="",
                     max_color=0.0, min_color=-6, num_color=13,
                     color_map='viridis', colorbar=True,
                     xvar=r'$\log_{10}$ (Period/days)',
                     yvar=r'Mass ($M_\odot$)',
                     title="",
                     cb_label=r"$\log_{10}(f)$", solo_fig=True):
        """
        A routine that produces a density plot using the density file given
        with the color showing different properties based on user choice.

        inputs
        ------
        rate_file : string
            The file we are plotting the density of
        bin_file : string
            The name of the file containing the bin size data and the
            bin edges.
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
        density_data = np.loadtxt(rate_file)
        xvalues = density_data[:, 0]
        xmin = min(xvalues)
        xmax = max(xvalues)

        yvalues = density_data[:, 1]
        ymin = min(yvalues)
        ymax = max(yvalues)

        xnum = len(np.unique(xvalues))
        ynum = len(np.unique(yvalues))

        rect_color = density_data[:, 2]

        # Define the colormap
        cmap = plt.get_cmap(color_map)
        cNorm = matplotlib.colors.Normalize(vmin=min_color,
                                            vmax=max_color)
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        # Grab the bin properties
        x_bins, y_bins, x_corner, y_corner = self.get_bin_sizes(xvalues,
                                                                yvalues)

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
                                     angle=0, edgecolor='None',
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

        # If it is a solo figure then draw the rest of the relevant properties
        if solo_fig:
            plt.xlabel(xvar, fontsize=18)
            plt.ylabel(yvar, fontsize=18)
            plt.title(title, fontsize=22)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)

    def plot_many(self, file_suffix="*max_to_plot.dat"):
        """
        A function that finds all of the max rate data generated
        and plots each of them individually

        Inputs
        ------
            file_suffix : string
                The trailing set of characters that all of the rate
                data has in common. Wildcards are necessary for this
                naming to work. Default is *max_to_plot.dat
        """

        # Initialize the system names and find number of systems to be plotted
        self.set_sys_names()
        sys_names = self.sys_names
        num_sys = len(sys_names)

        # Find all of the files with the inputted suffix
        files_to_plot = glob.glob(file_suffix)

        # Sort the files to ensure system names match with files
        # no rigorous checking that the two match is done
        files_to_plot.sort()

        # Loop through all of the files we're plotting
        for file_index in range(num_sys):

            # replace any spaces in the string with underscores
            sys_no_space = sys_names[file_index].replace(" ", "_")

            # Generate the plot
            self.plot_density(files_to_plot[file_index])

            # Depending on the observed source we have two different
            # sets of x and y limits.
            if "Cyg_X-2" not in sys_no_space:
                plt.xlim(-0.5, 1)
                plt.ylim(0.9, 4)
            else:
                plt.xlim(0, 1.5)
                plt.ylim(1.9, 5)

            # Title and save figure
            plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)
            plt.title(sys_names[file_index] + "Rate")
            plt.savefig(sys_no_space + "_rate.pdf")

            # Clear the figure so we don't overplot
            plt.clf()
