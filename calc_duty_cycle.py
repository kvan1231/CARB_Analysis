import mesa_read as ms
import numpy as np
import val_find
import os


"""
This code determine which models in our parameter space can reproduce
observed LMXBs and calculates the relevant properties, duty cycle
observable time and detectable time.

duty cycle = observed time / detectable time

    observed time :
        The amount of time a model appears similar to an observed LMXB
    detectable time :
        The amount of time the model is deemed to be detectable. A system
        is considered detectable if it is classified as persistent using
        a disk instability model or if it is within the persistent
        mass transfer rate.
"""

# The names of all the observed systems we will be comparing our models to
title_list = ["4U 0513-40",
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
              "Cyg X-2"]

# These values are the default values we used to find the progenitors

# The period ranges of the observed systems, the values are in log10(P/hr)
unique_P = [(-0.57, -0.52),   # "4U 0513-40"
            (-0.56, -0.51),   # "2S 0918-549"
            (-0.54, -0.49),   # "4U 1543-624"
            (-0.48, -0.43),   # "4U 1850-087"
            (-0.44, -0.39),   # "M15 X-2"
            (-0.17, -0.12),   # "4U 1626-67"
            (-0.10, -0.05),   # "4U 1916-053"
            (0.56, 0.61),     # "4U 1636-536"
            (0.60, 0.65),     # "GX 9+9"
            (0.65, 0.70),     # "4U 1735-444"
            (0.73, 0.78),     # "2A 1822-371"
            (1.26, 1.31),     # "Sco X-1"
            (1.33, 1.38),     # "GX 349+2"
            (2.35, 2.40)]     # "Cyg X-2"

# The mass ratio range of the observed systems, q = mass donor / mass accretor
unique_q = [(0.01, 0.06),     # "4U 0513-40"
            (0.01, 0.06),     # "2S 0918-549"
            (0.01, 0.06),     # "4U 1543-624"
            (0.01, 0.06),     # "4U 1850-087"
            (0.01, 0.06),     # "M15 X-2"
            (0.01, 0.06),     # "4U 1626-67"
            (0.03, 0.08),     # "4U 1916-053"
            (0.15, 0.40),     # "4U 1636-536"
            (0.20, 0.33),     # "GX 9+9"
            (0.29, 0.48),     # "4U 1735-444"
            (0.26, 0.36),     # "2A 1822-371"
            (0.15, 0.58),     # "Sco X-1"
            (0.39, 0.65),     # "GX 349+2"
            (0.25, 0.53)]     # "Cyg X-2"

# The mass transfer rate range of the observed systems in log10(Msun/yr)
unique_MT = [(-8.98, -8.38),  # "4U 0513-40"
             (-9.58, -8.38),  # "2S 0918-549"
             (-8.88, -8.38),  # "4U 1543-624"
             (-9.78, -8.18),  # "4U 1850-087"
             (-9.48, -8.88),  # "M15 X-2"
             (-9.48, -8.38),  # "4U 1626-67"
             (-9.38, -8.68),  # "4U 1916-053"
             (-8.88, -8.38),  # "4U 1636-536"
             (-8.48, -7.98),  # "GX 9+9"
             (-8.18, -7.68),  # "4U 1735-444"
             (-7.80, -7.08),  # "2A 1822-371"
             (-7.80, -7.08),  # "Sco X-1"
             (-7.80, -7.08),  # "GX 349+2"
             (-7.80, -6.98)]  # "Cyg X-2"

# These values are the wider values used to compare our results

# The period ranges of the observed systems, the values are in log10(P/hr)
# unique_P = [(-0.62, -0.02),   # "4U 0513-40"
#             (-0.62, -0.02),   # "2S 0918-549"
#             (-0.62, -0.02),   # "4U 1543-624"
#             (-0.62, -0.02),   # "4U 1850-087"
#             (-0.62, -0.02),   # "M15 X-2"
#             (-0.62, -0.02),   # "4U 1626-67"
#             (-0.62, -0.02),   # "4U 1916-053"
#             (0.56, 0.61),     # "4U 1636-536"
#             (0.60, 0.65),     # "GX 9+9"
#             (0.65, 0.70),     # "4U 1735-444"
#             (0.73, 0.78),     # "2A 1822-371"
#             (1.26, 1.31),     # "Sco X-1"
#             (1.33, 1.38),     # "GX 349+2"
#             (2.35, 2.40)]     # "Cyg X-2"

# The mass ratio range of the observed systems, q = mass donor / mass accretor
# unique_q = [(0.01, 0.08),     # "4U 0513-40"
#             (0.01, 0.08),     # "2S 0918-549"
#             (0.01, 0.08),     # "4U 1543-624"
#             (0.01, 0.08),     # "4U 1850-087"
#             (0.01, 0.08),     # "M15 X-2"
#             (0.01, 0.08),     # "4U 1626-67"
#             (0.01, 0.08),     # "4U 1916-053"
#             (0.15, 0.40),     # "4U 1636-536"
#             (0.20, 0.33),     # "GX 9+9"
#             (0.29, 0.48),     # "4U 1735-444"
#             (0.26, 0.36),     # "2A 1822-371"
#             (0.15, 0.58),     # "Sco X-1"
#             (0.39, 0.65),     # "GX 349+2"
#             (0.25, 0.53)]     # "Cyg X-2"

# # The mass transfer rate range of the observed systems in log10(Msun/yr)
# unique_MT = [(-12.0, -7.5),  # "4U 0513-40"
#              (-12.0, -7.5),  # "2S 0918-549"
#              (-12.0, -7.5),  # "4U 1543-624"
#              (-12.0, -7.5),  # "4U 1850-087"
#              (-12.0, -7.5),  # "M15 X-2"
#              (-12.0, -7.5),  # "4U 1626-67"
#              (-12.0, -7.5),  # "4U 1916-053"
#              (-9.50, -7.5),  # "4U 1636-536"
#              (-9.25, -7.5),  # "GX 9+9"
#              (-8.95, -7.5),  # "4U 1735-444"
#              (-8.35, -7.0),  # "2A 1822-371"
#              (-8.45, -7.0),  # "Sco X-1"
#              (-8.45, -7.0),  # "GX 349+2"
#              (-8.40, -6.98)]  # "Cyg X-2"


# The effective temperature range of the observed systems in K, if an effective
# temperature is not known then we set the range to be arbitrarily large to not
# exclude any systems

unique_T = [(0.00, 1.e9),    # "4U 0513-40"
            (0.00, 1.e9),    # "2S 0918-549"
            (0.00, 1.e9),    # "4U 1543-624"
            (0.00, 1.e9),    # "4U 1850-087"
            (0.00, 1.e9),    # "M15 X-2"
            (0.00, 1.e9),    # "4U 1626-67"
            (0.00, 1.e9),    # "4U 1916-053"
            (0.00, 1.e9),    # "4U 1636-536"
            (0.00, 1.e9),    # "GX 9+9"
            (0.00, 1.e9),    # "4U 1735-444"
            (0.00, 1.e9),    # "2A 1822-371"
            (0.00, 4800),    # "Sco X-1"
            (0.00, 1.e9),    # "GX 349+2"
            (7000, 8500)]    # "Cyg X-2"

# Open the traceback file to read in inputs
tb_file_name = "dc_template.dat"
input_file = open(tb_file_name, 'r')

# Grab the initalization quantities
num_p = int(float(input_file.readline().strip()))
num_m = int(float(input_file.readline().strip()))
xmax = float(input_file.readline().strip())
xmin = float(input_file.readline().strip())
ymax = float(input_file.readline().strip())
ymin = float(input_file.readline().strip())
dp = float(input_file.readline().strip())
dm = float(input_file.readline().strip())

# Initalize the lists we will be populating
period_from_file = []
mass_from_file = []
zz = []

p_bin = []
m_bin = []

count = 0

# Stream the input data through
for line in input_file:
    # Strip the newline at the end of each line
    linewithoutslashn = line.strip()
    # Split the line at the white spaces
    columns = linewithoutslashn.split()

    # Grab the appropriate values
    period1 = np.float128(columns[0])
    mass1 = np.float128(columns[1])

    # Append the values
    period_from_file.append(period1)
    mass_from_file.append(mass1)

# Close the input file
input_file.close()

# Find all of the mass directories and sort
mass_dirs = next(os.walk("."))[1]
mass_dirs.sort(key=val_find.natural_keys)
mass_vals = []

# Pull the unit portion of the directory name and convert to array
for mass in mass_dirs:
    mass_vals.append(float(mass[0:-1]))
mass_vals = np.array(mass_vals)

# Find all of the period directories and sort
pers_dirs = next(os.walk(mass_dirs[0]))[1]
pers_dirs.sort(key=val_find.natural_keys)
pers_vals = []

# Convert the directory strings to floats and convert to log values
for pers in pers_dirs:
    split_pers = pers.split("_")
    per_to_append = float(split_pers[0][0:-1])
    pers_vals.append([per_to_append,
                     round(np.log10(per_to_append / 1.00), 2)])

# Sort the period values then convert to numpy array
pers_vals.sort()
pers_vals = np.array(pers_vals)

# Grab the mass and period values from the input file
mass_from_file = np.asarray(mass_from_file)
period_from_file = np.asarray(period_from_file)

# Make sure that we remain within our parameter space
low_mass_ind = np.where(mass_from_file < 8)[0]
short_period_ind = np.where(period_from_file < 5)[0]
valid_ind = set(low_mass_ind) & set(short_period_ind)

# Generate empty arrays for the duty cycle (dc), observable age(oa),
# detectable age (da) and max age
dc = np.zeros((len(mass_from_file), len(unique_P)))
oa = np.zeros((len(mass_from_file), len(unique_P)))
da = np.zeros((len(mass_from_file)))
max_age = np.zeros((len(mass_from_file)))

# Loop through all the progenitor combinations in our input file
for binary_index in range(len(mass_from_file)):
    # Make sure its within our parameter range
    if binary_index in valid_ind:
        # Output the mass and period as a sanity check
        print(mass_from_file[binary_index])
        print(period_from_file[binary_index])

        # Define the progenitor mass and period
        tb_mass = mass_from_file[binary_index]
        tb_pers = period_from_file[binary_index]

        # Figure out which mass and period directory combination match with
        # our progenitor model
        mdir = mass_dirs[np.where(tb_mass <= mass_vals)[0][0]]
        subdirs = next(os.walk(mdir))[1]
        subdirs.sort(key=val_find.natural_keys)
        try:
            pdir_ind = np.where(tb_pers <= pers_vals[:, 1])[0][0]
        except IndexError:
            pdir_ind = np.where(tb_pers <= pers_vals[:, 1])[0]

        per_str = format(pers_vals[:, 0][pdir_ind], '.2f')
        pdir = [s for s in subdirs if per_str in s]
        pdir.sort(key=val_find.natural_keys)
        if len(pdir) == 0:
            per_str = format(pers_vals[:, 0][pdir_ind], '.0f')
            pdir = [s for s in subdirs if per_str in s]
            pdir.sort()

        # Generate the path to the model output
        path_to_file = mdir + '/' + pdir[0]
        # print(path_to_file)

        # Try to read in the model data, there are some cases where the
        # data doesn't exist, in these cases just move onto the next
        # mass and period combination
        try:
            m1 = ms.history_data(path_to_file + '/LOGS')

            # Pull the relevant data from the simulated model
            mt = m1.get('lg_mtransfer_rate')
            age = m1.get('star_age')
            period_days = m1.get('period_days')
            period_hr = period_days * 24.0
            mass_1 = m1.get('star_1_mass')
            mass_2 = m1.get('star_2_mass')
            mass_ratio = mass_1 / mass_2
            log_T = m1.get('log_Teff')
            Teff = 10**log_T

            # The last age entry is the maximum age of the system
            max_age[binary_index] = age[-1]

            # Calculate the critical mass transfer rate for the system
            # to be defined as persistent
            # https://ui.adsabs.harvard.edu/abs/2012MNRAS.424.1991C/abstract
            mt_crit = (1.6e-26) * (2.0e15) * (period_hr)**1.76
            mt_quies = mt_crit

            # Determine the indicies where the model mass transfer rate
            # exceeds the critival value
            detectable_inds = np.where(mt >= np.log10(mt_quies))
            detectable_age = age[detectable_inds]

            # Make sure that there is more than one data point above the
            # detectability limit and isnt just noise.
            detectable_chunk = val_find.Find_Consecutive(detectable_inds[0], 3)

            # Calculate the amount of time the system spends in a
            # persistent state
            detectable_age = 0
            for d_chunk in detectable_chunk:
                if len(d_chunk):
                    lower_d_lim = age[min(d_chunk)]
                    upper_d_lim = age[max(d_chunk)]
                    d_age = (upper_d_lim - lower_d_lim)
                    detectable_age += d_age

            # If the system spends any time in a detectable state
            if detectable_age > 0:

                # Increase the detectable age
                da[binary_index] += int(detectable_age)
                duty_cycle = 0
                observed_age = 0

                # Loop through the observed systems
                for obs_index in range(len(unique_P)):

                    # Grab the name and properties of interest
                    sys_name = title_list[obs_index]
                    P_bin = unique_P[obs_index]
                    q_bin = unique_q[obs_index]
                    MT_bin = unique_MT[obs_index]
                    T_bin = unique_T[obs_index]

                    min_P = min(P_bin)
                    max_P = max(P_bin)

                    min_q = min(q_bin)
                    max_q = max(q_bin)

                    min_MT = min(MT_bin)
                    max_MT = max(MT_bin)

                    min_T = min(T_bin)
                    max_T = max(T_bin)

                    # Check if the simulated model satisfies any of the
                    # period (P), mass ratio (q), mass transfer (MT) and
                    # temperature (T) condtions
                    P_check = np.where(np.logical_and(np.log10(period_hr) >= min_P,
                                                      np.log10(period_hr) <= max_P))[0]

                    q_check = np.where(np.logical_and(mass_ratio >= min_q,
                                                      mass_ratio <= max_q))[0]

                    MT_check = np.where(np.logical_and(mt >= min_MT,
                                                       mt <= max_MT))[0]

                    T_check = np.where(np.logical_and(Teff >= min_T,
                                                      Teff <= max_T))[0]

                    # Find the indices where all four conditions are satisfied
                    common_inds = set(P_check) & set(q_check) & \
                        set(MT_check) & set(T_check)

                    # Convert the indices back to a list and sort
                    common_inds_list = list(common_inds)
                    common_inds_list.sort()

                    # Output the system name if the conditions are satisfied
                    print('\n')
                    print("System name: " + str(sys_name))
                    print("===============================================")
                    print("Valid Periods: " + str(len(P_check)))
                    print("Valid mass ratios: " + str(len(q_check)))
                    print("Valid MTs: " + str(len(MT_check)))
                    print("Valid Ts: " + str(len(T_check)))
                    print("Observed points: " + str(len(common_inds)))

                    temp_obs_age = 0
                    # If all 4 conditions are satisfied
                    if len(common_inds_list):
                        print('\n')

                        # Make sure that the system satisfies all of these
                        # conditions for at least 3 data points consecutively
                        split_inds = val_find.Find_Consecutive(common_inds_list, 3)
                        for chunk in split_inds:

                            # Determine how long the system is observable for
                            print("lower index: " + str(chunk[0]))
                            print("upper index: " + str(chunk[-1]))
                            lower_age_lim = age[chunk[0]]
                            upper_age_lim = age[chunk[-1]]
                            age_chunk = upper_age_lim - lower_age_lim
                            print("age: " + str(age_chunk))
                            temp_obs_age += age_chunk

                        # Calculate the duty cycle of the model
                        temp_dc = temp_obs_age / detectable_age

                        # output the various values of interest
                        print("duty cycle: " + str(temp_dc))
                        print("observed age: " + str(temp_obs_age))
                        print("detectable age: " + str(detectable_age))

                        # save the values of interest
                        dc[binary_index][obs_index] += temp_dc
                        oa[binary_index][obs_index] += int(temp_obs_age)
                    print("===============================================")
        except IOError:
            print(path_to_file)

# Once the code has run through all the models, we need to output the various
# quantities of interest for each observed system

# Loop through all of the observed systems
for start_ind in range(len(unique_P)):

    # Define the name of the file we'll be outputting to
    prefix = str(start_ind).zfill(3)
    sys_str = title_list[start_ind].replace(" ", "_")
    output_name = prefix + '_' + sys_str + '_dc.dat'

    # Open the file to write to
    output = open(output_name, 'w')
    print(output_name)

    # Write the results to file
    output.write(str(num_p) + '\n')
    output.write(str(num_m) + '\n')
    output.write(str(xmax) + '\n')
    output.write(str(xmin) + '\n')
    output.write(str(ymax) + '\n')
    output.write(str(ymin) + '\n')
    output.write(str(dp) + '\n')
    output.write(str(dm) + '\n')

    for i in range(len(period_from_file)):
        per_str = str(format(period_from_file[i], '.3f')).ljust(10)
        mass_str = str(format(mass_from_file[i], '.3f')).ljust(10)
        dc_str = str(format(dc[i][start_ind], '.8f')).ljust(15)
        oa_str = str(format(oa[i][start_ind], '.1f')).ljust(15)
        da_str = str(format(da[i], '.1f')).ljust(15)
        ma_str = str(format(max_age[i], '.1f')).ljust(15)
        output_line = per_str + mass_str + dc_str + oa_str + da_str + ma_str + '\n'
        output.write(output_line)

    output.close()
