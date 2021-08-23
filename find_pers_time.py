import mesa_read as ms
import numpy as np
import val_find
import os


"""
This code cycles through the simulated systems to find
which ones result in persistent LMXBs

"""

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

unique_log_p = list(set(period_from_file))
unique_log_p.sort()

# Convert the directory strings to floats and convert to log values
for pers_dir_index in range(len(pers_dirs)):
    pers = pers_dirs[pers_dir_index]
    split_pers = pers.split("_")
    per_to_append = float(split_pers[0][0:-1])
    pers_vals.append([per_to_append,
                      unique_log_p[pers_dir_index]])

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

# Generate empty arrays for the when the system is deemed
# persistent by our MT criteria and for checking if the
# binary detaches by the end of the simulation and total age.
# Grab the mass, radius and luminosity at the start of MT

da = np.zeros((len(mass_from_file)))  # total detectable age
ta = np.zeros((len(mass_from_file)))  # total simulated age

smt = np.zeros((len(mass_from_file)))  # age at the start of MT
mt_M = np.zeros((len(mass_from_file)))  # Mass at start of MT
mt_R = np.zeros((len(mass_from_file)))  # Radius at start of MT
mt_L = np.zeros((len(mass_from_file)))  # Luminosity at start of MT

fmt = np.zeros((len(mass_from_file)))  # MT value at end of sim
detach_array = np.zeros((len(mass_from_file)))

# Loop through all the progenitor combinations in our input file
for binary_index in range(len(mass_from_file)):
    # Make sure its within our parameter range
    if binary_index in valid_ind:
        # Output the mass and period as a sanity check
        # print(mass_from_file[binary_index])
        # print(period_from_file[binary_index])

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
        pdir = pers_dirs[pdir_ind]

        # Generate the path to the model output
        # path_to_file = mdir + '/' + pdir[0]
        path_to_file = mdir + '/' + pdir
        print(path_to_file)

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

            mass = m1.get('star_1_mass')
            radius = 10**m1.get('log_R')
            luminosity = 10**m1.get('log_L')

            start_mt_ind = np.where(mt > -12)[0]

            # Calculate the critical mass transfer rate for the system
            # to be defined as persistent
            # https://ui.adsabs.harvard.edu/abs/2012MNRAS.424.1991C/abstract
            mt_crit = (2.0e-26) * (2.9e15) * (period_hr)**1.76
            mt_quies = mt_crit

            # Determine the indicies where the model mass transfer rate
            # exceeds the critival value
            detectable_inds = np.where(mt >= np.log10(mt_quies))[0]
            detectable_age = age[detectable_inds]

            if len(detectable_inds):
                da[binary_index] = detectable_age[0]

                # Output the system name and if the conditions are satisfied
                print('\n')
                print("Presistent System Found: " + path_to_file)
                print("===============================================")
                print(detectable_age[0])

            if mt[-1] < -12:
                detach_array[binary_index] = 1

            if len(start_mt_ind):
                smt[binary_index] = age[start_mt_ind[0]]
                mt_M[binary_index] = mass[start_mt_ind[0]]
                mt_R[binary_index] = radius[start_mt_ind[0]]
                mt_L[binary_index] = luminosity[start_mt_ind[0]]
            else:
                smt[binary_index] = -1
                mt_M[binary_index] = -99
                mt_R[binary_index] = -99
                mt_L[binary_index] = -99

            ta[binary_index] = age[-1]
            fmt[binary_index] = mt[-1]
        except IOError:
            print(path_to_file)

output_name = 'persistent_sys.dat'

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

    da_str = str(format(da[i], '.1f')).ljust(15)
    ta_str = str(format(ta[i], '.1f')).ljust(15)

    smt_str = str(format(smt[i], '.3f')).ljust(15)
    mt_M_str = str(format(mt_M[i], '.3f')).ljust(15)
    mt_R_str = str(format(mt_R[i], '.3f')).ljust(15)
    mt_L_str = str(format(mt_L[i], '.3f')).ljust(15)

    fmt_str = str(format(fmt[i], '.3f')).ljust(15)
    detatch_str = str(format(detach_array[i], '.1f')).ljust(10)

    output_line = per_str + mass_str + \
        da_str + ta_str + \
        smt_str + mt_M_str + mt_R_str + mt_L_str + \
        fmt_str + detatch_str + '\n'
    output.write(output_line)

output.close()
