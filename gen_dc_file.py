import numpy as np
import os
import val_find


dc_file_name = "dc_template.dat"
output = open(dc_file_name, 'w')

mass_dirs = next(os.walk("."))[1]
mass_dirs.sort(key=val_find.natural_keys)

mass_vals = [float(mass[:-1]) for mass in mass_dirs]
mass_vals = np.array(mass_vals)
mass_vals.sort()

m_min = float(min(mass_vals))
m_max = float(max(mass_vals))

m_num = len(mass_dirs) + 1
dm = round((m_max - m_max) / m_num, 3)

dm_fine = 0.05
dm_coarse = 0.10

# adjust these values to fit the high or low periods
p_min = -0.6
p_max = 1.63

# adjust the step sizes for the periods
dp = 0.02

period_vals = np.arange(p_min, p_max, dp)

m_total = np.round(np.tile(mass_vals, len(period_vals)), 3)
p_total = np.round(np.repeat(period_vals, len(mass_vals)), 3)

p_combined = np.repeat(period_vals, len(mass_vals))
p_num = len(p_combined)

# Write the results to file
output.write(str(p_num) + '\n')
output.write(str(m_num) + '\n')
output.write(str(m_max) + '\n')
output.write(str(m_min) + '\n')
output.write(str(p_max) + '\n')
output.write(str(p_min) + '\n')
output.write(str(dp) + '\n')
output.write(str(dm_coarse) + '\n')

for i in range(len(m_total)):
    per_str = str(format(p_total[i], '.3f')).ljust(10)
    mass_str = str(format(m_total[i], '.3f')).ljust(10)
    output_line = per_str + mass_str + '\n'
    output.write(output_line)

output.close()
