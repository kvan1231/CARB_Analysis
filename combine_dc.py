import numpy as np
import glob

output_name = 'combined_dc.dat'
output = open(output_name, 'w')

dc_files = glob.glob('0*dc.dat')
dc_files.sort()

for num in range(len(dc_files)):
    data_file = dc_files[num]

    f = open(data_file, 'r')

    num_p = int(float(f.readline().strip()))
    num_mt = int(float(f.readline().strip()))
    pini_max = float(f.readline().strip())
    pini_min = float(f.readline().strip())
    m_max = float(f.readline().strip())
    m_min = float(f.readline().strip())
    dp = float(f.readline().strip())
    dmt = float(f.readline().strip())

    temp_p = []
    temp_mt = []
    temp_dc = []
    temp_oa = []
    temp_da = []

    count = 0

    per_crit = []

    for line in f:
        linewithoutslashn = line.strip()
        columns = linewithoutslashn.split()

        file_p = np.float128(columns[0])
        file_mt = np.float128(columns[1])
        file_dc = np.float128(columns[2])
        file_oa = np.float128(columns[3])
        file_num = np.float128(columns[4])

        temp_p.append(file_p)
        temp_mt.append(file_mt)
        temp_dc.append(file_dc)
        temp_oa.append(file_oa)
        temp_da.append(file_num)

    if num == 0:
        period = np.array(temp_p)
        mtransfer = np.array(temp_mt)
        dc = np.array(temp_dc)
        oa = np.array(temp_oa)
        da = np.array(temp_da)
    else:
        for row in range(len(dc)):
            dc[row] = max(dc[row], temp_dc[row])
            oa[row] = oa[row] + temp_oa[row]
            da[row] = da[row] + temp_da[row]

    f.close()

# Write the results to file
output.write(str(num_p) + '\n')
output.write(str(num_mt) + '\n')
output.write(str(pini_max) + '\n')
output.write(str(pini_min) + '\n')
output.write(str(m_max) + '\n')
output.write(str(m_min) + '\n')
output.write(str(dp) + '\n')
output.write(str(dmt) + '\n')

for i in range(len(period)):
    per_str = str(format(period[i], '.3f')).ljust(10)
    mdot_str = str(format(mtransfer[i], '.3f')).ljust(10)
    dc_str = str(format(dc[i], '.8f')).ljust(15)
    oa_str = str(format(oa[i], '.1f')).ljust(15)
    num_str = str(format(da[i], '.1f'))
    output_line = per_str + mdot_str + dc_str + oa_str + num_str + '\n'
    output.write(output_line)

output.close()
