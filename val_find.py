import mesa_read as ms
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
import re
from operator import itemgetter
from itertools import groupby


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in 'human order'
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (see implementation by Toothy in comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def Find_Consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def Find_EvenSpacing(input_list, spacing):
    '''
    Poorly defines locations that are considered "evenly spaced" along an HRD

    Parameters
    ----------
    input_list : array_like
        The list of values to be evenly spaced
    spacing : int
        The size of the gap between data points

    Returns
    -------
    index_list: array_like
        The indicies of the "evenly spaced" points from the original dataset
    '''

    index_list = []
    i = 0
    j = 0
    while (j < len(input_list)) and (i < len(input_list)):
        power = int(round(math.log10(spacing)))
        value = abs(round(input_list[i] - input_list[j], -power))
        # print(i)
        # print(j)
        # print(value)
        if value >= spacing:
            index_list.append(i)
            i = j
        else:
            j = j + 1
    index_list.append(i)
    index_list.append(j - 1)

    # return return_list
    return index_list


def Find_Value(root_dir=".", value=None):
    '''
    Finds the initial and final value of a user given property. If the
    system is broken return "broken".

    Parameters
    ----------
    root_dir : string
        Path to the directory containing the history.data file to be used

    '''
    value_matrix = [["directory", "initial", "final"]]
    # value = raw_input("what value? ")
    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "history.data":
                try:
                    m1 = ms.history_data(subdir)
                    x_i = m1.get(value)[0]
                    x_f = m1.get(value)[-1]
                except OSError:
                    x_i = "broken"
                    x_f = "broken"

                # preface = subdir + " : "
                x_i_string = "initial " + str(value) + " : " + str(x_i)
                x_f_string = "final " + str(value) + " : " + str(x_f)

                # print(preface)
                # print(x_i_string)
                # print(x_f_string)

                value_matrix.append([subdir, x_i_string, x_f_string])

    return value_matrix


def Find_F_Properties(root_dir="."):
    '''
    Find the following properties:

        Initial and final helium core fraction
        Final donor mass
        Final companion mass
        Initial radius of donor
        Final radius of donor
        Final orbital period
        Final model number

    Can easily add more properties as necessary

    Parameters
    ----------
    root_dir : string
        Path to the directory containing the history.data file to be used

    '''
    value_matrix = [["directory",
                     "initial he core fraction",
                     "final star 1 mass",
                     "final star 2 mass",
                     "initial radius",
                     "final radius",
                     "initial period",
                     "final period",
                     "evolution time",
                     "final model number"]]

    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "history.data":
                m1 = ms.history_data(subdir)
                # want various values
                # initial center he core frac
                X_i = m1.get("center_he4")[0]

                # # initial star age
                t_i = m1.get("star_age")[0]

                # final masses
                M1_f = m1.get("star_1_mass")[-1]
                M2_f = m1.get("star_2_mass")[-1]

                # final period
                P_i = m1.get("period_days")[0]
                P_f = m1.get("period_days")[-1]

                # evolution time
                dt = m1.get("star_age")[-1] - t_i

                # radius
                r_i = 10**m1.get("log_R")[0]
                r_f = 10**m1.get("log_R")[-1]

                # model number
                model = m1.get("model_number")[-1]

                # preface = subdir + " : "
                # X_i_string = "initial he core frac: " + str(X_i)
                # M1_f_string = "final star 1 mass: " + str(M1_f)
                # M2_f_string = "final star 2 mass: " + str(M2_f)
                # r_i_string = "initial radius: " + str(r_i)
                # r_f_string = "final radius: " + str(r_f)
                # P_i_string = "initial period: " + str(P_i)
                # P_f_string = "final period: " + str(P_f)
                # dt_string = "evolution time: " + str(dt)
                # model_string = "final model number: " + str(model)
                # print(preface)
                # print(X_i_string)
                # print(M1_f_string)
                # print(M2_f_string)
                # print(r_i_string)
                # print(r_f_string)
                # print(P_i_string)
                # print(P_f_string)
                # print(dt_string)
                # print(model_string)

                value_matrix.append([subdir, X_i, M1_f, M2_f,
                                     r_i, r_f, P_i, P_f, dt, model])

    return value_matrix


def Find_RLOF_Prop(root_dir="."):
    '''
    Determine the value of various properties at the beginning of RLOF
    and at the end of RLOF.

    NOTE: The values representing the "end" of RLOF may not be correct,
          The star can easily stop transferring mass and restart multiple
          times, therefore the "end" value represents the last value in
          the simulations that had a relative RLOF greater than 0

          Values given:
            Initial time of RLOF
            Final donor mass after RLOF
            Final companion mass after RLOF
            Initial radius at the start of RLOF of donor
            Final radius after RLOF of donor
            Final period after RLOF
            Total time or RLOF

    Parameters
    ----------
    root_dir : string
        Path to the directory containing the history.data file to be used

    '''
    value_matrix = [["directory",
                     "rlof start",
                     "final star 1 mass",
                     "final star 2 mass",
                     "initial radius",
                     "final radius",
                     "initial period",
                     "final period",
                     "evolution time"]]

    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "history.data":
                m1 = ms.history_data(subdir)

                preface = subdir + " : "
                print(preface)

                try:
                    m1rlof = m1.get('rl_relative_overflow_1')
                    rlof_index = np.where(m1rlof > 0)[0]
                    # rlof_start = []
                    rlof_ends = []

                    i = 0
                    while ((rlof_index[i + 1] - rlof_index[i]) < 100)\
                            and (i < len(rlof_index) - 2):

                        rlof_ends.append(rlof_index[i])
                        i = i + 1

                    if len(rlof_ends) == 0:
                        rlof_ends = [rlof_index[-1]]
                    rlof_start = rlof_index[1]
                    rlof_end = rlof_ends[-1]

                    # # initial star age
                    t_i = m1.get("star_age")[rlof_start]

                    # final masses
                    M1_f = m1.get("star_1_mass")[rlof_end]
                    M2_f = m1.get("star_2_mass")[rlof_end]

                    # final period
                    P_f = m1.get("period_days")[rlof_end]
                    P_i = m1.get("period_days")[rlof_start]

                    # evolution time
                    dt = m1.get("star_age")[rlof_end] - t_i

                    # radius
                    r_i = 10**m1.get("log_R")[rlof_start]
                    r_f = 10**m1.get("log_R")[rlof_end]

                    # t_i_string = "RLOF  starts at: " + str(t_i)
                    # M1_f_string = "final star 1 mass: " + str(M1_f)
                    # M2_f_string = "final star 2 mass: " + str(M2_f)
                    # r_i_string = "initial radius: " + str(r_i)
                    # r_f_string = "final radius: " + str(r_f)
                    # P_i_string = "initial period: " + str(P_i)
                    # P_f_string = "final period: " + str(P_f)
                    # dt_string = "RLOF time: " + str(dt)
                    # print(t_i_string)
                    # print(M1_f_string)
                    # print(M2_f_string)
                    # print(r_i_string)
                    # print(r_f_string)
                    # print(P_i_string)
                    # print(P_f_string)
                    # print(dt_string)
                    value_matrix.append([subdir, t_i, M1_f, M2_f,
                                         r_i, r_f, P_i, P_f, dt])
                except KeyError:
                    print("no RLOF")
    return value_matrix


def Find_Mod_MB(root_dir, check_mod=True):
    '''
    Loops through the directories to determine of a system correctly has
    the modified magnetic braking scheme initialized

    Parameters
    ----------
    root_dir : string
        Path to the directory containing the history.data file to be used
    '''
    Mod_MB_list = []

    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "inlist":
                file_to_open = subdir + "/" + file_name
                inlist = open(file_to_open, 'r')
                inlist.seek(0)
                data = inlist.readlines()
                modified_MB = "!" not in data[15]
                in_src = "src" not in file_to_open
                check_file = in_src * modified_MB
                if check_file:
                    if check_mod:
                        pass
                    else:
                        Mod_MB_list.append(subdir)
                        # print(subdir)
                else:
                    if check_mod:
                        if not in_src:
                            pass
                        else:
                            Mod_MB_list.append(subdir)
                            # print(subdir)
                    else:
                        pass
                inlist.close()
    return Mod_MB_list


def Find_RLOF_Check(root_dir="."):
    '''
    Loops through the directories to determine of a system goes into RLOF.
    If it does then list the time it goes into RLOF and if it is still in
    that state

    Parameters
    ----------
    root_dir : string
        Path to the directory containing the history.data file to be used
    '''

    rlof_cont = ["rlof ongoing"]
    no_rlof = ["no rlof"]

    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "history.data":
                # print(subdir)
                try:
                    m1 = ms.history_data(subdir)
                except OSError:
                    print("Convergence issues")

                preface = subdir + " : "
                print(preface)

                try:
                    m1rlof = m1.get('rl_relative_overflow_1')
                    rlof_index = np.where(m1rlof > 0)[0]
                    # rlof_start = []
                    rlof_ends = []

                    i = 0
                    while ((rlof_index[i + 1] - rlof_index[i]) < 100)\
                            and (i < len(rlof_index) - 2):

                        rlof_ends.append(rlof_index[i])
                        i = i + 1

                    if len(rlof_ends) == 0:
                        rlof_ends = [rlof_index[-1]]
                    rlof_start = rlof_index[1]
                    rlof_end = rlof_ends[-1]

                    # print("RLOF Success")

                    # # initial star age
                    t_i = m1.get("star_age")[rlof_start]

                    # evolution time
                    dt = m1.get("star_age")[rlof_end] - t_i

                    if m1rlof[rlof_end] == m1rlof[-1]:
                        rlof_cont.append(subdir + " " + str(dt))
                        # print("RLOF still ongoing")
                except KeyError:
                    pass
                    # print("no RLOF")
    # print("RLOF still ongoing: ")
    # for line in rlof_cont:
    #     print(line)

    # print("")
    # print("No RLOF: ")
    # for line in no_rlof:
    #     print(line)

    return rlof_cont, no_rlof


def Find_Bifurcation(root_dir="."):

    mass = []
    bifur_period = []
    ordered_period_list = []
    convergence = []
    mass_dirs = os.walk(root_dir).next()[1]
    mass_dirs.sort()

    for mass_dir in mass_dirs:
        bp_found = False
        print(mass_dir)
        mass.append(mass_dir)
        os.chdir(mass_dir)
        period_dirs = []
        sub_dirs = filter(os.path.isdir, os.listdir(os.getcwd()))
        for directory in sub_dirs:
            split_vals = directory.split('_')
            period_val = split_vals[0][0:-1]
            period_dirs.append(float(period_val))
        sub_dirs = np.array(sub_dirs[0:len(period_dirs)])
        period_dirs = np.array(period_dirs)

        period_inds = period_dirs.argsort()
        sorted_dirs = sub_dirs[period_inds]
        ordered_period_list.append(sorted_dirs)
        sub_converge = []

        os.chdir("../")

        print(sorted_dirs)

        for subdir in sorted_dirs:
            print(subdir)
            try:
                file_str = mass_dir + "/" + subdir + "/LOGS"
                m1 = ms.history_data(file_str)
                period = m1.get('period_days')
                star_mass = m1.get('star_1_mass')
                rlof = m1.get('rl_relative_overflow_1')
                rlof_index = np.where(rlof > 0)[0][0]
                test_ind = np.where(star_mass < max(star_mass) / 2)[0][0]
                max_period = period[-1]
                # max_period = max(period)
                if max_period > period[rlof_index]:
                    sub_converge.append(False)
                else:
                    sub_converge.append(True)
            except OSError:
                sub_converge.append(True)

        convergence.append(sub_converge)

        i = 0
        while not bp_found and i < len(sub_converge) - 1:
            if sub_converge[i] != sub_converge[i + 1]:
                print(sorted_dirs[i], sorted_dirs[i + 1])
                bifur_period.append((sorted_dirs[i], sorted_dirs[i + 1]))
                bp_found = True
            else:
                i += 1

    return mass, ordered_period_list, convergence, bifur_period


def Find_Outburst(root_dir=".", outburst_tol=0.1, number_tol=0, time_tol=1e6,
                  plot=False):
    outburst_list = []
    outnum_list = []
    for subdir, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name == "history.data":
                # print(subdir)
                try:
                    outburst_num = 0
                    m1 = ms.history_data(subdir)
                    m1mt1 = 10**m1.get('lg_mstar_dot_1')[0:-1]
                    m1mt2 = 10**m1.get('lg_mstar_dot_1')[1:]
                    delta_MT = abs(m1mt1 - m1mt2)
                    percent_delta = (delta_MT / m1mt1)
                    m1_age = m1.get('star_age')
                    # model = m1.get('model_number')

                    delta_list = np.where(percent_delta >= outburst_tol)[0]
                    for k, g in groupby(enumerate(delta_list),
                                        lambda i, x: i - x):
                        group = map(itemgetter(1), g)
                        if len(group) > 2:
                            delta_list = np.setdiff1d(delta_list, group)
                    i = 0
                    tries = 10
                    while i <= len(delta_list) - 2 and tries > 0 \
                            and len(delta_list) > 1:
                        if abs(m1_age[delta_list[i]] -
                               m1_age[delta_list[i + 1]]) <= time_tol:
                            # print(abs(m1_age[i] - m1_age[i + 1]))
                            # print('restart')
                            delta_list = np.delete(delta_list, i)
                            i = 0
                            tries -= 1
                        else:
                            tries = 10
                            i += 1
                    if plot:
                        plt.clf()
                        plt.plot(m1_age[1:],
                                 np.log10(percent_delta))
                        if len(delta_list) >= 2:
                            plt.plot(m1_age[1:][delta_list],
                                     np.log10(percent_delta)[delta_list],
                                     'or')
                    outburst_num = len(delta_list) // 2
                    print(subdir, outburst_num)
                    if outburst_num > number_tol:
                        outburst_list.append(subdir)
                        outnum_list.append(outburst_num)

                except OSError:
                    print("Convergence issues")
    out_ind = np.array(outnum_list).argsort()[::-1]
    sorted_burst = np.array(outburst_list)[out_ind]
    sorted_num = np.array(outnum_list)[out_ind]
    results = np.array([sorted_burst, sorted_num]).T
    return results


def Find_Sco_Like(root_dir=".", output_file='TB_values.dat'):
    m1 = ms.history_data(root_dir)
    period = m1.get('period_days')
    q = m1.get('star_1_mass') / m1.get('star_2_mass')
    # rlof = m1.get('rl_relative_overflow_1')
    mt = m1.get('lg_mtransfer_rate')
    # q_diff = abs(q - 0.30)
    pdiff = abs(period * 24 - 18.89551)
    qpcheck = np.where(np.logical_and(q < 0.5, pdiff < 0.1))
    sco_ind = qpcheck[0][np.where(min(pdiff[qpcheck]) == pdiff[qpcheck])[0][0]]
    RLOF_ind = np.where(mt > -12)[0][0]

    Sco_donor = str(round(m1.get('star_1_mass')[sco_ind], 2))
    Sco_ns = str(round(m1.get('star_2_mass')[sco_ind], 2))
    Sco_q = str(round(q[sco_ind], 2))
    Sco_P = str(round(m1.get('period_days')[sco_ind], 5))
    Sco_mt = str(round(10**(m1.get('lg_mtransfer_rate')[sco_ind]), 12))
    Sco_T = str(round(10**(m1.get('log_Teff')[sco_ind]), 2))
    Sco_c = str(round(m1.get('surface_c12')[sco_ind], 5))
    Sco_o = str(round(m1.get('surface_o16')[sco_ind], 5))

    RLOF_donor = str(round(m1.get('star_1_mass')[RLOF_ind], 2))
    RLOF_ns = str(round(m1.get('star_2_mass')[RLOF_ind], 2))
    RLOF_q = str(round(q[RLOF_ind], 2))
    RLOF_P = str(round(m1.get('period_days')[RLOF_ind], 5))
    RLOF_mt = str(round(10**(m1.get('lg_mtransfer_rate')[RLOF_ind]), 12))
    RLOF_T = str(round(10**(m1.get('log_Teff')[RLOF_ind]), 2))
    RLOF_c = str(round(m1.get('surface_c12')[RLOF_ind], 5))
    RLOF_o = str(round(m1.get('surface_o16')[RLOF_ind], 5))

    try:
        init_mass = str(round(float(root_dir.split("/")[0][0:-1]), 2))
        init_p_flt = float(root_dir.split("/")[1].split("_")[1][0:-1])
        init_per = str(round(init_p_flt / 1.1, 2))
    except ValueError:
        init_mass = " "
        init_per = " "

    Sco_string = ' & ' +\
        Sco_donor + ' & ' +\
        Sco_ns + ' & ' +\
        Sco_q + ' & ' +\
        Sco_P + ' & ' +\
        Sco_mt + ' & ' +\
        Sco_T + ' & ' +\
        Sco_c + ' & ' +\
        Sco_o + '\\\\'
    RLOF_string = init_mass + ' & ' +\
        init_per + ' & ' +\
        RLOF_donor + ' & ' +\
        RLOF_ns + ' & ' +\
        RLOF_q + ' & ' +\
        RLOF_P + ' & ' +\
        RLOF_mt + ' & ' +\
        RLOF_T + ' & ' +\
        RLOF_c + ' & ' +\
        RLOF_o

    f = open(output_file, 'a+')
    f.write(RLOF_string + Sco_string)
    # f.write(root_dir + '\n')
    # f.write("Teff = " + str(Sco_T[0]) + '\n')
    # f.write("Donor M = " + str(Sco_donor[0]) + '\n')
    # f.write("NS M = " + str(Sco_ns[0]) + '\n')
    # f.write("Period = " + str(Sco_P[0]) + '\n')
    # f.write("Sco_mt = " + str(Sco_mt[0]) + '\n')
    f.write('\n')
    f.close()


def Find_TB_init(file_name, output_file='init_TB.dat',
                 save_file=True, full_output=True, shift=True,
                 zdata='oa', dm_scale=True):
    f = open(file_name, 'r')

    num_p = int(float(f.readline().strip()))
    num_m = int(float(f.readline().strip()))
    xmax = float(f.readline().strip())
    xmin = float(f.readline().strip())
    ymax = float(f.readline().strip())
    ymin = float(f.readline().strip())
    dp = float(f.readline().strip())
    dm = float(f.readline().strip())

    period = []
    mass = []
    zz = []
    frac = []

    p_bin = []
    m_bin = []

    count = 0

    for line in f:
        linewithoutslashn = line.strip()
        columns = linewithoutslashn.split()

        period1 = np.float128(columns[0])
        mass1 = np.float128(columns[1])

        try:
            dc = np.float128(columns[2])
            log_dc = np.log10(np.maximum(np.array(dc), 1.e-20))

            oa = np.float128(columns[3])
            log_oa = np.log10(np.maximum(np.array(oa), 1.e-20))

            da = np.float128(columns[4])
            log_da = np.log10(np.maximum(np.array(da), 1.e-20))

            ta = np.float128(columns[5])
            log_ta = np.log10(np.maximum(np.array(ta), 1.e-20))

            if zdata == 'dc':
                dens10 = log_dc
                dens1 = dc
            elif zdata == 'oa':
                dens10 = log_oa
                dens1 = oa
            elif zdata == 'da':
                dens10 = log_da
                dens1 = da
            # elif zdata == 'ta':
            #     dens10 = log_ta
            #     dens1 = ta

        except IndexError:
            dc = np.float128(columns[2])
            log_dc = np.log10(np.maximum(np.array(dc), 1.e-20))
            dens10 = log_dc
            dens1 = dc

        period.append(period1)
        mass.append(mass1)

        zz.append(dens10)

        frac.append(da / ta)

        if dm_scale:
            if mass1 >= 4.9:
                dm_mult = 10.0
            elif 5.0 > mass1 > 2.9:
                dm_mult = 5.0
            elif 2.9 > mass1 > 2.3:
                dm_mult = 2.0
            else:
                dm_mult = 1.0
        else:
            dm_mult = 1.0

        p_bin.append(round(dp, 2))
        m_bin.append(round(dm * dm_mult, 2))

        count = count + 1

    per_array = np.array(period)
    m_array = np.array(mass)
    zz_array = np.array(zz)

    dp_array = np.array(p_bin)
    dm_array = np.array(m_bin)
    f_array = np.array(frac)

    inds = np.where(zz_array > -20)[0]

    init_per = per_array[inds]
    init_m = m_array[inds]
    if shift:
        init_m_shift = init_m + 0.09
    else:
        init_m_shift = init_m
    init_zz = zz_array[inds]
    init_dp = dp_array[inds]
    init_dm = dm_array[inds]
    init_f = f_array[inds]

    if full_output:
        res = np.array([init_per, init_dp, init_m_shift, init_dm, init_zz, init_f]).T
        if save_file:
            np.savetxt(output_file, res, fmt="%1.2f, %1.2f, %1.2f, %1.2f, %1.7f, %1.7f")
        else:
            return res
    else:
        res = np.array([init_per, init_m, init_zz, init_f]).T
        if save_file:
            np.savetxt(output_file, res, fmt="%1.2f, %1.2f, %1.7f, %1.7f")
        else:
            return res


def Find_Combined_TB(file_names, output_file='combined_TB.dat'):
    res1 = Find_TB_init(file_names[0], save_file=False)
    combined_set = set([tuple(x) for x in res1])

    for file_name in file_names[1:]:
        temp_res = Find_TB_init(file_name, save_file=False)
        temp_set = set([tuple(x) for x in temp_res])
        combined_set = combined_set & temp_set

    combined_res = np.array([x for x in combined_set])
    np.savetxt(output_file, combined_res, fmt="%s")

    return combined_res


def Find_TB_Props(x=(0.3, 0.6), y=(-0.5, 0)):
    x_min = float(min(x))
    x_max = float(max(x))
    y_min = float(min(y))
    y_max = float(max(y))

    sys_of_interest = []
    mass_dirs = os.walk(".").next()[1]
    mass_dirs.sort()
    mass_vals = []

    for mass in mass_dirs:
        mass_vals.append(float(mass[0:-1]))
    mass_vals = np.array(mass_vals)

    pers_dirs = os.walk(mass_dirs[0]).next()[1]
    pers_vals = []
    for pers in pers_dirs:
        split_pers = pers.split("_")
        per_to_append = float(split_pers[1][0:-1])
        pers_vals.append([per_to_append,
                         round(np.log10(per_to_append / 1.10), 2)])
    pers_vals.sort()
    pers_vals = np.array(pers_vals)

    TB_file = pd.read_csv("Sco_init", sep=" ", names=["period", "mass"])
    mass = TB_file.mass.values
    pers = TB_file.period.values

    inds = mass.argsort()

    sorted_mass = mass[inds]
    sorted_pers = pers[inds]

    i = 0
    for i in xrange(len(sorted_mass)):
        tb_mass = sorted_mass[i]
        tb_pers = sorted_pers[i]

        mdir = mass_dirs[np.where(tb_mass < mass_vals)[0][0]]
        subdirs = os.walk(mdir).next()[1]
        pdir_ind = np.where(tb_pers < pers_vals[:, 1])[0][0]
        per_str = str(pers_vals[:, 0][pdir_ind])
        pdir = [s for s in subdirs if '_' + per_str in s][0]

        path_to_file = mdir + '/' + pdir

        print(path_to_file)
        m1 = ms.history_data(path_to_file + '/LOGS')

        # final check
        mass_ratio = m1.get('star_1_mass') / m1.get('star_2_mass')
        log_period = np.log10(m1.get('period_days'))
        qcheck = np.where(np.logical_and(mass_ratio >= x_min,
                                         mass_ratio <= x_max))
        pcheck = np.where(np.logical_and(log_period >= y_min,
                                         log_period <= y_max))
        qp_check = np.where(np.in1d(qcheck[0],
                                    pcheck[0]))

        if len(qp_check[0] > 0):
            sys_of_interest.append(path_to_file)
            print(path_to_file)
            Find_Sco_Like(path_to_file + '/LOGS')


def Find_P_q_MT_Ranges(data_file="NS_Range.csv"):

    unique_P = []
    unique_q = []
    unique_MT = []

    all_data = pd.read_csv(data_file)

    low_P = all_data.lowP.values
    high_P = all_data.highP.values
    P_range = np.array([low_P, high_P]).T

    low_q = all_data.lowq.values
    high_q = all_data.highq.values
    q_range = np.array([low_q, high_q]).T

    low_MT = all_data.lowMT.values
    high_MT = all_data.highMT.values
    MT_range = np.array([low_MT, high_MT]).T

    range_array = np.concatenate((P_range, q_range, MT_range), axis=1)

    unique_rows = np.vstack({tuple(row) for row in range_array})
    for row in xrange(len(unique_rows)):
        unique_P.append(unique_rows[row][0:2])
        unique_q.append(unique_rows[row][2:4])
        unique_MT.append(unique_rows[row][4:6])

    return unique_P, unique_q, unique_MT


def Find_Bins(data, sep):
    array_len = len(data)
    rounder = 1. / sep

    bins = np.zeros(shape=(array_len, 2))

    for index in range(array_len):
        value = data[index]
        rounded_value = round(value * rounder) / rounder
        if rounded_value <= value:
            min_val = rounded_value
            max_val = rounded_value + sep
        elif rounded_value >= value:
            min_val = rounded_value - sep
            max_val = rounded_value
        bins[index] = [min_val, max_val]

    return bins
