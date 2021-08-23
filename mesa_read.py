import os
import numpy as np

'''
MESA data reading code adapted from the NuGrid Group
'''


class mesa_profile():
    '''
    read in a profile output from MESA

    Parameters
    ----------
    dir_name : string
        Directory where profile is located
    num: integer
        The profile number of the file to open. Would be denoted as
        profile<num>.dat

    Example
    -------
    Read in MESA profile 5 like this:

    >>> ms_profile = ms.mesa_profile(dir_name="./LOGS", num=5)
    '''

    file_name = ''

    def __init__(self, dir_name='./LOGS', num=1):
        self.dir_name = dir_name

        filename = self.dir_name + '/profile' + str(num) + '.data'
        if not os.path.exists(filename):
            # raise IOError('ERROR: no profile.data file in ' + self.dir_name)
            print('ERROR: no profile.data file found in ' + self.dir_name)

        header_attr = _read_mesafile(filename, only='header_attr')
        num_zones = int(header_attr['num_zones'])
        header_attr, cols, data =\
            _read_mesafile(filename, data_rows=num_zones, only='all')

        self.cols = cols
        self.header_attr = header_attr
        self.data = data

    def __del__(self):
        print('Closing profile tool ...')

    def get(self, str_name):
        '''
        return a column of data with the name str_name.

        Parameters
        ----------
        str_name : string
            Is the name of the column as printed in the
            profile<num>.data or log<num>.data file; get the available
            columns from self.cols (where you repltace self with the
            name of your instance)

        '''

        column_array = self.data[:, self.cols[str_name] - 1].astype('float')
        return column_array


class history_data():
    '''
    Read in the time evolution data output from MESA simulations

    Parameters
    ----------
    dir_name : string
        The directory where the time evolution data is contained
    file_name : string
        The name of the time evolution data file

    Example
    -------
    Read in a history.data file from the LOGS directory

    m1 = ms.history_data("LOGS")
    '''
    dir_name = ''
    file_name = ''
    header = []
    cols = []

    def __init__(self, dir_name='./LOGS', file_name='history.data'):
        self.dir_name = dir_name
        self.file_name = file_name

        if not os.path.exists(self.dir_name + '/' + self.file_name):
            raise IOError('ERROR: no history.data file in ' + self.dir_name)
            # print('ERROR: no history.data file found in ' + self.dir_name)
        else:
            self._read_starlog()

    def __del__(self):
        print('Closing ' + self.file_name + ' tool ...')

    def _read_starlog(self):
        ''' read history.data or star.log file again'''

        dir_name = self.dir_name
        file_name = self.file_name
        afile_name = file_name

        if not os.path.exists(dir_name + '/' + afile_name):
            print('No ' + self.file_name + ' file found')
            _cleanstarlog(dir_name + '/' + file_name)
        else:
            print('Using old ' + self.file_name + ' file ...')

        cmd = os.popen('wc ' + dir_name + '/' + afile_name)
        cmd_out = cmd.readline()
        cnum_cycles = cmd_out.split()[0]
        num_cycles = int(cnum_cycles) - 6

        filename = dir_name + '/' + afile_name

        header_attr, cols, data = \
            _read_mesafile(filename, data_rows=num_cycles)

        self.cols = cols
        self.header_attr = header_attr
        self.data = data

    def get(self, str_name):
        '''
        return a column of data with the name str_name.

        Parameters
        ----------
        str_name : string
            The name of the column as printed in history.data or
            star.log get the available columns from self.cols (where
            you repltace self with the name of your instance
        '''

        column_array = self.data[:, self.cols[str_name] - 1].astype('float')
        return column_array


def _read_mesafile(filename, data_rows=0, only='all'):
    ''' private routine that is not directly called by the user'''
    f = open(filename, 'r')
    vv = []
    v = []
    lines = []
    line = ''
    for i in range(0, 6):
        line = f.readline()
        lines.extend([line])

    hval = lines[2].split()
    hlist = lines[1].split()
    header_attr = {}
    for a, b in zip(hlist, hval):
        header_attr[a] = str(b)
    if only is 'header_attr':
        return header_attr

    cols = {}
    colnum = lines[4].split()
    colname = lines[5].split()
    for a, b in zip(colname, colnum):
        cols[a] = int(b)

    data = []

    for i in range(data_rows):
        line = f.readline()
        v = line.split()
        try:
            vv = np.array(v, dtype='float64')
        except ValueError:
            for item in v:
                if item.__contains__('.') and not item.__contains__('E'):
                    v[v.index(item)] = '0'
        data.append(vv)

    print(' \n')
    f.close()
    a = np.array(data)
    data = []
    return header_attr, cols, a


def _cleanstarlog(file_in):
    '''
    cleaning history.data or star.log file, e.g. to take care of
    repetitive restarts.

    private, should not be called by user directly

    Parameters
    ----------
    file_in : string
        Typically the filename of the mesa output history.data or
        star.log file, creates a clean file called history.datasa or
        star.logsa.

    '''

    file_out = file_in
    f = open(file_in)
    lignes = f.readlines()
    f.close()

    nb = np.array([], dtype=int)   # model number
    nb = np.concatenate((nb, [int(lignes[len(lignes) - 1].split()[0])]))
    nbremove = np.array([], dtype=int)   # model number
    i = -1

    for i in np.arange(len(lignes) - 1, 0, -1):
        line = lignes[i - 1]

        if i > 6 and line != "":
            if int(line.split()[0]) >= nb[-1]:
                nbremove = np.concatenate((nbremove, [i - 1]))
            else:
                nb = np.concatenate((nb, [int(line.split()[0])]))
    i = -1
    for j in nbremove:
        lignes.remove(lignes[j])

    fout = file(file_out, 'w')
    for j in np.arange(len(lignes)):
        fout.write(lignes[j])
    fout.close()
