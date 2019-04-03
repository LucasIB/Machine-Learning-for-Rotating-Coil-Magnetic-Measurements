'''
Script for load data files for machine learn main software
version: v1.0 
'''
import sys
import numpy as np
import pandas as pd
import os
import random as random
import time
import datetime
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore


class DataFileError(Exception):
    """Data file error"""

    def __init__(self, message, *args):
        """Initialize variables"""
        self.message = message

class Main_Script(object):
    """Rotating coil file data"""
    def __init__(self):
        self.file = np.array([])
        
            
    def load_files(self):
        """Load input database"""
        try:
            app=QtWidgets.QApplication.instance()
            if not app:
                app = QtWidgets.QApplication(sys.argv)
            file_path = QtWidgets.QFileDialog.getOpenFileNames()
            self.files = self.sort_list(file_path[0])
        #self.DataFile()
            return True
        except:
            return False
        
    def sort_list(self,list):
        """Sort data by input date, avoiding wrong information"""
        index = np.array([])
        for i in range(len(list)):
            index = np.append(index, time.mktime(datetime.datetime.strptime(list[i][list[i].find('.dat')-13:list[i].find('.dat')], '%y%m%d_%H%M%S').timetuple()))
        index = index.argsort()

        _file_List = np.array([])
        for i in range(len(list)):
            _file_List = np.append(_file_List,list[index[i]])

        return _file_List
    
    def _search_in_file_lines(self, lines, search_str, alt_search_str=None):
        """Search individual infos from each data"""
        found_in_file = np.where(np.char.find(lines, search_str) > -1)
        if len(found_in_file) == 0 and alt_search_str is not None:
            found_in_file = np.where(
                np.char.find(lines, alt_search_str) > -1)[0]
        if len(found_in_file) != 0:
            index = found_in_file[0][0]
        else:
            index = None
        return index
    
    def DataFile(self):
        """ Rotating coil file data"""
        self.Data = np.array([])
        n = len(self.files)
        for i in range (n):
            self.Data = np.append(self.Data, Main_Script())
            self.Data[i].file = self.files[i]
            arq = open(self.Data[i].file)
            self.Data[i].Raw = np.array(arq.read().splitlines())
            
            #Parse filename
            filename_split = (os.path.split(self.Data[i].file)[1].split('.')[0].split('_'))
            
            #Read Magnet Name
            index = self._search_in_file_lines(self.Data[i].Raw, 'file', 'arquivo')
            if index is not None:
                self.Data[i].magnet_name = (
                    self.Data[i].Raw[index].split('\t')[1].split('\\')[-1].split('_')[0])
            else:
                self.Data[i].magnet_name = filename_split[0]
                
            #Read Date
            index = self._search_in_file_lines(self.Data[i].Raw, 'date', 'data')
            if index is not None:
                self.Data[i].date = (
                    self.Data[i].Raw[index].split('\t')[1].split('\\')[-1].split('_')[0])        
            else:
                if len(filename_split) > 1:
                    self.Data[i].date = filename_split[-2]
            
            #Read Hour
            index = self._search_in_file_lines(self.Data[i].Raw, 'hour', 'hora')
            if index is not None:
                self.Data[i].hour = (
                    self.Data[i].Raw[index].split('\t')[1].split('\\')[-1].split('_')[0])
            else:
                if len(filename_split) > 2:
                    self.Data[i].hour = filename_split[-1]
            
            #Read Measure Number
            index = self._search_in_file_lines(self.Data[i].Raw, 'analysis_interval', 'intervalo_analise')
            if index is not None:
                self.Data[i].measure_number = (
                    self.Data[i].Raw[index].split('\t')[1].split('-'))
                
            #Read Number of Measures Used to Calculate the Mean Value
            index = self._search_in_file_lines(self.Data[i].Raw, 'n_turns', 'nr_voltas')
            if index is not None:
                self.Data[i].measure_number_mean = self.Data[i].Raw[index].split('\t')[1]
                
            #Read Temperature
            index = self._search_in_file_lines(self.Data[i].Raw, 'temperature', 'temperatura_ima')
            if index is not None:
                self.Data[i].temperature = self.Data[i].Raw[index].split('\t')[1]
            
            #Read encoder start pulse
            index = self._search_in_file_lines(self.Data[i].Raw, 'pulse_start_collect', 'pulso_start_coleta')
            if index is not None:
                self.Data[i].start_pulse = self.Data[i].Raw[index].split('\t')[1]
            
            #### Take Currents ####
                
            # Read main current
            index = self._search_in_file_lines(self.Data[i].Raw, 'main_coil_current_avg', 'corrente_alim_principal_avg')
            if index is not None:
                self.Data[i].main_current = float(self.Data[i].Raw[index].split('\t')[1])
            
            index = self._search_in_file_lines(self.Data[i].Raw, 'main_coil_current_std', 'corrente_alim_principal_std')
            if index is not None:
                self.Data[i].main_current_std = float(self.Data[i].Raw[index].split('\t')[1])

            # Read Trim Current
            index = self._search_in_file_lines(self.Data[i].Raw, 'trim_coil_current_avg', 'corrente_alim_secundaria_avg')
            if index is not None:
                self.Data[i].trim_current = float(self.Data[i].Raw[index].split('\t')[1])

            index = self._search_in_file_lines(self.Data[i].Raw, 'trim_coil_current_std', 'corrente_alim_secundaria_std')
            if index is not None:
                self.Data[i].trim_current_std = float(self.Data[i].Raw[index].split('\t')[1])

            # Read CH Current
            index = self._search_in_file_lines(self.Data[i].Raw, 'ch_coil_current_avg')
            if index is not None:
                self.Data[i].ch_current = float(self.Data[i].Raw[index].split('\t')[1])

            index = self._search_in_file_lines(self.Data[i].Raw, 'ch_coil_current_std')
            if index is not None:
                self.Data[i].ch_current_std = float(self.Data[i].Raw[index].split('\t')[1])

            # Read CV Current
            index = self._search_in_file_lines(self.Data[i].Raw, 'cv_coil_current_avg')
            if index is not None:
                self.Data[i].cv_current = float(self.Data[i].Raw[index].split('\t')[1])

            index = self._search_in_file_lines(self.Data[i].Raw, 'cv_coil_current_std')
            if index is not None:
                self.Data[i].cv_current_std = float(self.Data[i].Raw[index].split('\t')[1])

            # Read QS Current
            index = self._search_in_file_lines(self.Data[i].Raw, 'qs_coil_current_avg')
            if index is not None:
                self.Data[i].qs_current = float(self.Data[i].Raw[index].split('\t')[1])

            index = self._search_in_file_lines(self.Data[i].Raw, 'qs_coil_current_std')
            if index is not None:
                self.Data[i].qs_current_std = float(self.Data[i].Raw[index].split('\t')[1])
            
            #print(self.Data[i].main_current)
        
        self._get_multipoles_from_file_data()
                
    def _get_multipoles_from_file_data(self):
        n = len(self.files)
        i=0
        for i in range (n):
            index = self._search_in_file_lines(
                self.Data[i].Raw, 'Reading Data', 'Dados de Leitura')
            if index is not None:
                index_multipoles = index + 3
                multipoles_str = self.Data[i].Raw[index_multipoles:index_multipoles+15]
                multipoles = np.array([])
                for value in multipoles_str:
                    multipoles = np.append(multipoles, value.split('\t'))
                self.Data[i].multipoles = multipoles.reshape(15, 13).astype(np.float64)
                self.Data[i].magnet_type = np.nonzero(self.Data[i].multipoles[:, 7])[0][0]
                self.Data[i].columns_names = np.array(self.Data[i].Raw[index + 2].split('\t'))
                self.Data[i].reference_radius = float(
                    self.Data[i].Raw[index + 2].split("@")[1].split("mm")[0])/1000
            else:
                message = (
                    'Failed to read multipoles from file: \n\n"%s"' %
                    self.Data[i].file)
                raise DataFileError(message)
            
            ### Getting Raw data ###
            index = self._search_in_file_lines(self.Data[i].Raw, 'Raw Data Stored', 'Dados Brutos')
            if index is not None:
                curves_str = self.Data[i].Raw[index+3:]
                curves = np.array([])
                for value in curves_str:
                    curves = np.append(curves, value[:-1].split('\t'))
                self.Data[i].curves = curves.reshape(
                    int(len(curves_str)),
                    int(len(curves)/len(curves_str))).astype(np.float64)*1e-12
            else:
                message = (
                    'Failed to read raw data from file: \n\n"%s"' % self.Data[i].file)
                raise DataFileError(message)
        
        self._create_data_frames()
        
    def _create_data_frames(self):
        n = len(self.files)
        i=0
        for i in range (n):
            if (self.Data[i].multipoles is None or self.Data[i].curves is None or
               self.Data[i].columns_names is None):
                return

            index = np.char.mod('%d', np.linspace(1, 15, 15))
            self.Data[i].multipoles_df = pd.DataFrame(
                self.Data[i].multipoles, columns=self.Data[i].columns_names, index=index)

            _npoints = self.Data[i].curves.shape[0]
            _ncurves = self.Data[i].curves.shape[1]
            index = np.char.mod('%d', np.linspace(1, _npoints, _npoints))
            columns = np.char.mod('%d', np.linspace(1, _ncurves, _ncurves))
            self.Data[i].curves_df = pd.DataFrame(
                self.Data[i].curves, index=index, columns=columns)
    
    def _calc_offsets(self):
        n = len(self.files)
        i=0
        for i in range (n):
            if self.Data[i].multipoles is None or self.Data[i].magnet_type is None:
                return
            if self.Data[i].magnet_type != 0:
                n = self.Data[i].magnet_type
                normal = self.Data[i].multipoles[:, 1]
                normal_err = self.Data[i].multipoles[:, 2]
                skew = self.Data[i].multipoles[:, 3]
                skew_err = self.Data[i].multipoles[:, 4]

                self.Data[i].offset_x = normal[n-1]/(n*normal[n])
                self.Data[i].offset_x_err = (
                    ((normal_err[n-1]/(n*normal[n]))**2 -
                     (normal[n-1]*normal_err[n]/(n*(normal[n]**2)))**2)**(1/2))

                self.Data[i].offset_y = skew[n-1]/(n*normal[n])
                self.Data[i].offset_y_err = (
                    ((skew_err[n-1]/(n*normal[n]))**2 -
                     (skew[n-1]*normal_err[n]/(n*(normal[n]**2)))**2)**(1/2))
            else:
                self.Data[i].offset_x = 0
                self.Data[i].offset_x_err = 0
                self.Data[i].offset_y = 0
                self.Data[i].offset_y_err = 0
    
    def _set_roll(self):
        n = len(self.files)
        i=0
        for i in range (n):
            if self.Data[i].multipoles is None or self.Data[i].magnet_type is None:
                return
            self.Data[i].roll = self.Data[i].multipoles[self.Data[i].magnet_type, 7]
            self.Data[i].roll_err = self.Data[i].multipoles[self.Data[i].magnet_type, 8]
            
    def calc_residual_field(self, pos):
        """Calculate residual field.

        Args:
            pos (array): transversal position values [m].

        Returns:
            residual_normal (array): normal residual field [T].
            residual_skew (array): skew residual field [T].
        """
        n = len(self.files)
        i=0
        for i in range (n):
            if self.Data[i].multipoles is None or self.Data[i].magnet_type is None:
                return None, None

            n = self.Data[i].magnet_type
            nr_harmonics = self.Data[i].multipoles.shape[0]

            nrpts = len(pos)
            residual_normal = np.zeros(nrpts)
            residual_skew = np.zeros(nrpts)

            normal = self.Data[i].multipoles[:, 1]
            skew = self.Data[i].multipoles[:, 3]

            for i in range(nrpts):
                for m in range(n+1, nr_harmonics):
                    residual_normal[i] += (normal[m]/normal[n])*(pos[i]**(m - n))
                    residual_skew[i] += (skew[m]/normal[n])*(pos[i]**(m - n))

            return residual_normal, residual_skew
        
    def calc_residual_multipoles(self, pos):
        """Calculate residual field multipoles.

        Args:
            pos (array): transversal position values [m].

        Returns:
            residual_mult_normal (array): normal residual multipoles table.
            residual_mult_skew (array): skew residual multipoles table.
        """
        n = len(self.files)
        i=0
        for i in range (n):
            if self.Data[i].multipoles is None or self.Data[i].magnet_type is None:
                return None, None

            n = self.Data[i].magnet_type
            nr_harmonics = self.Data[i].multipoles.shape[0]

            nrpts = len(pos)
            residual_mult_normal = np.zeros([nr_harmonics, nrpts])
            residual_mult_skew = np.zeros([nr_harmonics, nrpts])

            normal = self.Data[i].multipoles[:, 1]
            skew = self.Data[i].multipoles[:, 3]

            for i in range(nrpts):
                for m in range(n+1, nr_harmonics):
                    residual_mult_normal[m, i] = (
                        normal[m]/normal[n])*(pos[i]**(m - n))
                    residual_mult_skew[m, i] = (
                        skew[m]/normal[n])*(pos[i]**(m - n))

            return residual_mult_normal, residual_mult_skew