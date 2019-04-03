#!/usr/bin/python
# -*- coding: utf-8 -*-
#Library
import sys
import threading
import traceback
import numpy as np
import pandas as pd
import pyqtgraph as pg
import load_data as ld
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox as QMessageBox
#Interface
from main_screen import *

class ApplicationWindow(QtWidgets.QWidget):
    """Machine Learning for Magnetic Measurement user interface"""
    def __init__(self, parent=None):
        super(ApplicationWindow, self).__init__(parent)
        
        self.ui = Ui_Machine_Learn_Interface()
        self.ui.setupUi(self)
        self.signals()
        
    def signals(self):
        """Connects UI signals and functions."""
        self.ui.pb_openfiles.clicked.connect(self.master_data) 
        self.ui.pb_kmeans.clicked.connect(self.k_means)
        self.ui.cb_x_values.currentIndexChanged.connect(self.change_x_graphs_label)
        self.ui.cb_y_values.currentIndexChanged.connect(self.change_y_graphs_label)
        
    def master_data(self):
        try:
            self.data_in = ld.Main_Script()         # self.data_in is all variables that will be used (ex.:a.myapp.data_in.Data[1])
            _archives = self.data_in.load_files()
            if _archives:
                self.data_in.DataFile()
                if len(self.data_in.files) != 0:
                    self.combo_box_x_value()
                    self.combo_box_y_value()
                    self.get_offsets()
                    self.get_roll()
                    self.ui.pb_kmeans.setEnabled(True)
                    QtWidgets.QMessageBox.information(self,'Info','Files successfully updated',QtWidgets.QMessageBox.Ok)
                else:
                    return
                QtWidgets.QApplication.processEvents()
            else:
                QtWidgets.QMessageBox.critical(self,'Info','Files do not loaded.',QtWidgets.QMessageBox.Ok)
                return
        except:
            traceback.print_exc(file=sys.stdout)
        
    def get_offsets(self):
        """ Pick up the offsets values from data"""
        self.data_in._calc_offsets()
        
    def get_roll(self):
        """Pick up the roll angle"""
        self.data_in._set_roll()        

    def combo_box_x_value(self):
        self.ui.cb_x_values.addItems(
            [s.replace("(T/m^n-2)", "").strip() for s in self.data_in.Data[0].columns_names])
        self.ui.cb_x_values.addItems(["main currents", "X offset", "roll angle"])
    
    def combo_box_y_value(self):
        self.ui.cb_y_values.addItems(
            [s.replace("(T/m^n-2)", "").strip() for s in self.data_in.Data[0].columns_names])
        self.ui.cb_y_values.addItems(["main currents", "Y offset", "roll angle"])
    
    def change_x_graphs_label(self):
        try:
            self.var_x = np.array([])
            idx_label = self.ui.cb_x_values.currentIndex()
            for i in range(len(self.data_in.files)):
                if idx_label == 14:
                    self.var_x = np.append(self.var_x, self.data_in.Data[i].offset_x)
                elif idx_label == 0:
                    pass
                else:
                    self.var_x = np.append(self.var_x, self.data_in.Data[i].multipoles[self.data_in.Data[i].magnet_type][idx_label])                
        except:
            traceback.print_exc(file=sys.stdout)            
        
    def change_y_graphs_label(self):
        try:
            self.var_y = np.array([])
            idx_label = self.ui.cb_y_values.currentIndex()
            for i in range(len(self.data_in.files)):
                if idx_label == 0:
                    pass
                elif idx_label == 14:
                    self.var_y = np.append(self.var_y, self.data_in.Data[i].offset_y)
                else:
                    self.var_y = np.append(self.var_y, self.data_in.Data[i].multipoles[self.data_in.Data[i].magnet_type][idx_label])             
            
            if (len(self.var_x)) and (len(self.var_y)) > 0:
                self.data_frame_manager(self.var_x, self.var_y)
        except:
            traceback.print_exc(file=sys.stdout)
            
    def data_frame_manager(self, var_x, var_y):
        self.DF = {'x': var_x,
                   'y': var_y}
        self.DF = pd.DataFrame(self.DF, columns=['x', 'y'])
        print(self.DF)
        print(self.ui.cb_x_values.currentIndex())
        print(self.ui.cb_y_values.currentIndex())       
    
    def k_means(self):
        """Method for clustering data with k-means"""
        _n_cluster = self.ui.sb_cluster_number.value()
        
        #Applying kmeans functions        
        _kmeans = KMeans(n_clusters=_n_cluster).fit(self.DF)
        
        #Centroids
        _centroids = _kmeans.cluster_centers_
        
        self.plot_view(self.DF, _centroids, _kmeans)
        
    def plot_view(self, df, center, kmeans=0.0):
        try:           
            #Clear screen
            self.ui.graphicsView.clear()

            _s1 = pg.ScatterPlotItem(size = 10)#, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
            
            spots_2 = [{'pos': df.values.T[:,i], 'data': 1} for i in range(len(df))]

            _s1.addPoints(spots_2)
            self.ui.graphicsView.plotItem.showGrid(x=True, y=True, alpha=0.2)
            self.ui.graphicsView.addItem(_s1)
        except:
            traceback.print_exc(file=sys.stdout)           
        
class main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        # Starts Graphic Interface
        self.App = QtWidgets.QApplication(sys.argv)
        self.myapp = ApplicationWindow()
        self.myapp.show()
        self.App.exec_()
        
a = main()
