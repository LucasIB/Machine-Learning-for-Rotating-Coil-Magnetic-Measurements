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
import tabledialog as _tabledialog
from sklearn.cluster import KMeans
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure as _Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as _FigureCanvas,
    NavigationToolbar2QT as _Toolbar)
from PyQt5.QtWidgets import QMessageBox as QMessageBox

#Interface
from main_screen import *
from numpy.f2py.auxfuncs import isarray, isinteger

class PlotDialog(QtWidgets.QDialog):
    """Matplotlib plot dialog."""

    def __init__(self, parent=None):
        """Add figure canvas to layout."""
        super().__init__(parent)

        self.figure = _Figure()
        self.canvas = _FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.canvas)
        self.toolbar = _Toolbar(self.canvas, self)
        _layout.addWidget(self.toolbar)
        self.setLayout(_layout)

    def updatePlot(self):
        """Update plot."""
        self.canvas.draw()

    def show(self):
        """Show dialog."""
        self.updatePlot()
        super().show()

class ApplicationWindow(QtWidgets.QWidget):
    """Machine Learning for Magnetic Measurement user interface"""
    def __init__(self, parent=None):
        super(ApplicationWindow, self).__init__(parent)

        self.ui = Ui_Machine_Learn_Interface()
        self.ui.setupUi(self)
        self.plot_dialog = PlotDialog()
        self.flag = False
        self.lastClicked = []
        self.dictionary()
        self.signals()
        
    def signals(self):
        """Connects UI signals and functions."""
        self.ui.pb_openfiles.clicked.connect(self.master_data)
        self.ui.pb_kmeans.clicked.connect(self.k_means)
        self.ui.cb_x_values.currentIndexChanged.connect(self.change_x_graphs_values)
        self.ui.cb_x_n_order.currentIndexChanged.connect(self.change_x_graphs_values)
        self.ui.cb_y_values.currentIndexChanged.connect(self.change_y_graphs_values)
        self.ui.cb_y_n_order.currentIndexChanged.connect(self.change_y_graphs_values)
        self.ui.pb_viewtable.clicked.connect(self.screen_table)
        self.ui.pb_cluster_hint.clicked.connect(self.cluster_hint)
        
    def master_data(self):
        """Data manipulate from database"""
        try:
            self.data_in = ld.Main_Script()
            _archives = self.data_in.load_files()
            if _archives:
                self.data_in.DataFile()
                if self.data_in is not None:
                    if (self.ui.cb_x_values.count() == 0) and (self.ui.cb_y_values.count() == 0): 
                        self.combo_box_x_value()
                        self.combo_box_y_value()
                    self.get_offsets()
                    self.get_roll()
                    self.ui.pb_kmeans.setEnabled(True)
                    self.magnet_name()
                    QtWidgets.QMessageBox.information(self,
                                                      'Info','Files successfully updated',QtWidgets.QMessageBox.Ok)
                else:
                    return
                QtWidgets.QApplication.processEvents()
            else:
                raise
        except:
            QtWidgets.QMessageBox.critical(self,
                                           'Critical','Files do not loaded.',QtWidgets.QMessageBox.Ok)
            return
        
    def get_offsets(self):
        """ Pick up the offsets values from data"""
        self.data_in._calc_offsets()
        
    def get_roll(self):
        """Pick up the roll angle"""
        self.data_in._set_roll()        

    def combo_box_x_value(self):
        """Fill the X values combo box after load data"""
        self.ui.cb_x_values.addItems(
            [s.replace("(T/m^n-2)", "").strip() for s in self.data_in.Data[0].columns_names])
        self.ui.cb_x_values.addItems(["main currents", "X offset", "roll angle"])
    
    def combo_box_y_value(self):
        """Fill the Y values combo box after load data"""
        self.ui.cb_y_values.addItems(
            [s.replace("(T/m^n-2)", "").strip() for s in self.data_in.Data[0].columns_names])
        self.ui.cb_y_values.addItems(["main currents", "Y offset", "roll angle"])
            
    def dictionary(self):
        self._dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                      '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                      '11': 11, '12': 12}
        
    def change_x_n_order(self, i, idx_label):
        _x_order = self.ui.cb_x_n_order.currentIndex()
        self.var_x = np.append(self.var_x,
                               self.data_in.Data[i].multipoles[_x_order][idx_label])
#         print('harmonic X: ', _x_order)
       
    def change_y_n_order(self, i, idx_label):        
        _y_order = self.ui.cb_y_n_order.currentIndex()
        self.var_y = np.append(self.var_y,
                               self.data_in.Data[i].multipoles[_y_order][idx_label])
#         print('harmonic Y: ', _y_order)
    
    def change_x_graphs_values(self):
        try:
            self.var_x = np.array([])
            idx_label = self.ui.cb_x_values.currentIndex()
            for i in range(len(self.data_in.files)):
                if idx_label == 13:
                    self.ui.cb_x_n_order.setEnabled(False)
                    self.var_x = np.append(self.var_x, self.data_in.Data[i].main_current)
                elif idx_label == 14:
                    self.ui.cb_x_n_order.setEnabled(False)
                    self.var_x = np.append(self.var_x, self.data_in.Data[i].offset_x)
                elif idx_label == 15:
                    self.ui.cb_x_n_order.setEnabled(False)
                    self.var_x = np.append(self.var_x, self.data_in.Data[i].roll)
                elif idx_label == 0:
                    self.ui.cb_x_n_order.setEnabled(False)
                    pass
                elif str(idx_label) in self._dict:
                    self.ui.cb_x_n_order.setEnabled(True)
                    self.change_x_n_order(i, idx_label)
#             print(self.var_x)
                
        except:
            traceback.print_exc(file=sys.stdout)            
        
    def change_y_graphs_values(self):
        try:
            self.var_y = np.array([])
            idx_label = self.ui.cb_y_values.currentIndex()
            for i in range(len(self.data_in.files)):
                if idx_label == 0:
                    pass
                elif idx_label == 13:
                    self.ui.cb_y_n_order.setEnabled(False)
                    self.var_y = np.append(self.var_y, self.data_in.Data[i].main_current)
                elif idx_label == 14:
                    self.ui.cb_y_n_order.setEnabled(False)
                    self.var_y = np.append(self.var_y, self.data_in.Data[i].offset_y)
                elif idx_label == 15:
                    self.ui.cb_y_n_order.setEnabled(False)
                    self.var_y = np.append(self.var_y, self.data_in.Data[i].roll)
                elif str(idx_label) in self._dict:
                    self.ui.cb_y_n_order.setEnabled(True)
                    self.change_y_n_order(i, idx_label)                   
#             print(self.var_y)
        except:
            traceback.print_exc(file=sys.stdout)
            
    def data_frame_manager(self):      
        if (len(self.var_x)) and (len(self.var_y)) > 0:
            self.DF = {'x': self.var_x,
                       'y': self.var_y}
            self.DF = pd.DataFrame(self.DF, columns=['x', 'y'])
    
    def k_means(self):
        """Method for clustering data with k-means"""
        try:
            self.data_frame_manager()
                        
            _n_cluster = self.ui.sb_cluster_number.value()
            
            if len(self.DF.columns) > 2:
                self.DF.drop(['magnet', 'class'], axis=1, inplace=True)       
            
            #Applying kmeans functions
            _kmeans = KMeans(n_clusters=_n_cluster).fit(self.DF)
                   
            #Predicts
            _predicts = _kmeans.labels_
            
            #Centroids
            _centroids = _kmeans.cluster_centers_
            
            if len(self.DF.columns) == 2:
                _names = np.array([])
                _prev = np.array([])
                for j in range(len(self.data_in.files)):
                    _names = np.append(_names, self.data_in.Data[j].magnet_name)
                    _prev = np.append(_prev, _predicts[j])
                self.DF['magnet'] = pd.Series(_names, index=self.DF.index)
                self.DF['class'] = pd.Series(_prev, index=self.DF.index)
#                 print(self.DF.head())
            
            self.ui.pb_viewtable.setEnabled(True)
            self.plot_view(self.DF, _centroids)
        except:
            traceback.print_exc(file=sys.stdout)
        
    def filter_class(self, df):#, n):
        '''**Implementar para qualquer númaero de cluster. Dica: Tentar usar dicionários**
                for i in range (n):    #n = number of clusters
                _class[i]...
            '''
        try:
            _class_0, _class_0_x, _class_0_y = 0,0,0
            _class_1, _class_1_x, _class_1_y = 0,0,0
            _class_2, _class_2_x, _class_2_y = 0,0,0
    
            #Filter spots class == 0
            _class_0 = df.loc[(df['class'] == 0)]
            _class_0_x = _class_0.x.values
            _class_0_y = _class_0.y.values
            
            #Filter spots class == 1
            _class_1 = df.loc[(df['class'] == 1)]
            _class_1_x = _class_1.x.values
            _class_1_y = _class_1.y.values
            
            #Filter spots class == 2
            _class_2 = df.loc[(df['class'] == 2)]
            _class_2_x = _class_2.x.values
            _class_2_y = _class_2.y.values
            
            return _class_0_x, _class_0_y, _class_1_x, _class_1_y, _class_2_x, _class_2_y
        except:
            traceback.print_exc(file=sys.stdout)
            
    def plot_view(self, df, center):
        """Plotting graphs in Graphics View QWidget"""
        try:           
            #Clear screen
            self.ui.graphicsView.clear()
            
            #Closing previous legend
            if self.flag:
                self.ui.graphicsView.plotItem.legend.close()
            
            #Creating the new legend
            self.ui.graphicsView.plotItem.addLegend()
            
            _vars = self.filter_class(df)
            
            _s1 = pg.ScatterPlotItem(_vars[0],
                                     _vars[1],
                                     size=10,
                                     pen=pg.mkPen({'color': "F4425F", 'width': 1}), #Red
                                     brush=pg.mkBrush(244, 115, 136, 120),
                                     name='class 0') # Class = 0
            
            _s2 = pg.ScatterPlotItem(_vars[2],
                                     _vars[3],
                                     size=10,
                                     pen=pg.mkPen({'color': "1003BC", 'width': 1}), #Blue
                                     brush=pg.mkBrush(49, 104, 224, 120),
                                     name='class 1') # Class = 1
            
            _s3 = pg.ScatterPlotItem(_vars[4],
                                     _vars[5],
                                     size=10,
                                     pen=pg.mkPen({'color': "DD9D1C", 'width': 1}), #Yellow
                                     brush=pg.mkBrush(237, 192, 101, 120),
                                     name='class 2') # Class = 2
            #Adding centers points
            _s4 = pg.ScatterPlotItem(center[:, 0],
                                     center[:, 1],
                                     symbol='d',
                                     size=12,
                                     pen=pg.mkPen(None),
                                     brush='g') # Centers

            self.ui.graphicsView.plotItem.setLabel('left', self.ui.cb_y_values.currentText())
            self.ui.graphicsView.plotItem.setLabel('bottom', self.ui.cb_x_values.currentText())
            self.ui.graphicsView.plotItem.showGrid(x=True, y=True, alpha=0.2)
            self.ui.graphicsView.addItem(_s1)
            self.ui.graphicsView.addItem(_s2)
            self.ui.graphicsView.addItem(_s3)
            self.ui.graphicsView.addItem(_s4)
            self.ui.graphicsView.plotItem.legend.addItem(_s1, ' class 0')
            self.ui.graphicsView.plotItem.legend.addItem(_s2, ' class 1')
            self.ui.graphicsView.plotItem.legend.addItem(_s3, ' class 2')
            self.ui.graphicsView.plotItem.legend.addItem(_s4, ' centroids')
            _s1.sigClicked.connect(self.clicked)
            _s2.sigClicked.connect(self.clicked)
            _s3.sigClicked.connect(self.clicked)
            self.flag = True
            QtWidgets.QApplication.processEvents()
        except:
            traceback.print_exc(file=sys.stdout)
            
    def counter(self):
        """Count the number of elements in each cluster"""
        pass
    
    def magnet_name(self):
        _magnet = self.data_in.Data[0].magnet_name[:3]
        return self.ui.lb_magnt_name.setText(_magnet)
    
    def cluster_hint(self):
        """Ideal determination of number of clusters (elbow method)"""
        try:                
            wcss = [] 
            for i in range(1, 11):
                kmeans = KMeans(n_clusters = i, init = 'random')
                if len(self.DF.columns) > 2:
                    _DF_for_hint = self.DF.drop(['magnet', 'class'], axis=1)
                    kmeans.fit(_DF_for_hint)
                else:
                    kmeans.fit(self.DF)
                wcss.append(kmeans.inertia_)  
            fig = self.plot_dialog.figure
            ax = self.plot_dialog.ax
            ax.clear()
            ax.plot(np.arange(1, 11), wcss)
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Within cluster sum of squares (WSS)')
            ax.set_title('Elbow Method')
            ax.grid('on', alpha=0.3)
            fig.tight_layout()
            self.plot_dialog.show()
        except:
            QtWidgets.QMessageBox.information(self,'Info',
                                              'Please, select X or Y values and click in K-Means button.',QtWidgets.QMessageBox.Ok)
            return
               
    def screen_table(self):
        """Create new screen with table."""
        try:
            dialog_table = _tabledialog.TableDialog(table_df=self.DF)
            dialog_table.exec_()

        except Exception:
            QtWidgets.QMessageBox.critical(
                self, 'Failure', 'Failed to open table.', _QMessageBox.Ok)
            
    def clicked(self, plot, points):
        """Make all plots clickable"""
        try:
            #global lastClicked
            for p in self.lastClicked:
                p.resetPen()
            print("clicked points: ", points[0].pos())
            for p in points:
                p.setPen('b', width=2)
            self.lastClicked = points
        except:
            traceback.print_exc(file=sys.stdout)
            return            
                   
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
