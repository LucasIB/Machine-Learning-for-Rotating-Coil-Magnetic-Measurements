# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_screen.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Machine_Learn_Interface(object):
    def setupUi(self, Machine_Learn_Interface):
        Machine_Learn_Interface.setObjectName("Machine_Learn_Interface")
        Machine_Learn_Interface.resize(753, 664)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Machine_Learn_Interface.sizePolicy().hasHeightForWidth())
        Machine_Learn_Interface.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/resource/cropped-Brain-Tool-Icon-2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Machine_Learn_Interface.setWindowIcon(icon)
        self.gridLayout_2 = QtWidgets.QGridLayout(Machine_Learn_Interface)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(Machine_Learn_Interface)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setObjectName("tabWidget")
        self.Tab = QtWidgets.QWidget()
        self.Tab.setObjectName("Tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.Tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.Tab)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.pb_openfiles = QtWidgets.QPushButton(self.Tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pb_openfiles.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/resource/file_add.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_openfiles.setIcon(icon1)
        self.pb_openfiles.setObjectName("pb_openfiles")
        self.horizontalLayout_2.addWidget(self.pb_openfiles)
        self.pb_kmeans = QtWidgets.QPushButton(self.Tab)
        self.pb_kmeans.setEnabled(False)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pb_kmeans.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/resource/cluster.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_kmeans.setIcon(icon2)
        self.pb_kmeans.setObjectName("pb_kmeans")
        self.horizontalLayout_2.addWidget(self.pb_kmeans)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.Tab)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.Tab)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.pb_cluster_hint = QtWidgets.QPushButton(self.Tab)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/resource/Magic-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_cluster_hint.setIcon(icon3)
        self.pb_cluster_hint.setIconSize(QtCore.QSize(18, 18))
        self.pb_cluster_hint.setObjectName("pb_cluster_hint")
        self.horizontalLayout.addWidget(self.pb_cluster_hint)
        self.sb_cluster_number = QtWidgets.QSpinBox(self.Tab)
        self.sb_cluster_number.setProperty("value", 3)
        self.sb_cluster_number.setObjectName("sb_cluster_number")
        self.horizontalLayout.addWidget(self.sb_cluster_number)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.Tab)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.Tab)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.Tab)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.Tab)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 1, 1, 1)
        self.lb_magnt_name = QtWidgets.QLabel(self.Tab)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lb_magnt_name.setFont(font)
        self.lb_magnt_name.setText("")
        self.lb_magnt_name.setObjectName("lb_magnt_name")
        self.gridLayout.addWidget(self.lb_magnt_name, 3, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout.addItem(spacerItem3, 3, 4, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 3, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.Tab)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 1, 1, 1)
        self.cb_y_values = QtWidgets.QComboBox(self.Tab)
        self.cb_y_values.setObjectName("cb_y_values")
        self.gridLayout.addWidget(self.cb_y_values, 0, 2, 1, 1)
        self.cb_x_values = QtWidgets.QComboBox(self.Tab)
        self.cb_x_values.setObjectName("cb_x_values")
        self.gridLayout.addWidget(self.cb_x_values, 1, 2, 1, 1)
        self.cb_y_n_order = QtWidgets.QComboBox(self.Tab)
        self.cb_y_n_order.setEnabled(False)
        self.cb_y_n_order.setObjectName("cb_y_n_order")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.cb_y_n_order.addItem("")
        self.gridLayout.addWidget(self.cb_y_n_order, 0, 4, 1, 1)
        self.cb_x_n_order = QtWidgets.QComboBox(self.Tab)
        self.cb_x_n_order.setEnabled(False)
        self.cb_x_n_order.setObjectName("cb_x_n_order")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.cb_x_n_order.addItem("")
        self.gridLayout.addWidget(self.cb_x_n_order, 1, 4, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.pb_viewtable = QtWidgets.QPushButton(self.Tab)
        self.pb_viewtable.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pb_viewtable.sizePolicy().hasHeightForWidth())
        self.pb_viewtable.setSizePolicy(sizePolicy)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/resource/table.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_viewtable.setIcon(icon4)
        self.pb_viewtable.setObjectName("pb_viewtable")
        self.verticalLayout_2.addWidget(self.pb_viewtable)
        self.graphicsView = PlotWidget(self.Tab)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.NoBrush)
        self.graphicsView.setBackgroundBrush(brush)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_2.addWidget(self.graphicsView)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.tabWidget.addTab(self.Tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_2.addWidget(self.tabWidget, 1, 0, 1, 1)

        self.retranslateUi(Machine_Learn_Interface)
        self.tabWidget.setCurrentIndex(0)
        self.cb_y_n_order.setCurrentIndex(1)
        self.cb_x_n_order.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Machine_Learn_Interface)

    def retranslateUi(self, Machine_Learn_Interface):
        _translate = QtCore.QCoreApplication.translate
        Machine_Learn_Interface.setWindowTitle(_translate("Machine_Learn_Interface", "Machine Learn Interface"))
        self.label.setText(_translate("Machine_Learn_Interface", "Machine Learning for Magnetic Measurements"))
        self.pb_openfiles.setText(_translate("Machine_Learn_Interface", "Open Files"))
        self.pb_kmeans.setText(_translate("Machine_Learn_Interface", " Process K-Means"))
        self.label_2.setText(_translate("Machine_Learn_Interface", "Type Number of Cluster:"))
        self.pb_cluster_hint.setText(_translate("Machine_Learn_Interface", "Cluster Hint"))
        self.label_5.setText(_translate("Machine_Learn_Interface", "n order:"))
        self.label_6.setText(_translate("Machine_Learn_Interface", "n order:"))
        self.label_3.setText(_translate("Machine_Learn_Interface", "Y Values:"))
        self.label_4.setText(_translate("Machine_Learn_Interface", "X Values:"))
        self.label_7.setText(_translate("Machine_Learn_Interface", "Magnet:"))
        self.cb_y_n_order.setItemText(0, _translate("Machine_Learn_Interface", "1"))
        self.cb_y_n_order.setItemText(1, _translate("Machine_Learn_Interface", "2"))
        self.cb_y_n_order.setItemText(2, _translate("Machine_Learn_Interface", "3"))
        self.cb_y_n_order.setItemText(3, _translate("Machine_Learn_Interface", "4"))
        self.cb_y_n_order.setItemText(4, _translate("Machine_Learn_Interface", "5"))
        self.cb_y_n_order.setItemText(5, _translate("Machine_Learn_Interface", "6"))
        self.cb_y_n_order.setItemText(6, _translate("Machine_Learn_Interface", "7"))
        self.cb_y_n_order.setItemText(7, _translate("Machine_Learn_Interface", "8"))
        self.cb_y_n_order.setItemText(8, _translate("Machine_Learn_Interface", "9"))
        self.cb_y_n_order.setItemText(9, _translate("Machine_Learn_Interface", "10"))
        self.cb_y_n_order.setItemText(10, _translate("Machine_Learn_Interface", "11"))
        self.cb_y_n_order.setItemText(11, _translate("Machine_Learn_Interface", "12"))
        self.cb_y_n_order.setItemText(12, _translate("Machine_Learn_Interface", "13"))
        self.cb_y_n_order.setItemText(13, _translate("Machine_Learn_Interface", "14"))
        self.cb_y_n_order.setItemText(14, _translate("Machine_Learn_Interface", "15"))
        self.cb_x_n_order.setItemText(0, _translate("Machine_Learn_Interface", "1"))
        self.cb_x_n_order.setItemText(1, _translate("Machine_Learn_Interface", "2"))
        self.cb_x_n_order.setItemText(2, _translate("Machine_Learn_Interface", "3"))
        self.cb_x_n_order.setItemText(3, _translate("Machine_Learn_Interface", "4"))
        self.cb_x_n_order.setItemText(4, _translate("Machine_Learn_Interface", "5"))
        self.cb_x_n_order.setItemText(5, _translate("Machine_Learn_Interface", "6"))
        self.cb_x_n_order.setItemText(6, _translate("Machine_Learn_Interface", "7"))
        self.cb_x_n_order.setItemText(7, _translate("Machine_Learn_Interface", "8"))
        self.cb_x_n_order.setItemText(8, _translate("Machine_Learn_Interface", "9"))
        self.cb_x_n_order.setItemText(9, _translate("Machine_Learn_Interface", "10"))
        self.cb_x_n_order.setItemText(10, _translate("Machine_Learn_Interface", "12"))
        self.cb_x_n_order.setItemText(11, _translate("Machine_Learn_Interface", "13"))
        self.cb_x_n_order.setItemText(12, _translate("Machine_Learn_Interface", "14"))
        self.cb_x_n_order.setItemText(13, _translate("Machine_Learn_Interface", "15"))
        self.pb_viewtable.setText(_translate("Machine_Learn_Interface", " View Table"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab), _translate("Machine_Learn_Interface", "Clustering"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Machine_Learn_Interface", "Predicting"))

from pyqtgraph import PlotWidget
import resource_ml
