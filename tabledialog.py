"""Table dialog widget."""

import os.path as _path
import PyQt5.uic as _uic
from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QDesktopWidget as _QDesktopWidget,
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem)


_basepath = _path.dirname(_path.abspath(__file__))


class TableDialog(_QDialog):
    """Table dialog."""

    def __init__(self, parent=None, table_df=None):
        """Initialize the dialog."""
        super(TableDialog, self).__init__(parent)

        self.table_df = table_df

        # setup the ui
        uifile = _path.join(_basepath, _path.join('ui', 'tabledialog.ui'))
        self.ui = _uic.loadUi(uifile, self)
        self.setAttribute(_Qt.WA_DeleteOnClose)

        self.move(
            _QDesktopWidget().availableGeometry().center().x() -
            self.geometry().width()/2,
            _QDesktopWidget().availableGeometry().center().y() -
            self.geometry().height()/2)

        self.ui.bt_copy_to_clipboard.clicked.connect(
            self.copy_to_clipboard)
        self.create_table()

    def create_table(self):
        """Create table."""
        if self.table_df is None:
            return

        df = self.table_df

        _n_columns = len(df.columns)
        _n_rows = len(df.index)

        if _n_columns != 0:
            self.ui.tb_general.setColumnCount(_n_columns)
            self.ui.tb_general.setHorizontalHeaderLabels(
                df.columns)

        if _n_rows != 0:
            self.ui.tb_general.setRowCount(_n_rows)
            #self.ui.tb_general.setVerticalHeaderLabels(df.index)

        for idx in range(_n_rows):
            for _jdx in range(_n_columns):
                if _jdx == 0:
                    self.ui.tb_general.setItem(
                     idx, _jdx,
                     _QTableWidgetItem(
                        '{0:1g}'.format(df.iloc[idx, _jdx])))
                else:
                    self.ui.tb_general.setItem(
                        idx, _jdx,
                        _QTableWidgetItem(
                            (df.iloc[idx, _jdx])))

        _QApplication.processEvents()

        self.ui.tb_general.resizeColumnsToContents()

    def copy_to_clipboard(self):
        """Copy table to clipboard."""
        if self.table_df is not None:
            self.table_df.to_clipboard(excel=True)
