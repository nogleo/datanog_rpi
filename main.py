# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/main.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import datanog as nog
import time
from collections import deque
dn = nog.DATANOG()
fs = 3333
dt = 1/fs

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(410, 110, 181, 121))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(170, 110, 181, 121))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionIMU = QtWidgets.QAction(MainWindow)
        self.actionIMU.setObjectName("actionIMU")
        self.actionRotor = QtWidgets.QAction(MainWindow)
        self.actionRotor.setObjectName("actionRotor")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_4.setText(_translate("MainWindow", "Stop"))
        self.pushButton_5.setText(_translate("MainWindow", "Start"))
        self.actionIMU.setText(_translate("MainWindow", "IMU"))
        self.actionRotor.setText(_translate("MainWindow", "Rotor"))
        self.pushButton_5.clicked.connect(collect)
        self.pushButton_4.clicked.connect(stopcollect)


def stopcollect():
    global state
    state = False

def collect():
    global state
    state = True
    data0 = deque()
    t0=tf = time.perf_counter()
    while state:
        ti=time.perf_counter()
        if ti-tf>=dt:
            tf = ti
            data0.append(dn.pull())


    t1 = time.perf_counter()
    print(t1-t0)
    #dn.lograw(data0)
    #dn.log(data0)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
