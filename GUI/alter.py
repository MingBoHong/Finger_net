
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys
sys.path.append("..")
from identify import identify
from scipy import misc
import os
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.image_dir = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(612, 519)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(80, 100, 441, 311))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Pic = QtWidgets.QLabel(self.widget)
        self.Pic.setObjectName("Pic")
        self.verticalLayout_2.addWidget(self.Pic)
        self.result = QtWidgets.QTextBrowser(self.widget)
        self.result.setObjectName("result")
        self.verticalLayout_2.addWidget(self.result)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.openButton = QtWidgets.QPushButton(self.widget)
        self.openButton.setObjectName("openButton")
        self.verticalLayout.addWidget(self.openButton)
        self.startButton = QtWidgets.QPushButton(self.widget)
        self.startButton.setObjectName("startButton")
        self.verticalLayout.addWidget(self.startButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.closeSystem = QtWidgets.QPushButton(self.widget)
        self.closeSystem.setObjectName("closeSystem")
        self.verticalLayout.addWidget(self.closeSystem)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 612, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.closeSystem.clicked.connect(self.close_system)
        self.openButton.clicked.connect(self.open_pic)
        self.startButton.clicked.connect(self.predict)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Pic.setText(_translate("MainWindow", "TextLabel"))
        self.openButton.setText(_translate("MainWindow", "打开图片"))
        self.startButton.setText(_translate("MainWindow", "开始判断"))
        self.closeSystem.setText(_translate("MainWindow", "关闭系统"))
    def close_system(self):
        sys.exit()

    def open_pic(self):

        openfile_name, filetype = QFileDialog.getOpenFileNames(self, '选择文件', r"F:\KDR\BRL\project\LivDet2009\Training\Biometrika\Alive", 'Image Files(*.jpg *.png *.tif)')
        self.image_dir = openfile_name[0]
        pix = QPixmap(self.image_dir)
        self.Pic.setPixmap(pix)
        self.Pic.setScaledContents(True)

    def predict(self):
        if os.path.exists(self.image_dir):
            pic = misc.imread(self.image_dir)
            result = identify(pic)
            self.result.setText(result)
        else:
            reply = QMessageBox.information(self,
                                            "Error",
                                            "FileNotFoundError",
                                            QMessageBox.Yes | QMessageBox.No)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("Finger")
    MainWindow.setFixedSize(640,480)
    MainWindow.show()
    sys.exit(app.exec_())




