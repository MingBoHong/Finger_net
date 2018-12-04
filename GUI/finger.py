

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
import sys
sys.path.append("..")
from identify import identify
from scipy import misc
import os


class Example(QThread):
    signal = pyqtSignal(str)    # 括号里填写信号传递的参数
    def __init__(self,image_dir):
        super().__init__()
        self.image_dir = image_dir


    def __del__(self):
        self.wait()

    def run(self):
        global result_name
        # 进行任务操作
        if os.path.exists(self.image_dir):
            pic = misc.imread(self.image_dir)
            result = identify(pic)
            self.signal.emit(result)
        else:
            reply = QMessageBox.information(self,
                                            "Error",
                                            "FileNotFoundError",
                                            QMessageBox.Yes | QMessageBox.No)






class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.image_dir = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(21, 102, 531, 291))
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.Pic = QtWidgets.QLabel(self.widget)
        self.Pic.setObjectName("Pic")
        self.verticalLayout.addWidget(self.Pic)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label1 = QtWidgets.QLabel(self.widget)
        self.label1.setObjectName("label1")
        self.horizontalLayout.addWidget(self.label1)
        self.result = QtWidgets.QTextBrowser(self.widget)
        self.result.setObjectName("result")
        self.horizontalLayout.addWidget(self.result)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label2 = QtWidgets.QLabel(self.widget)
        self.label2.setObjectName("label2")
        self.horizontalLayout_2.addWidget(self.label2)
        self.progressBar = QtWidgets.QProgressBar(self.widget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.openButton = QtWidgets.QPushButton(self.widget)
        self.openButton.setObjectName("openButton")
        self.verticalLayout_2.addWidget(self.openButton)
        self.startButton = QtWidgets.QPushButton(self.widget)
        self.startButton.setObjectName("startButton")
        self.verticalLayout_2.addWidget(self.startButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.closeSystem = QtWidgets.QPushButton(self.widget)
        self.closeSystem.setObjectName("closeSystem")
        self.verticalLayout_2.addWidget(self.closeSystem)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 612, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.openButton.clicked.connect(self.open_pic)
        self.startButton.clicked.connect(self.predict)
        self.closeSystem.clicked.connect(self.close_system)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Pic.setText(_translate("MainWindow", "TextLabel"))
        self.label1.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600; color:#ff0000;\">判断结果:</span></p></body></html>"))
        self.label2.setText(_translate("MainWindow", "进度:"))
        self.openButton.setText(_translate("MainWindow", "打开图片"))
        self.startButton.setText(_translate("MainWindow", "开始判断"))
        self.closeSystem.setText(_translate("MainWindow", "关闭系统"))
    def close_system(self):
        sys.exit()

    def open_pic(self):

        openfile_name, filetype = QFileDialog.getOpenFileNames(self, '选择文件',
                                                               r"F:\KDR\BRL\project\LivDet2009\Training\Biometrika\Alive",
                                                               'Image Files(*.jpg *.png *.tif)')
        self.image_dir = openfile_name[0]
        pix = QPixmap(self.image_dir)
        self.Pic.setPixmap(pix)
        self.Pic.setScaledContents(True)
        
    def setText(self,result_name):
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 100)
        self.result.setText(result_name)
        
    def predict(self):
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.thread = Example(self.image_dir)
        self.thread.start()  # 启动线程
        self.thread.signal.connect(self.setText)


   



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("Finger")
    MainWindow.show()
    sys.exit(app.exec_())
