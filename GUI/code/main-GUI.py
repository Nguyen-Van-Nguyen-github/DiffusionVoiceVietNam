from PyQt5 import QtCore, QtGui, QtWidgets

from GUI import Ui_MainWindow
from func import VideoPlayerController

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.controller = VideoPlayerController(self)
        self.controller = AudioConverter(self)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

