from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

app = QApplication([])
window = QMainWindow()
window.setWindowTitle("Phát Video")
window.setGeometry(120, 120, 900, 700)
video_widget = QVideoWidget()
layout = QVBoxLayout()
layout.addWidget(video_widget)

select_button = QPushButton("Chọn Video")
layout.addWidget(select_button)

central_widget = QWidget()
central_widget.setLayout(layout)
window.setCentralWidget(central_widget)
media_player = QMediaPlayer()
media_player.setVideoOutput(video_widget)
def open_file():
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(window, "Chọn tệp video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
    if file_name:
        media_content = QMediaContent(QUrl.fromLocalFile(file_name))
        media_player.setMedia(media_content)
        media_player.play()

select_button.clicked.connect(open_file)
window.show()
app.exec_()




