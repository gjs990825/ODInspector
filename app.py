import io
import os
import sys
import logging

import cv2
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication, Qt, QDir, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage, QKeySequence
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QMenu, QHBoxLayout, QStyle, \
    QSlider, QFileDialog, QVBoxLayout, QLabel

logging.basicConfig(level=logging.INFO)


class ODInspector(QMainWindow):
    video_file = 'videos/elephants_dream.mp4'

    def __init__(self):
        super().__init__()

        self.current_frame = None
        self.window_size = QSize(700, 500)
        self.camera = None
        self.video_name = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.frame_out_sync = False

        self.skip_seconds = 15
        self.fps = 25
        self.timer_interval = int(1000 / self.fps)

        self.frame_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_position_slider.setRange(0, 1000)

        self.frame_position_slider.sliderReleased.connect(self.video_resume)
        self.frame_position_slider.sliderPressed.connect(self.video_pause)
        self.frame_position_slider.sliderMoved.connect(self.frame_position_slider_callback)

        # Create a QHBoxLayout instance
        central_widget = QWidget(self)

        inner_layout = QHBoxLayout()
        # Add widgets to the layout

        self.button_pause_resume = QPushButton("Pause/Resume")
        self.button_pause_resume.setShortcut(Qt.Key.Key_Space)

        self.button_pause_resume.clicked.connect(self.video_pause_resume)
        inner_layout.addWidget(self.button_pause_resume, 1)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.video_stop)
        inner_layout.addWidget(self.button_stop, 1)

        outer_layout = QVBoxLayout()
        self.frame_display = QLabel('Press Ctrl+O to open a video')
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(self.frame_display, 10)
        outer_layout.addLayout(inner_layout, 1)
        outer_layout.addWidget(self.frame_position_slider)

        # Set the layout on the application's window
        central_widget.setLayout(outer_layout)

        # Configure menu bar items
        file_menu = QMenu('&File', self)
        menu_bar = self.menuBar()
        menu_bar.addMenu(file_menu)

        open_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveDVDIcon)
        open_action = QAction(open_icon, '&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open video file')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        exit_action = QAction(exit_icon, '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QCoreApplication.exit)
        file_menu.addAction(exit_action)

        self.widgets_enabled(False)

        # Window management
        self.setCentralWidget(central_widget)
        self.setWindowTitle("OD Inspector")
        self.setWindowIcon(QIcon('images/icons/ic_app.png'))
        self.statusBar().showMessage('Ready')
        self.setGeometry(300, 300, 700, 500)
        self.center()
        self.show()

    def timerEvent(self, timer_event) -> None:
        # frame = self.get_a_frame(self.frame_counter)
        if not self.frame_out_sync:
            self.frame_position += 1
        else:
            logging.info(f'Seeking to {self.frame_position}th frame')
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)
            self.frame_out_sync = False

        success, img = self.camera.read()
        if success:
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_current_frame()

    def display_current_frame(self):
        q_image = QtGui.QImage(self.current_frame.data,
                               self.current_frame.shape[1],
                               self.current_frame.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        q_pixmap = QtGui.QPixmap.fromImage(q_image)
        q_pixmap_scaled = q_pixmap.scaled(self.frame_display.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.FastTransformation)
        self.frame_display.setPixmap(q_pixmap_scaled)
        self.frame_position_slider.setValue(self.frame_position)
        logging.debug(f'Showing {self.frame_position}th frame')

    def frame_position_slider_callback(self):
        position = self.frame_position_slider.value()
        self.update_frame_position(position)

    def update_frame_position(self, position):
        self.frame_position = position
        self.frame_out_sync = True

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        if self.video_timer_id is None:
            a0.accept()
            return

        if a0.key() == Qt.Key.Key_Right:
            self.update_frame_position(self.frame_position + self.skip_seconds * self.fps)
        elif a0.key() == Qt.Key.Key_Left:
            self.update_frame_position(self.frame_position - self.skip_seconds * self.fps)
        else:
            a0.accept()

    def video_pause(self):
        if self.video_timer_id is not None:
            self.killTimer(self.video_timer_id)
            self.video_timer_id = None
            logging.info('Video playback paused')

    def video_resume(self):
        if self.video_timer_id is None:
            self.video_timer_id = self.startTimer(self.timer_interval, Qt.TimerType.CoarseTimer)
            logging.info('Video playback resumed')

    def video_stop(self):
        self.video_pause()
        self.camera.release()
        self.camera = None
        self.current_frame = None
        self.video_name = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.frame_out_sync = False
        self.frame_display.setText('Video unloaded')
        self.widgets_enabled(False)
        logging.info('Video playback stopped')

    def widgets_enabled(self, status: bool):
        self.button_pause_resume.setEnabled(status)
        self.button_stop.setEnabled(status)
        self.frame_position_slider.setEnabled(status)
        logging.info(f'Widgets: {status}')

    def video_pause_resume(self):
        if self.camera is None:
            logging.error(f'No video file loaded')
            return
        if self.video_timer_id is not None:
            self.video_pause()
        else:
            self.video_resume()

    def open_file(self):
        file_name, video_type = QFileDialog.getOpenFileName(self,
                                                            "Open a video file",
                                                            "videos",
                                                            "*.mp4;;*.asf;;All Files(*)")
        if file_name != '':
            logging.info(f'Open file {file_name}')
            self.video_name = file_name
            self.load_video()

    def load_video(self):
        self.camera = cv2.VideoCapture(self.video_name)
        self.total_frame_number = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_timer_id = self.startTimer(self.timer_interval, Qt.TimerType.CoarseTimer)
        self.frame_position = 0
        self.frame_position_slider.setRange(0, self.total_frame_number)
        self.widgets_enabled(True)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.window_size = a0.size()
        a0.accept()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        cp = self.screen().availableGeometry().center()
        qr = self.frameGeometry()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    od_inspector = ODInspector()
    sys.exit(app.exec())
