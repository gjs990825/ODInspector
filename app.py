import sys
import logging
import random

import cv2
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication, Qt, QTimerEvent
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QMenu, QHBoxLayout, QStyle, \
    QSlider, QFileDialog, QVBoxLayout, QLabel, QSizePolicy

logging.basicConfig(level=logging.INFO)


class InspectorImageProcessInterface:
    def detect(self, image):
        result = image.copy()
        c = random.randint(0, 0xFFFFFF)
        color = (c & 0xFF, c >> 8 & 0xFF, c >> 16 & 0xFF)
        cv2.putText(result, 'PROCESSED', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        return result


image_process = InspectorImageProcessInterface()


def clamp(n, min_n, max_n):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    else:
        return n


class ODInspector(QMainWindow):

    def __init__(self):
        super().__init__()

        self.current_frame = None
        self.current_frame_pixmap = None
        self.current_output_frame = None
        self.current_output_frame_pixmap = None
        self.camera = None
        self.video_name = None
        self.total_frame_number = None
        self.frame_position: int = 0
        self.video_timer_id = None
        self.frame_out_sync = False

        # Some default settings
        self.skip_seconds = 15
        self.fps = 25
        self.playback_speed = 1.0
        self.timer_interval = lambda: int(1000 / (self.fps * self.playback_speed))
        self.skip_frames = lambda: self.fps * self.skip_seconds

        # Slider configuration
        self.frame_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_position_slider.sliderReleased.connect(self.video_resume)
        self.frame_position_slider.sliderPressed.connect(self.video_pause)
        self.frame_position_slider.sliderMoved.connect(self.frame_position_slider_callback)

        # Control center (just some buttons
        control_center_layout = QHBoxLayout()

        self.button_pause_resume = QPushButton("Resume")
        play_pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.button_pause_resume.setIcon(play_pause_icon)
        self.button_pause_resume.setShortcut('Ctrl+P')
        self.button_pause_resume.clicked.connect(self.video_pause_resume)
        control_center_layout.addWidget(self.button_pause_resume, 2)

        self.button_stop = QPushButton("&Stop")
        stop_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        self.button_stop.setIcon(stop_icon)
        # self.button_pause_resume.setShortcut(QKeySequence('Ctrl+Z'))
        self.button_stop.clicked.connect(self.video_stop)
        control_center_layout.addWidget(self.button_stop, 2)

        fastforward_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        self.button_speed_double = QPushButton("2x")
        self.button_speed_double.clicked.connect(self.speed_double)
        self.button_speed_double.setIcon(fastforward_icon)
        control_center_layout.addWidget(self.button_speed_double, 1)

        self.button_speed_half = QPushButton("0.5x")
        self.button_speed_half.clicked.connect(self.speed_half)
        self.button_speed_half.setIcon(fastforward_icon)
        control_center_layout.addWidget(self.button_speed_half, 1)

        # Input and output viewport
        self.frame_input_display = QLabel('Press Ctrl+O to open a video')
        self.frame_input_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_input_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.frame_output_display = QLabel('Output will be here')
        self.frame_output_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_output_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        viewport_layout = QHBoxLayout()
        viewport_layout.addWidget(self.frame_input_display, 1)
        viewport_layout.addWidget(self.frame_output_display, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(viewport_layout, 10)
        main_layout.addLayout(control_center_layout, 1)
        main_layout.addWidget(self.frame_position_slider)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)

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
        self.resize(800, 500)
        self.center()
        self.show()

    def timerEvent(self, event: QTimerEvent):
        if event.timerId() == self.video_timer_id:
            self.video_timer_handler()

    def video_timer_handler(self):
        if not self.frame_out_sync:
            self.frame_position += 1
        else:
            logging.info(f'Seeking to {self.frame_position}th frame')
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)
            self.frame_out_sync = False

        success, img = self.camera.read()
        if success:
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_image = QtGui.QImage(self.current_frame.data,
                                   self.current_frame.shape[1],
                                   self.current_frame.shape[0],
                                   QtGui.QImage.Format.Format_RGB888)
            self.current_frame_pixmap = QtGui.QPixmap.fromImage(q_image)

            self.current_output_frame = image_process.detect(self.current_frame)
            q_image = QtGui.QImage(self.current_output_frame.data,
                                   self.current_output_frame.shape[1],
                                   self.current_output_frame.shape[0],
                                   QtGui.QImage.Format.Format_RGB888)
            self.current_output_frame_pixmap = QtGui.QPixmap.fromImage(q_image)

            self.display_current_frame()

    def display_current_frame(self):
        scaled = self.current_frame_pixmap.scaled(self.frame_input_display.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
        self.frame_input_display.setPixmap(scaled)

        scaled = self.current_output_frame_pixmap.scaled(self.frame_output_display.size(),
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)
        self.frame_output_display.setPixmap(scaled)

        self.frame_position_slider.setValue(self.frame_position)
        logging.debug(f'Showing {self.frame_position}th frame')

    def frame_position_slider_callback(self):
        position = self.frame_position_slider.value()
        self.update_frame_position(position)
        logging.debug(f'Slider moved to {position}')

    def update_frame_position(self, position):
        self.frame_position = position
        self.frame_out_sync = True

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent):
        if self.video_timer_id is None:
            a0.accept()
            return
        if a0.key() == Qt.Key.Key_Right:
            position = clamp(self.frame_position + self.skip_frames(), 0, self.total_frame_number)
            self.update_frame_position(position)
        elif a0.key() == Qt.Key.Key_Left:
            position = clamp(self.frame_position - self.skip_frames(), 0, self.total_frame_number)
            self.update_frame_position(position)
        else:
            a0.accept()

    def video_pause(self):
        if self.video_timer_id is not None:
            self.killTimer(self.video_timer_id)
            self.video_timer_id = None
            play_pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.button_pause_resume.setIcon(play_pause_icon)
            self.button_pause_resume.setText('Resume')
            logging.info('Video playback paused')

    def video_resume(self):
        if self.video_timer_id is None:
            play_pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.button_pause_resume.setIcon(play_pause_icon)
            self.button_pause_resume.setText('Pause')
            self.video_timer_id = self.startTimer(self.timer_interval(), Qt.TimerType.CoarseTimer)
            logging.info('Video playback resumed')

    def video_restart(self):
        self.video_pause()
        self.video_resume()

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
        self.frame_input_display.setText('Video unloaded')
        self.frame_output_display.setText('Video unloaded')
        self.widgets_enabled(False)
        logging.info('Video playback stopped')

    def widgets_enabled(self, status: bool):
        self.button_pause_resume.setEnabled(status)
        self.button_stop.setEnabled(status)
        self.frame_position_slider.setEnabled(status)
        self.button_speed_half.setEnabled(status)
        self.button_speed_double.setEnabled(status)
        logging.info(f'Widgets: {status}')

    def video_pause_resume(self):
        if self.camera is None:
            logging.error(f'No video file loaded')
            return
        if self.video_timer_id is not None:
            self.video_pause()
        else:
            self.video_resume()

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        self.video_restart()
        self.statusBar().showMessage(f'Playback speed: {speed}x', 1000)

    def speed_double(self):
        self.set_playback_speed(self.playback_speed * 2)

    def speed_half(self):
        self.set_playback_speed(self.playback_speed * 0.5)

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
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        if self.fps > 60:
            logging.error(f'Abnormal fps: {self.fps}, reset to default fps')
            self.fps = 25
        else:
            logging.info(f'Video fps: {self.fps}')
        self.frame_position = 0
        self.frame_position_slider.setRange(0, self.total_frame_number)
        self.widgets_enabled(True)
        self.video_resume()

    def closeEvent(self, event: QtGui.QCloseEvent):
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
