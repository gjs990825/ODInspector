import logging
import queue
import sys
import threading
import time
from io import BytesIO

import cv2
import requests
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication, Qt, QTimerEvent
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QMenu, QHBoxLayout, QStyle, \
    QSlider, QFileDialog, QVBoxLayout, QLabel, QSizePolicy, QComboBox, QLineEdit

from maverick.object_detection.api.v1 import ODResult, Model

logging.basicConfig(level=logging.DEBUG)


class InspectorImageProcessInterface:
    def detect(self, image):
        cp = image.copy()
        in_memory_image = self.create_in_memory_image(image)
        results = self.request_detection(in_memory_image)

        t_s = time.time()
        self.get_result_image(cp, results)
        logging.debug('Image drawing uses %.4fs' % (time.time() - t_s))
        return cp

    def request_detection(self, in_memory_image):
        raise NotImplementedError

    @staticmethod
    def create_in_memory_image(image):
        file = BytesIO()
        image_f = Image.frombuffer('RGB', (image.shape[1], image.shape[0]), image, 'raw')
        image_f.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
        file.name = 'test.bmp'
        file.seek(0)
        return file

    @staticmethod
    def get_result_image(image, results: list[ODResult]):
        for result in results:
            cv2.rectangle(image, (result.points[0], result.points[1]),
                          (result.points[2], result.points[3]), (0, 255, 0), thickness=10)

    def get_models(self) -> list[Model]:
        raise NotImplementedError

    def set_current_model(self, model_name):
        raise NotImplementedError

    def get_model_names(self):
        return [model.name for model in self.get_models()]


class ImageProcessor(InspectorImageProcessInterface):
    DETECTION_URL = '/api/v1/detect/'
    GET_MODELS_URL = '/api/v1/model/list'
    SET_MODEL_URL = '/api/v1/model/set'

    def __init__(self, base_url):
        self.base = base_url

    def get_models(self):
        response = requests.get(self.base + self.GET_MODELS_URL)
        return Model.from_json(response.json())

    def set_current_model(self, model_name):
        response = requests.get(self.base + self.SET_MODEL_URL, params={'model_name': model_name})
        logging.info(response.text)

    def request_detection(self, in_memory_image):
        t_s = time.time()
        response = requests.post(self.base + self.DETECTION_URL, data=in_memory_image)
        logging.debug('Request uses %.4fs' % (time.time() - t_s))
        results = ODResult.from_json(response.json())
        logging.info(results)
        return results

    def set_base_url(self, url):
        self.base = url


def clamp(n, min_n, max_n):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    else:
        return n


class ImageProcessQueue(threading.Thread):
    def __init__(self, processor: InspectorImageProcessInterface, result_callback):
        super().__init__()
        self.exit_flag = False
        self.image_queue = queue.Queue(3)
        self.callback = result_callback
        self.processor = processor
        self.queue_lock = threading.Lock()

    def add(self, image):
        self.queue_lock.acquire()
        if self.image_queue.full():
            self.image_queue.get()  # remove one when full
            logging.debug('Lagging occurred')
        self.image_queue.put(image)
        self.queue_lock.release()

    def run(self):
        while self.exit_flag is False:
            self.queue_lock.acquire()
            if not self.image_queue.empty():
                image = self.image_queue.get()
                self.queue_lock.release()
                t_s = time.time()
                output = self.processor.detect(image)
                logging.info('Detection: %.4fs used' % (time.time() - t_s))
                self.callback(output)
            else:
                self.queue_lock.release()
            time.sleep(0.01)  # TODO This loop runs too frequently

    def exit(self):
        self.exit_flag = True
        self.join()


class ODInspector(QMainWindow):
    APP_NAME = "OD Inspector"

    def __init__(self):
        super().__init__()

        self.capture = None
        self.video_name = None
        self.video_fps = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.current_frame = None
        self.current_frame_pixmap = None
        self.current_output_frame = None
        self.current_output_frame_pixmap = None
        self.frame_seeking_flag = False
        self.frame_seeking_position = None
        self.frame_jumping_flag = False
        self.frame_jumping_position = None
        self.image_process_queue = None

        # Some settings
        # self.transformation_mode = Qt.TransformationMode.FastTransformation  # Performance
        self.transformation_mode = Qt.TransformationMode.SmoothTransformation  # Better viewing quality
        self.forward_seconds = 15  # Seconds skipped using arrow key
        self.max_fps = 30  # Limit of output rate, see frame seeking section for details
        self.playback_speed_max = 32.0
        self.playback_speed_min = 1 / 16
        self.playback_speed = 1.0  # Default playback speed
        self.server_url = 'http://localhost:5000'

        self.image_processor = ImageProcessor(self.server_url)

        # Some lambdas
        # Get current playback fps
        self.playback_fps = lambda: self.playback_speed * self.video_fps
        # Get current frame interval with max fps limitation
        self.timer_interval = lambda: int(1000 / min(float(self.max_fps), (self.video_fps * self.playback_speed)))
        # Get frame number needed to skip
        self.forward_frames = lambda: self.video_fps * self.forward_seconds

        model_setting_layout = QHBoxLayout()
        self.server_url_input = QLineEdit(self.server_url)
        load_button = QPushButton('Load')
        load_button.clicked.connect(self.load_model_info)

        self.model_combobox = QComboBox()
        model_setting_layout.addWidget(QLabel('Server Address:'))
        model_setting_layout.addWidget(self.server_url_input, 4)
        model_setting_layout.addWidget(load_button, 2)
        model_setting_layout.addWidget(QLabel('Select Model:'))
        model_setting_layout.addWidget(self.model_combobox, 3)

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

        self.fps_display = QLabel()
        control_center_layout.addWidget(self.fps_display)

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

        # Central widget and it's layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(model_setting_layout, 1)
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
        self.setWindowTitle()
        self.setWindowIcon(QIcon('images/icons/ic_app.png'))
        self.statusBar().showMessage('Ready')
        self.resize(800, 500)
        self.center()
        self.show()

    def load_model_info(self):
        self.image_processor.set_base_url(self.server_url_input.text())
        self.model_combobox.clear()
        for model in self.image_processor.get_models():
            self.model_combobox.addItem(f'{model.name}: {len(model.classes)}class(es)', model.name)
        self.model_combobox.currentIndexChanged.connect(
            lambda: self.image_processor.set_current_model(self.model_combobox.currentData()))

    def check_frame_seeking(self):
        playback_fps = self.playback_fps()
        if playback_fps <= self.max_fps:
            return
        ratio = playback_fps / self.max_fps
        self.frame_seek(self.frame_position + ratio)

    def timerEvent(self, event: QTimerEvent):
        if event.timerId() == self.video_timer_id:
            self.video_timer_handler()

    def video_timer_handler(self):
        self.check_frame_seeking()

        if not self.frame_jumping_flag and not self.frame_seeking_flag:
            self.frame_position += 1
        else:
            if self.frame_seeking_flag:
                if self.frame_position >= self.frame_seeking_position:
                    logging.warning(f'Can not seek backwards')
                    self.frame_position += 1
                else:
                    t_s = time.time()
                    for i in range(int(self.frame_position), int(self.frame_seeking_position) - 1):
                        self.capture.grab()  # grab() does not process frame data, for performance improvement
                    self.frame_position = self.frame_seeking_position
                    t = time.time() - t_s
                    logging_using = logging.debug if self.playback_speed > 1.0 else logging.info
                    logging_using('Seeking from %.1f to %.1f, %.3fs used' %
                                  (self.frame_position, self.frame_seeking_position, t))
                self.frame_seeking_flag = False
            if self.frame_jumping_flag:
                t_s = time.time()
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jumping_position)
                t = time.time() - t_s
                logging.info('Jumping from %.1f to %.1f, %.3fs used' %
                             (self.frame_position, self.frame_jumping_position, t))
                self.frame_position = self.frame_jumping_position
                self.frame_jumping_flag = False

        success, img = self.capture.read()
        if success:
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_image = QtGui.QImage(self.current_frame.data,
                                   self.current_frame.shape[1],
                                   self.current_frame.shape[0],
                                   QtGui.QImage.Format.Format_RGB888)
            self.current_frame_pixmap = QtGui.QPixmap.fromImage(q_image)

            self.display_current_frame()
            self.image_process_queue.add(self.current_frame)

    def display_current_frame(self):
        scaled = self.current_frame_pixmap.scaled(self.frame_input_display.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  self.transformation_mode)
        self.frame_input_display.setPixmap(scaled)
        self.frame_position_slider.setValue(int(self.frame_position))
        logging.debug(f'Showing {self.frame_position}th frame')

    def display_output_frame(self, image):
        self.current_output_frame = image
        q_image = QtGui.QImage(self.current_output_frame.data,
                               self.current_output_frame.shape[1],
                               self.current_output_frame.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        self.current_output_frame_pixmap = QtGui.QPixmap.fromImage(q_image)
        scaled = self.current_output_frame_pixmap.scaled(self.frame_output_display.size(),
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         self.transformation_mode)
        self.frame_output_display.setPixmap(scaled)

    def frame_position_slider_callback(self):
        position = self.frame_position_slider.value()
        logging.debug(f'Slider moved to {position}')
        self.frame_jump_to(position)

    def frame_jump_to(self, position):
        self.frame_jumping_position = position
        self.frame_jumping_flag = True

    def frame_seek(self, position):
        self.frame_seeking_position = position
        self.frame_seeking_flag = True

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent):
        if self.video_timer_id is None:
            a0.accept()
            return
        if a0.key() == Qt.Key.Key_Right:
            position = clamp(self.frame_position + self.forward_frames(), 0, self.total_frame_number)
            self.frame_jump_to(position)
        elif a0.key() == Qt.Key.Key_Left:
            position = clamp(self.frame_position - self.forward_frames(), 0, self.total_frame_number)
            self.frame_jump_to(position)
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

    def video_stop(self):
        if self.image_process_queue is not None:
            self.image_process_queue.exit()
        self.video_pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.current_frame = None
        self.video_name = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.frame_jumping_flag = False
        self.frame_seeking_flag = False
        self.frame_input_display.setText('Video unloaded')
        self.frame_output_display.setText('Video unloaded')
        self.widgets_enabled(False)
        self.setWindowTitle()
        self.fps_display.setText('')
        logging.info('Video playback stopped')

    def widgets_enabled(self, status: bool):
        self.button_pause_resume.setEnabled(status)
        self.button_stop.setEnabled(status)
        self.frame_position_slider.setEnabled(status)
        self.button_speed_half.setEnabled(status)
        self.button_speed_double.setEnabled(status)
        logging.info(f'Widgets: {status}')

    def video_pause_resume(self):
        if self.capture is None:
            logging.error(f'No video file loaded')
            return
        if self.video_timer_id is not None:
            self.video_pause()
        else:
            self.video_resume()

    def set_playback_speed(self, speed):
        self.playback_speed = clamp(speed, self.playback_speed_min, self.playback_speed_max)
        self.video_pause()
        self.video_resume()
        self.fps_display.setText(f'<b>{self.playback_speed}x '
                                 f'{self.video_fps * self.playback_speed}fps</b>')

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
        self.capture = cv2.VideoCapture(self.video_name)
        self.total_frame_number = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps > 60:
            logging.error(f'Abnormal fps: {self.video_fps}, reset to default fps')
            self.video_fps = 25
        else:
            logging.info(f'Video fps: {self.video_fps}')
        self.frame_position = 0
        self.frame_position_slider.setRange(0, self.total_frame_number)
        self.setWindowTitle(self.video_name)
        self.widgets_enabled(True)
        self.set_playback_speed(1.0)
        self.image_process_queue = ImageProcessQueue(self.image_processor, self.display_output_frame)
        self.image_process_queue.start()
        self.video_resume()

    def setWindowTitle(self, title: str = None):
        if title is None:
            super().setWindowTitle(self.APP_NAME)
            return
        if len(title) > 55:
            title = title[:25] + '...' + title[-25:]
        super().setWindowTitle(f'{self.APP_NAME} - {title}')

    def closeEvent(self, event: QtGui.QCloseEvent):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.video_stop()  # Stop all work
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
