import colorsys
import logging
import queue
import sys
import threading
import time

import cv2
import numpy
import requests
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication, Qt, QTimerEvent
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QMenu, QHBoxLayout, QStyle, \
    QSlider, QFileDialog, QVBoxLayout, QLabel, QSizePolicy, QComboBox, QLineEdit, QCheckBox

from maverick.object_detection.api.utils import create_in_memory_image
from maverick.object_detection.api.v1 import ODResult, Model, ODServiceInterface

logging.basicConfig(level=logging.DEBUG)


class InspectorImageProcessInterface:
    def __init__(self, binary_result=False):
        self.binary_result = binary_result
        self.models = []
        self.current_model = None
        self.current_classes = []
        self.colors = []

    def detect(self, image):
        in_memory_image = create_in_memory_image(image)
        if not self.binary_result:
            cp = image.copy()
            results = self.request_detection(in_memory_image)
            self.get_result_image(cp, results)
            return cp
        else:
            return self.request_detection_for_image_result(in_memory_image)

    def request_detection(self, in_memory_image) -> list[ODResult]:
        raise NotImplementedError

    def request_detection_for_image_result(self, in_memory_image):
        raise NotImplementedError

    def get_result_image(self, image, results: list[ODResult]):
        t_s = time.time()
        thickness = max(int(min((image.shape[1], image.shape[0])) / 150), 1)
        for result in results:
            label = '{} {:.2f}'.format(result.label, float(result.confidence))
            color = self.get_class_color(result.label)
            p1 = (result.points[0], result.points[1])
            p2 = (result.points[2], result.points[3])
            cv2.rectangle(image, p1, p2, color, thickness)
            cv2.putText(image, label, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        color, 2)
        logging.debug('Image drawing uses %.4fs' % (time.time() - t_s))

    def get_models(self) -> list[Model]:
        if len(self.models) == 0:
            self.update_models()
        return self.models

    def update_models(self):
        raise NotImplementedError

    def get_current_classes(self) -> list[str]:
        return self.current_model.classes

    def set_current_model(self, model_name):
        self.current_model = next(model for model in self.models if model.name == model_name)
        self.current_classes = self.current_model.classes
        self.colors = self.get_colors(self.current_classes)

    def get_class_color(self, class_name):
        try:
            index = self.current_classes.index(class_name)
            return self.colors[index]
        except ValueError as e:
            logging.error(e)
            return self.colors[0]

    def get_model_names(self):
        return [model.name for model in self.get_models()]

    @staticmethod
    def get_colors(classes):
        num_classes = len(classes)
        # see: https://github.com/bubbliiiing/yolov7-pytorch/blob/master/yolo.py#L92
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        return colors


class ImageProcessor(InspectorImageProcessInterface):
    proxies = {
        "http": None,
        "https": None,
    }

    def __init__(self, base_url, binary_result=False):
        super().__init__(binary_result)
        self.base = base_url

    def update_models(self):
        response = requests.get(self.base + ODServiceInterface.PATH_LIST_MODELS, proxies=self.proxies)
        self.models = Model.from_json(response.json())
        self.set_current_model(self.models[0].name)

    def set_current_model(self, model_name):
        super().set_current_model(model_name)
        response = requests.get(self.base + ODServiceInterface.PATH_SET_MODEL, params={'model_name': model_name},
                                proxies=self.proxies)
        logging.info(response.text)

    def request_detection(self, in_memory_image):
        t_s = time.time()
        response = requests.post(self.base + ODServiceInterface.PATH_DETECT_WITH_BINARY, data=in_memory_image,
                                 proxies=self.proxies)
        logging.debug('Request uses %.4fs' % (time.time() - t_s))
        results = ODResult.from_json(response.json())
        logging.info(results)
        return results

    def request_detection_for_image_result(self, in_memory_image):
        t_s = time.time()
        response = requests.post(self.base + ODServiceInterface.PATH_DETECT_WITH_BINARY_FOR_IMAGE_RESULT,
                                 data=in_memory_image, proxies=self.proxies)
        logging.debug('Request for image result uses %.4fs' % (time.time() - t_s))
        buffer = numpy.frombuffer(response.content, dtype=numpy.uint8)
        image_result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

    def set_base_url(self, url):
        self.base = url


class DummyImageProcessor(InspectorImageProcessInterface):
    def __init__(self, sleep=0.0):
        super().__init__()
        self.sleep = sleep

    def request_detection(self, in_memory_image) -> list[ODResult]:
        time.sleep(self.sleep)
        return []

    def update_models(self):
        self.models = [
            Model('Model1', 'model_path1', 'weight_path1', ['class1', 'class2']),
            Model('Model2', 'model_path2', 'weight_path2', ['class1', 'class2', 'class3'])
        ]

    def set_base_url(self, url):
        pass


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
        self.is_processing = False

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
                self.is_processing = True
                self.queue_lock.release()
                t_s = time.time()
                output = self.processor.detect(image)
                logging.info('Detection: %.4fs used' % (time.time() - t_s))
                self.callback(output)
                self.is_processing = False
            else:
                self.queue_lock.release()
            time.sleep(0.01)  # TODO This loop runs too frequently

    def finished(self):
        self.queue_lock.acquire()
        ret = self.image_queue.empty()
        self.queue_lock.release()
        return ret

    def wait_for_complete(self):
        while not self.finished() or self.is_processing:
            time.sleep(0.01)

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

        self.input_playback_fps = 0.0
        self.input_playback_last_t = 0
        self.output_playback_fps = 0.0
        self.output_playback_last_t = 0

        # Some settings
        # self.transformation_mode = Qt.TransformationMode.FastTransformation  # Performance
        self.transformation_mode = Qt.TransformationMode.SmoothTransformation  # Better viewing quality
        self.forward_seconds = 15  # Seconds skipped using arrow key
        self.max_fps = 60  # Limit of output rate, see frame seeking section for details
        self.playback_speed_max = 32.0
        self.playback_speed_min = 1 / 16
        self.playback_speed = 1.0  # Default playback speed
        self.frame_sync = False  # Wait the detection output
        self.server_url = 'http://localhost:5000'

        # self.image_processor = ImageProcessor(self.server_url, binary_result=False)
        self.image_processor = ImageProcessor(self.server_url, binary_result=True)
        # self.image_processor = DummyImageProcessor(1/15)  # Fake image processor that processes 15 image per second

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

        frame_sync_check_box = QCheckBox('Frame Sync')
        frame_sync_check_box.toggled.connect(self.set_frame_sync)
        control_center_layout.addWidget(frame_sync_check_box)

        play_back_infos = QHBoxLayout()
        self.input_playback_info = QLabel()
        self.input_playback_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_playback_info = QLabel()
        self.output_playback_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        play_back_infos.addWidget(self.input_playback_info)
        play_back_infos.addWidget(self.output_playback_info)

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
        main_layout.addLayout(play_back_infos, 1)
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

    def set_frame_sync(self, status):
        logging.info(f'Frame sync: {status}')
        self.frame_sync = status

    def load_model_info(self):
        self.server_url = self.server_url_input.text()
        self.image_processor.set_base_url(self.server_url)
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

            if self.image_process_queue is not None:
                self.image_process_queue.add(self.current_frame)
                if self.frame_sync:
                    self.image_process_queue.wait_for_complete()
            self.display_current_frame()
        else:
            logging.warning('Frame read failed')
            self.video_pause()

    def display_current_frame(self):
        now = time.time()
        self.input_playback_fps = (self.input_playback_fps + (1.0 / (now - self.input_playback_last_t))) / 2
        self.input_playback_last_t = now
        self.input_playback_info.setText("Input: %.1f FPS" % self.input_playback_fps)
        scaled = self.current_frame_pixmap.scaled(self.frame_input_display.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  self.transformation_mode)
        self.frame_input_display.setPixmap(scaled)
        self.frame_position_slider.setValue(int(self.frame_position))
        logging.debug(f'Showing {self.frame_position}th frame')

    def display_output_frame(self, image):
        now = time.time()
        self.output_playback_fps = (self.output_playback_fps + (1.0 / (now - self.output_playback_last_t))) / 2
        self.output_playback_last_t = now
        self.output_playback_info.setText("Output: %.1f FPS" % self.output_playback_fps)

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
        self.input_playback_info.clear()
        self.output_playback_info.clear()
        self.widgets_enabled(False)
        self.fps_display.clear()
        self.setWindowTitle()
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
        self.fps_display.setText(f'<b>{self.playback_speed}x -> {self.video_fps * self.playback_speed} FPS</b>')

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
