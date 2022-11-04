import os
import queue
import sys
import threading

import requests
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication, Qt, QTimerEvent
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QMenu, QHBoxLayout, QStyle, \
    QSlider, QFileDialog, QVBoxLayout, QLabel, QSizePolicy, QComboBox, QLineEdit, QCheckBox

from maverick.object_detection.analyzer import *
from maverick.object_detection.api.v2 import Model, ODServiceOverNetworkConfig, ODServiceInterface, \
    DetectionSequenceConfig
from maverick.object_detection.api.v2.helper import ImageProcessingHelper
from maverick.object_detection.utils import clamp, create_in_memory_image, camel_to_snake

logging.basicConfig(level=logging.INFO)


class ODServiceOverNetworkClient(ODServiceInterface):
    def __init__(self, base_url, proxies=None):
        super().__init__()
        self.base = base_url
        self.proxies = proxies

        if self.proxies is not None:
            logging.warning('Proxy setting might introduce serious detection lagging')
        else:
            logging.warning('No proxy designated, this can be a problem when your system proxy is set but proxy '
                            'server is not running.')

    def update_models(self):
        response = requests.get(self.base + ODServiceOverNetworkConfig.PATH_LIST_MODELS, proxies=self.proxies)
        self.models = Model.from_json(response.json())

    def do_detections(self, image: numpy.ndarray, model_names: list[str]) -> list[ODResult]:
        t_s = time.time()
        in_mem_image = create_in_memory_image(image)
        params = {'model_names': model_names}
        response = requests.post(self.base + ODServiceOverNetworkConfig.PATH_DETECT_WITH_BINARY,
                                 data=in_mem_image,
                                 params=params,
                                 proxies=self.proxies)
        logging.debug('Request uses %.4fs' % (time.time() - t_s))
        logging.debug(response.json())
        results = ODResult.from_json(response.json())
        logging.debug(results)
        return results

    def set_base_url(self, url):
        self.base = url


class ImageProcessingQueue(threading.Thread):
    def __init__(self, helper: ImageProcessingHelper, result_callback):
        super().__init__()
        self.exit_flag = False
        self.queue = queue.Queue(3)
        self.callback = result_callback
        self.helper = helper
        self.queue_lock = threading.Lock()
        self.is_processing = False

    def add(self, item: tuple[numpy.ndarray, str]):
        self.queue_lock.acquire()
        if self.queue.full():
            self.queue.get()  # remove one when full
            logging.debug('Lagging occurred')
        self.queue.put(item)
        self.queue_lock.release()

    def run(self):
        while self.exit_flag is False:
            self.queue_lock.acquire()
            if not self.queue.empty():
                item = self.queue.get()
                self.is_processing = True
                self.queue_lock.release()
                t_s = time.time()
                output = self.helper.detect(*item)
                logging.debug('Detection: %.4fs used' % (time.time() - t_s))
                self.callback(output)
                self.is_processing = False
            else:
                self.queue_lock.release()
            time.sleep(0.01)  # TODO This loop runs too frequently

    def finished(self):
        self.queue_lock.acquire()
        ret = self.queue.empty()
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

        self.recorder = None
        self.capture = None
        self.video_path = None
        self.video_fps = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.current_frame = None
        self.current_output_frame = None
        self.frame_seeking_flag = False
        self.frame_seeking_position = None
        self.frame_jumping_flag = False
        self.frame_jumping_position = None
        self.image_process_queue = None
        self.configs = None
        self.current_config = None

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
        self.show_input = False  # Display input frame
        self.server_url = 'http://localhost:5000'
        self.window_size = 1200, 800
        self.save_path = './recordings'  # recording path

        self.processing_helper = ImageProcessingHelper(ODServiceOverNetworkClient(self.server_url))

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

        self.config_combobox = QComboBox()
        model_setting_layout.addWidget(QLabel('Server Address:'))
        model_setting_layout.addWidget(self.server_url_input, 4)
        model_setting_layout.addWidget(load_button, 2)
        model_setting_layout.addWidget(QLabel('Select Model:'))
        model_setting_layout.addWidget(self.config_combobox, 3)

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
        self.button_stop.clicked.connect(lambda: self.video_stop())
        control_center_layout.addWidget(self.button_stop, 2)

        fastforward_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        self.button_speed_double = QPushButton("2x")
        self.button_speed_double.clicked.connect(lambda: self.set_playback_speed(self.playback_speed * 2))
        self.button_speed_double.setIcon(fastforward_icon)
        control_center_layout.addWidget(self.button_speed_double, 1)

        self.button_speed_half = QPushButton("0.5x")
        self.button_speed_half.clicked.connect(lambda: self.set_playback_speed(self.playback_speed * 0.5))
        self.button_speed_half.setIcon(fastforward_icon)
        control_center_layout.addWidget(self.button_speed_half, 1)

        self.fps_display = QLabel()
        control_center_layout.addWidget(self.fps_display)

        frame_sync_check_box = QCheckBox('Frame Sync')
        frame_sync_check_box.toggled.connect(self.set_frame_sync)
        control_center_layout.addWidget(frame_sync_check_box)

        show_input_check_box = QCheckBox('Show Input')
        show_input_check_box.setChecked(self.show_input)
        show_input_check_box.toggled.connect(self.set_show_input)
        control_center_layout.addWidget(show_input_check_box)

        self.play_back_infos = QHBoxLayout()
        self.input_playback_info = QLabel()
        self.input_playback_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_playback_info = QLabel()
        self.output_playback_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_back_infos.addWidget(self.input_playback_info)
        self.play_back_infos.addWidget(self.output_playback_info)

        # Input and output viewport
        self.frame_input_display = QLabel('Press Ctrl+O to open a video')
        self.frame_input_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_input_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.frame_output_display = QLabel('Output will be here')
        self.frame_output_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_output_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.viewport_layout = QHBoxLayout()
        self.viewport_layout.addWidget(self.frame_input_display)
        self.viewport_layout.addWidget(self.frame_output_display)

        # Central widget and it's layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(model_setting_layout, 1)
        main_layout.addLayout(self.viewport_layout, 10)
        main_layout.addLayout(self.play_back_infos)
        main_layout.addLayout(control_center_layout, 1)
        main_layout.addWidget(self.frame_position_slider)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)

        # Configure menu bar items
        file_menu = QMenu('&File', self)
        menu_bar = self.menuBar()
        menu_bar.addMenu(file_menu)

        analyzer_menu = QMenu('&Analyzer', self)
        menu_bar.addMenu(analyzer_menu)

        for analyzer_cls, analyzer_name in [(cls, cls.__name__) for cls in ODResultAnalyzer.__subclasses__()]:
            action = QAction(f'Add {analyzer_name}', self)
            action.setStatusTip(f'Open {analyzer_name} config file')
            action.triggered.connect(lambda _, cls=analyzer_cls: self.load_analyzer(cls))
            analyzer_menu.addAction(action)

        clear_analyzer_action = QAction('Clear Analyzer', self)
        clear_analyzer_action.setShortcut('Ctrl+Shift+A')
        clear_analyzer_action.triggered.connect(lambda: self.processing_helper.set_analyzers([]))
        analyzer_menu.addAction(clear_analyzer_action)

        recorder_menu = QMenu('&Recorder', self)
        menu_bar.addMenu(recorder_menu)
        start_rec_action = QAction('Start Recording', self)
        start_rec_action.setShortcut('Ctrl+R')
        start_rec_action.setStatusTip('Save output as video')
        start_rec_action.triggered.connect(self.start_recording)
        stop_rec_action = QAction('Stop Recording', self)
        stop_rec_action.setShortcut('Ctrl+Shift+R')
        stop_rec_action.triggered.connect(self.stop_recording)
        recorder_menu.addAction(start_rec_action)
        recorder_menu.addAction(stop_rec_action)

        open_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveDVDIcon)
        open_action = QAction(open_icon, '&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open video file')
        open_action.triggered.connect(self.open_video_file)
        file_menu.addAction(open_action)

        exit_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        exit_action = QAction(exit_icon, '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QCoreApplication.exit)
        # TODO exit problem
        file_menu.addAction(exit_action)

        self.widgets_enabled(False)

        # Window management
        self.set_show_input(self.show_input)
        self.setCentralWidget(central_widget)
        self.setWindowTitle()
        self.setWindowIcon(QIcon('images/icons/ic_app.png'))
        self.statusBar().showMessage('Ready')
        self.resize(*self.window_size)
        self.center()
        self.show()

    def start_recording(self):
        try:
            size = self.get_output_size()
        except ValueError:
            self.statusBar().showMessage('Nothing playing!')
            return
        path = os.path.join(self.save_path, f'{time.time()}.mp4')
        logging.info(f'Recording to {path}')
        self.recorder = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.video_fps), size)
        self.statusBar().showMessage('Recording...')

    def stop_recording(self):
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None
            self.statusBar().showMessage('Stop recording.')

    def set_frame_sync(self, status):
        logging.info(f'Frame sync: {status}')
        self.frame_sync = status

    def set_show_input(self, status):
        logging.info(f'Show input: {status}')
        self.show_input = status
        if status:
            self.viewport_layout.removeWidget(self.frame_output_display)
            self.viewport_layout.addWidget(self.frame_input_display)
            self.viewport_layout.addWidget(self.frame_output_display)

            self.play_back_infos.removeWidget(self.output_playback_info)
            self.play_back_infos.addWidget(self.input_playback_info)
            self.play_back_infos.addWidget(self.output_playback_info)
        else:
            self.frame_input_display.clear()
            self.viewport_layout.removeWidget(self.frame_input_display)

            self.input_playback_info.clear()
            self.play_back_infos.removeWidget(self.input_playback_info)

    def load_model_info(self):
        self.server_url = self.server_url_input.text()
        if isinstance(self.processing_helper.service, ODServiceOverNetworkClient):
            self.processing_helper.service.set_base_url(self.server_url)
        self.config_combobox.clear()

        self.configs = DetectionSequenceConfig.from_file('configs/detection_sequence_configs.json')
        for config in self.configs:
            self.config_combobox.addItem(f'{config.name}: {len(config.model_names)}model(s)', config)
        self.config_combobox.currentIndexChanged.connect(
            lambda: self.change_config(self.config_combobox.currentData()))
        self.change_config(self.configs[0])
        self.processing_helper.service.update_models()
        self.processing_helper.update_colors()

    def change_config(self, config):
        if config is None:
            return
        self.current_config = config

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

            if self.image_process_queue is not None:
                self.image_process_queue.add((self.current_frame, self.current_config.model_names))
                if self.frame_sync:
                    self.image_process_queue.wait_for_complete()
            if self.show_input:
                self.display_current_frame()

            self.frame_position_slider.setValue(int(self.frame_position))
        else:
            playback_end = self.frame_position >= self.total_frame_number

            if playback_end:
                self.video_pause()
                self.frame_jump_to(0)  # From the top!
            else:
                message = 'Frame read failed, file might be corrupted'
                logging.warning(message)
                logging.warning(f'Frame info:current {self.frame_position}, total {self.total_frame_number}')
                self.video_stop(message)

    def get_output_size(self):
        if self.current_frame is None:
            raise ValueError
        return self.current_frame.shape[1], self.current_frame.shape[0]

    def ui_image_process(self, source, target):
        q_image = QtGui.QImage(source.data,
                               source.shape[1],
                               source.shape[0],
                               QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(target.size(),
                               Qt.AspectRatioMode.KeepAspectRatio,
                               self.transformation_mode)
        target.setPixmap(scaled)

    def display_current_frame(self):
        now = time.time()
        self.input_playback_fps = (self.input_playback_fps + (1.0 / (now - self.input_playback_last_t))) / 2
        self.input_playback_last_t = now
        self.input_playback_info.setText("Input: %.1f FPS" % self.input_playback_fps)

        self.ui_image_process(self.current_frame, self.frame_input_display)
        logging.debug(f'Showing {self.frame_position}th frame')

    def display_output_frame(self, image):
        self.current_output_frame = image

        now = time.time()
        self.output_playback_fps = (self.output_playback_fps + (1.0 / (now - self.output_playback_last_t))) / 2
        self.output_playback_last_t = now
        self.output_playback_info.setText("Output: %.1f FPS" % self.output_playback_fps)

        self.ui_image_process(self.current_output_frame, self.frame_output_display)

        if self.recorder is not None:
            self.recorder.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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
            message = 'Playback Paused'
            self.statusBar().showMessage(message)
            logging.info(message)

    def video_resume(self):
        if self.video_timer_id is None:
            play_pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.button_pause_resume.setIcon(play_pause_icon)
            self.button_pause_resume.setText('Pause')
            self.video_timer_id = self.startTimer(self.timer_interval(), Qt.TimerType.CoarseTimer)
            message = 'Playback Resumed'
            self.statusBar().showMessage(message)
            logging.info(message)

    def video_stop(self, message=None):
        if self.image_process_queue is not None:
            self.image_process_queue.exit()
        self.video_pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.current_frame = None
        self.video_path = None
        self.total_frame_number = None
        self.frame_position = None
        self.video_timer_id = None
        self.frame_jumping_flag = False
        self.frame_seeking_flag = False
        message = 'Playback Stopped' if message is None else message
        logging.info(message)
        if self.show_input:
            self.frame_input_display.setText(message)
        self.frame_output_display.setText(message)
        self.input_playback_info.clear()
        self.output_playback_info.clear()
        self.widgets_enabled(False)
        self.fps_display.clear()
        self.setWindowTitle()
        self.statusBar().showMessage(message)
        logging.info(message)

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
        self.fps_display.setText('<b>%.2fx -> %.2f FPS</b>' %
                                 (self.playback_speed, self.video_fps * self.playback_speed))

    def open_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open a video file', 'videos', '*.mp4;;All Files(*)')
        if file_name == '':
            return
        logging.info(f'Video file: {file_name}')
        self.video_path = file_name
        self.load_video()

    def load_analyzer(self, analyzer_cls):
        analyzer_name = analyzer_cls.__name__
        path = os.path.join('configs/', camel_to_snake(analyzer_name) + 's')
        logging.info(f'Open path: {path}')
        file_name, _ = QFileDialog.getOpenFileName(self, f'Open a {analyzer_name} configuration file', path, '*.json')
        if file_name == '':
            return
        logging.info(f'Open file {file_name} for {analyzer_name}')
        analyzers = analyzer_cls.from_file(file_name)
        self.processing_helper.set_analyzers(analyzers)

    def load_video(self):
        self.capture = cv2.VideoCapture(self.video_path)
        self.total_frame_number = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f'Video frame count: {self.total_frame_number}')
        if self.total_frame_number <= 0:
            self.video_stop('Frame error, file might be corrupted')
            return

        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps > 60:
            logging.error(f'Abnormal fps: {self.video_fps}, reset to default fps')
            self.video_fps = 25
        else:
            logging.info(f'Video fps: {self.video_fps}')
        self.frame_position = 0
        self.frame_position_slider.setRange(0, self.total_frame_number)
        self.setWindowTitle(self.video_path)
        self.widgets_enabled(True)
        self.set_playback_speed(1.0)
        self.image_process_queue = ImageProcessingQueue(self.processing_helper, self.display_output_frame)
        self.image_process_queue.start()
        self.video_resume()

    def setWindowTitle(self, title: str = None):
        if title is None:
            super().setWindowTitle(self.APP_NAME)
            return
        if len(title) > 55:
            title = f'{title[:25]}...{title[-25:]}'
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
