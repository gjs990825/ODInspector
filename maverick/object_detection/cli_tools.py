import argparse
import os

from maverick.object_detection.analyzer import *
from maverick.object_detection.api.v2 import ODServiceInterface, DetectionConfig
from maverick.object_detection.api.v2.helper import ImageProcessingHelper

analyzer_class_names = [cls.__name__ for cls in ODResultAnalyzer.__subclasses__()]


def launch(service: ODServiceInterface):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='source video')
    parser.add_argument(f'--config_name', type=str, default=None, help=f'configuration name')
    parser.add_argument('--output_directory', type=str, default='./')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='video result output')
    opt = parser.parse_args()

    config_name = opt.config_name
    video_path = opt.input
    output_directory = opt.output_directory
    output_video = opt.output_video

    os.makedirs(output_directory, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    total_frame_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video frame count: {total_frame_number}')
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(size)

    output_video_path = os.path.join(output_directory, output_video)
    recorder = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(video_fps), size)

    analyzers: list[ODResultAnalyzer] = []
    configs = DetectionConfig.from_file('configs/detection_configs.json')
    config = next(config for config in configs if config.name == config_name)
    for analyzer_path in config.analyzer_files:
        analyzers.extend(ODResultAnalyzer.from_file(analyzer_path))

    if len(analyzers) == 0:
        print('No analyzers configured')

    processing_helper = ImageProcessingHelper(service, analyzers)
    processing_helper.service.update_models()
    processing_helper.set_model_names(config.model_names)
    processing_helper.set_analyzers(analyzers)
    processing_helper.update_colors()

    for analyzer in analyzers:
        analyzer.redirect_saving_path(output_directory)
        analyzer.set_class_name_converter(service.convert_name)
    print(analyzers)

    output_playback_fps = 0
    output_playback_last_t = time.time()

    count = 0
    while True:
        success, img = capture.read()
        if not success:
            break
        input_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_rgb = processing_helper.detect(input_rgb, config.model_names)  # TODO: use threading here?
        recorder.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

        now = time.time()
        output_playback_fps = (output_playback_fps + (1.0 / (now - output_playback_last_t))) / 2
        output_playback_last_t = now

        count += 1
        if count % 10 == 0:
            print(f'{count}th frame, fps: {output_playback_fps}')

    capture.release()
    recorder.release()
    print('DONE')
