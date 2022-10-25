import argparse
import os

from maverick.object_detection import ImageProcessingHelper, ODServiceInterface
from maverick.object_detection.analyzer import *
from maverick.object_detection.utils import camel_to_snake

analyzer_class_names = [cls.__name__ for cls in ODResultAnalyzer.__subclasses__()]


def launch(service: ODServiceInterface):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='source video')
    parser.add_argument('--model_name', type=str, default='yolov5m')

    for name in analyzer_class_names:
        parser.add_argument(f'--{camel_to_snake(name)}', type=str, default=None, help=f'{name} configuration file')
    parser.add_argument('--output_directory', type=str, default='./')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='video result output')
    opt = parser.parse_args()

    video_path = opt.input
    model_name = opt.model_name
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

    for name in analyzer_class_names:
        config_file = getattr(opt, camel_to_snake(name), None)
        if config_file is None:
            continue
        analyzers.extend(globals()[name].from_file(config_file))

    if len(analyzers) == 0:
        print('No analyzers configured')

    processing_helper = ImageProcessingHelper(service, analyzers)
    processing_helper.service.update_models()
    processing_helper.set_current_model(model_name)

    for analyzer in analyzers:
        analyzer.redirect_saving_path(output_directory)
        analyzer.set_class_name_converter(service.current_model.class_name_converter)
    print(analyzers)

    output_playback_fps = 0
    output_playback_last_t = time.time()

    count = 0
    while True:
        success, img = capture.read()
        if not success:
            break
        input_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_rgb = processing_helper.detect(input_rgb)  # TODO: use threading here?
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
