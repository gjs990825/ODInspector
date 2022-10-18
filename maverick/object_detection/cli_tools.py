import argparse
import time

import cv2

from maverick.object_detection import ImageProcessingHelper, ODServiceInterface
from maverick.object_detection.analyzer import TrespassingAnalyzer, IllegalEnteringAnalyzer


def launch(service: ODServiceInterface):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model_name', type=str, default='yolov5m')
    parser.add_argument('--trespassing_analyzer', type=str, default=None)
    parser.add_argument('--illegal_entering_analyzer', type=str, default=None)
    parser.add_argument('--output', type=str, default='./output.mp4', help='video result output')
    opt = parser.parse_args()

    video_path = opt.input
    model_name = opt.model_name
    trespassing_analyzer = opt.trespassing_analyzer
    illegal_entering_analyzer = opt.illegal_entering_analyzer
    output_rgb = opt.output

    capture = cv2.VideoCapture(video_path)
    total_frame_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video frame count: {total_frame_number}')
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(size)

    recorder = cv2.VideoWriter(output_rgb, cv2.VideoWriter_fourcc(*'mp4v'), int(video_fps), size)

    analyzers = []
    if trespassing_analyzer is not None:
        analyzers.extend(TrespassingAnalyzer.from_file(trespassing_analyzer))
    if illegal_entering_analyzer is not None:
        analyzers.extend(IllegalEnteringAnalyzer.from_file(illegal_entering_analyzer))

    processing_helper = ImageProcessingHelper(service, analyzers)
    processing_helper.service.update_models()
    processing_helper.set_current_model(model_name)

    output_playback_fps = 0
    output_playback_last_t = time.time()

    count = 0
    while True:
        success, img = capture.read()
        if not success:
            break
        input_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_rgb = processing_helper.detect(input_rgb)
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
