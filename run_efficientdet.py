import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from efficientdet import hparams_config
from efficientdet import inference
from efficientdet import utils


def maskframe(frame):  # mask image useless regions
    # Create the basic black image
    mask_left = np.zeros(frame.shape, dtype="uint8")
    mask_right = np.zeros(frame.shape, dtype="uint8")
    mask_top = np.zeros(frame.shape, dtype="uint8")
    mask_bot = np.zeros(frame.shape, dtype="uint8")
    # mask = np.zeros(frame.shape, dtype="uint8")

    # Draw a white, filled rectangle on the mask image
    cv2.rectangle(mask_left, (0, 45), (240, 240), (255, 255, 255), -1)
    cv2.rectangle(mask_right, (720, 45), (475, 240), (255, 255, 255), -1)
    cv2.rectangle(mask_top, (0, 0), (720, 45), (255, 255, 255), -1)
    cv2.rectangle(mask_bot, (0, 395), (720, 576), (255, 255, 255), -1)
    mask = cv2.bitwise_or(mask_left, mask_right)
    mask = cv2.bitwise_or(mask, mask_top)
    mask = cv2.bitwise_or(mask, mask_bot)
    mask = cv2.bitwise_not(mask)

    # Apply the mask and display the result
    maskedImg = cv2.bitwise_and(frame, mask)
    maskedImg = maskedImg[45:395]
    return maskedImg

class effdet:

  def __init__(self, gpath):
    det_nb = 2 # EfficientDet model verison
    model_name = f'efficientdet-d{det_nb}'

    # video to read if you want to use detect_video function in this file
    input_video = 'C:/Bertrandt/EffDet/000000005167_722.avi'
    output_video = f'C:/Bertrandt/EffDet/Output_efficientdet-d{det_nb}_temp.avi'
    output_video = None

    model_config = hparams_config.get_detection_config(model_name)
    # model_config.override(hparams)  # Add custom overrides
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    self.driver = inference.ServingDriver(
        model_name,
        f'{gpath}/efficientdet - models/efficientdet-d{det_nb}',
        batch_size=1,
        use_xla=False,
        model_params=model_config.as_dict())
    self.driver.load(f'{gpath}/efficientdet - models/efficientdet-d{det_nb}_export')

    self.config_dict = {'min_score_thresh': 0.4}

  def detect_image(self, frame):
    detections_bs = self.driver.serve_images([np.array(frame)])
    return detections_bs[0]

  def draw_objects(self, raw_frame, detections_bs, **kwargs):
    return self.driver.visualize(raw_frame, detections_bs, **self.config_dict, **kwargs)

  def detect_video(self):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print('Error opening input video: {}'.format(input_video))

    out_ptr = None
    if output_video:
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        frame_height = 395 - 45
        out_ptr = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(
          'm', 'p', '4', 'v'), 25, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = maskframe(frame)
        raw_frames = [np.array(frame)]
        if (frame_count-1) % 3 == 0 or frame_count == 1:
            detections_bs = driver.serve_images(raw_frames)
        new_frame = driver.visualize(raw_frames[0], detections_bs[0])

        if out_ptr:
            out_ptr.write(new_frame)
        else:
            cv2.imshow('Frame', new_frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 