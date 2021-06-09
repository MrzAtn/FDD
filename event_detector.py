import cv2
import numpy as np
import tensorflow as tf
import glob
import os, signal, queue

from lanenetdet.config import global_config
from lanenetdet.lanenet_model import lanenet

from run_efficientdet import effdet

from module.functions import*
from module.object_Tracker import Object_Tracker
from module.line_Tracker import LineTracker
from module.event_Classifier import EventClassifier

class Event_Detector:

    def __init__(self):
        # setup Tracker objects
        self.ot = Object_Tracker()
        self.lt = LineTracker()
        self.ec = EventClassifier(object_Tracker=self.ot, line_Tracker=self.lt)
        self.gpath = "C:/FusionData/5A/ClassEvent_IHM"

    def detect_events(self, action_interface, stop_process_queue, videoFolderPath, flag_crossing, flag_acc, flag_cico, flag_cut):
        # Set sess configuration / init lanenet
        sess_config = tf.ConfigProto()
        CFG = global_config.cfg
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)
        lane_frame_width = 720  # lanenet input size
        lane_frame_height = 256  # lanenet input size

        with sess.as_default():
            input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, lane_frame_height, lane_frame_width, 3], name='input_tensor')
            net = lanenet.LaneNet(phase='test', net_flag='vgg')
            binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path= self.gpath + "/lanenetdet/model/tusimple_lanenet_vgg.ckpt")  # C:/Tools/Python37/Lib/lanenetdet/model/tusimple_lanenet_vgg.ckpt

        # init efficientdet
        efficient_det = effdet(self.gpath)
        vid_folders = glob.glob(videoFolderPath + "/*")
        for vid in vid_folders:
            vidname = vid.split("\\")[-1]
            print("Processing file : " + vidname)
            vidonlyname = vidname.split(".")[0]
            cap = cv2.VideoCapture(vid)
            self.fps = cap.get(cv2.CAP_PROP_FPS) # raw is 25 fps
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_width = 720  # after maskframe
            frame_height = 255  # after maskframe
            frame_count = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    rects = list()
                    frame_count += 1
                    frame = maskframe(frame)  # raw image is 720x576, output is 720x350
                    # efficientdet
                    if (frame_count-1) % 2 == 0 or frame_count == 1:
                        action_interface.affichage(frame_count, total_frame)
                        detections = efficient_det.detect_image(frame) # ymin, xmin, ymax, xmax frame[80:, 150:550]
                        for detect in detections:
                            if detect[6] in [3, 4, 6, 8]:  # car, motorcycle, bus, truck
                                rects.append(detect) # (?, ymin, xmin, ymax, xmax, %, class)
                        # Object Tracking 
                        objects = self.ot.update(rects, self.lt.lines)
                    if len(objects) > 0:
                        new_frame = efficient_det.draw_objects(frame, np.array(list(objects.values())), track_ids=list(objects.keys()))
                    else: 
                        new_frame = frame

                    # lanenet  
                    frame = cv2.resize(frame, (lane_frame_width, lane_frame_height), interpolation=cv2.INTER_LINEAR)
                    frame = frame / 127.5 - 1.0
                    with sess.as_default():
                        binary_seg_image, instance_seg_image = sess.run(
                            [binary_seg_ret, instance_seg_ret],
                            feed_dict={input_tensor: [frame]}
                        )
                    if binary_seg_image is not None:
                        # Compute lines for each step
                        output, right_lane, left_lane = self.lt.update(binary_seg_image[0], new_frame)
                        self.ec.update(vidonlyname, frame_count, flag_acc, flag_cico, flag_crossing, flag_cut)
                    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    try:
                        stop_process = stop_process_queue.get(timeout=1)
                        if stop_process:
                            sess.close()
                            action_interface.register_Event(self.ec.hist_event)
                            print("=== FIN de l'analyse vidéo ===")
                            return
                    except queue.Empty:
                        pass
                else:
                    break
            # The following frees up ressources and closes all windows
            cap.release()
        print("FIN DE LA LECTURE VIDEO")
        action_interface.register_Event(self.ec.hist_event)
        sess.close()