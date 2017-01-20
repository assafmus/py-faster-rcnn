# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:23:11 2016

@author: assaf
"""

import glob
import sys
sys.path.insert(0, '/home/assaf/py-faster-rcnn/tools')
import _init_paths
from fast_rcnn.test import im_detect
import caffe
import cv2
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
import numpy as np
from tqdm import tqdm
import time

prototxt = '/home/assaf/py-faster-rcnn/models/btsd/VGG16/faster_rcnn_end2end/test.prototxt'
caffemodel = '/home/assaf/py-faster-rcnn/output/faster_rcnn_end2end/btsd_2016_train/vgg16_faster_rcnn_iter_70000.caffemodel'
if (True):
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

lines = open('/home/assaf/py-faster-rcnn/data/btsd/ImageSets/Main/test.txt','r').readlines()
lines = [line.strip() for line in lines]
#lines = lines[:10]

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.GPU_ID = 0

cfg.TEST.SCALES = [1236]

total_time = 0
total_count = 0

with open('/home/assaf/btsd_faster_rcnn_results_1236_alt.txt','w') as out:
    for line in tqdm(lines):
        im_path = '/home/assaf/py-faster-rcnn/data/btsd/JPEGImages/' + line + '.jpg'
        im = cv2.imread(im_path)
        t = time.time()
        scores, boxes = im_detect(net, im)
        elapsed = time.time() - t
        
        total_time = total_time + elapsed
        total_count = total_count + 1

        # Visualize detections for each class
        CONF_THRESH = 0.1
        NMS_THRESH = 0.3
        for cls_ind in range(1,12):
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            keep = np.where(dets[:,4] > CONF_THRESH)
            dets = dets[keep]
            for det in dets:
                det = det.flatten()
                out.write("%s,%f,%f,%f,%f,%f,%d\n" % (line, det[0], det[1], det[2], det[3], det[4], cls_ind))
                
                
print "Average time: " + str(total_time/total_count)
print "Average speed: " + str(total_count/total_time) + " FPS"
