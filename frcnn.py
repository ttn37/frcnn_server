from keras import backend as K
from tensorflow.python.keras.backend import set_session
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
import numpy as np
import copy
import random
import cv2
import base64
import math
import pickle
import sys

from flask import Flask, jsonify, request
from frcnn_layers import *
from rpn_network import *

class Config:
	def __init__(self):
		self.verbose = True
		self.network = 'resnet'
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False
		self.anchor_box_scales = [64, 128, 256] 
		self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
		self.im_size = 600
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0
		self.num_rois = 4
		self.rpn_stride = 16

		self.balanced_classes = False
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		self.class_mapping = None
		self.model_path = None


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
global sess
global graph
sess = tf.Session(config=config)
set_session(sess)
graph = tf.get_default_graph()
sys.setrecursionlimit(40000)

from frcnn_layers import *
from rpn_network import *

def format_img_size(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def get_predict(img):
    bbox_threshold = 0.8
    X, ratio = format_img(img, C)
    st = time.time()
    X = np.transpose(X, (0, 2, 3, 1))
    global sess
    set_session(sess)
    global graph
    with graph.as_default():
        [Y1, Y2, F] = model_rpn.predict(X)
    R = rpn_to_roi(Y1, Y2, C, 'tf', overlap_thresh=0.7)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
        with graph.as_default():
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            #cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

            #textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append({"class":key,  "score":100*new_probs[jk], "xmin":real_x1, "xmax":real_x2, "ymin":real_y1, "ymax":real_y2})

            # (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            # textOrg = (real_x1, real_y1-0)

            # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            # cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            # cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    #print('Elapsed time = {}'.format(time.time() - st))
    return all_dets
    #cv2.imwrite('./results_imgs/test.png')

def readb64(encoded_data):
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

model_best_path = './model/model_frcnn_resnet_best.hdf5'
C = pickle.load( open("./model_resnet_config.pickle", "rb" ))
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

num_features = 1024
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)
shared_layers = nn_base(img_input, trainable=True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)
classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(model_best_path))
model_rpn.load_weights(model_best_path, by_name=True)
model_classifier.load_weights(model_best_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    ##recieve image
    data = request.get_json(force=True)
    image = data['photo']
    decoded_img = readb64(image)

    ##display shape and save (debug)
    #print(decoded_img.shape)
    #cv2.imwrite('./test.png',decoded_img)

    ##predict and return list of results
    results = get_predict(decoded_img)
    #print(results)

    ##response back in json format 
    print('PROCESS Elapsed time = {}'.format(time.time() - start_time))
    return jsonify({"results": results}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)