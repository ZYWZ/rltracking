import gym
import torch
import os
from os import walk
from gym.spaces import Discrete, Tuple
from PIL import Image

from models.GAT import GAT
from util.constants import LayerType
from utils.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as T
import torch.nn.functional as F
import configparser
import motmetrics as mm
import glob
from collections import OrderedDict
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d

from proto import detection_results_pb2
from torch.distributions.categorical import Categorical


# INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"
from visualize_track_from_object import load_graph_data, load_graph_labels, visualize_graph, \
    extract_tracklet_features_from_object

INPUT_PATH_TRAIN = "datasets/MOT17/train"
INPUT_PATH_TEST = "datasets/MOT17/test"
# NO_INTERP = ['MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP', 'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP']
# NO_INTERP = ['MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP', 'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP']
NO_INTERP = []

MOVING_SEQUENCE = ['MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13', 'MOT17-06', 'MOT17-07', 'MOT17-12', 'MOT17-14']

MODEL_PATH = "models/checkpoints/gat_MOT17_ckpt_epoch_100.pth"

def name_to_layer_type(name):
    if name == LayerType.IMP1.name:
        return LayerType.IMP1
    elif name == LayerType.IMP2.name:
        return LayerType.IMP2
    elif name == LayerType.IMP3.name:
        return LayerType.IMP3
    else:
        raise Exception(f'Name {name} not supported.')

def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.

    # w = bbox[2]
    # h = bbox[3]
    # x = bbox[0] + w / 2.
    # y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    # if x[2] * x[3] < 0:
    #     x[2] = -x[2]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [x_c, y_c,
         (x_c + w), (y_c + h)]
    return b


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _matcher(src, tgt, type='block'):
    src_boxes = src['bbox']
    tgt_boxes = tgt['bbox']

    src_feat = src['feature']
    tgt_feat = tgt['feature']

    cost_bbox = torch.cdist(src_boxes, tgt_boxes, p=1)
    # cost_giou = -generalized_box_iou(src_boxes, tgt_boxes)
    cost_feat = _cosine_distance(src_feat, tgt_feat)

    cost_feat = torch.Tensor(cost_feat)

    if cost_bbox.shape[1] == 1:
        cost_bbox = cost_bbox.permute(1, 0)
        cost_bbox = F.normalize(cost_bbox)
        cost_bbox = cost_bbox.permute(1, 0)
    else:
        cost_bbox = F.normalize(cost_bbox)

    # if type == 'block':
    #     # C = 1 * cost_bbox + 0.5 * cost_feat
    #     C = 1 * cost_bbox
    # else:
    #     C = 1 * cost_bbox
    C = cost_feat + cost_bbox

    C = C.cpu()

    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind

# match the detection to env objects
# impact factor:
#       detection: bbox position, ReID feature
#       env obj: time since update
def _matcher2(src, tgt):
    return 0


class GymRltrackingEnv(gym.Env):
    def __init__(self):
        self.source = None
        self.train_mode = True
        self.extract_feature = False
        self.obj_count = 0
        self.step_count = 0
        self.start_frame = -1
        self.frame = -1
        self.img_w = -1
        self.img_h = -1
        action_list = []
        self.input_path = ""

        for i in range(self.obj_count):
            action = Discrete(4)
            action_list.append(action)

        self.action_space = Tuple(action_list)
        self.objects = []
        self.tracks = []
        self.detection_memory = None
        self.detection_memory_old = None

        self.extractor = None
        self.act_reward = 0
        self.max_track_number = 200
        self.id = uuid.uuid4()
        self.train_length = 500

        self.mota = []
        self.idf1 = []
        self.IDs = 0

        self.log = []
        self.gt_for_reward = []
        self.track_for_reward = []
        self.no_detection = False
        self.seq_name = ""

        self.processed_objects = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_lpc_results(self, sequence):
        input_path = "datasets/MOT17/LPC_detections/MOT17/results_reid_with_traindata/detection"
        _, _, files = next(walk(input_path))

        for filename in files:
            if filename[:-3] == sequence:

                file = os.path.join(input_path, filename)
                detections = detection_results_pb2.Detections()
                f = open(file, "rb")
                detections.ParseFromString(f.read())

                track_frame_res = {}
                frame_index, bounding_box = [], []
                for detection in detections.tracked_detections:
                    frame_idx = detection.frame_index
                    box_x = detection.box_x
                    box_y = detection.box_y
                    box_width = detection.box_width
                    box_height = detection.box_height
                    bbx = [box_x, box_y, box_width, box_height]
                    feature = list(detection.features.features[0].feats)

                    frame_index.append(frame_idx)
                    bounding_box.append(bbx)
                    track_frame_res.setdefault(frame_idx, []).append([bbx, feature])

                return track_frame_res
        return None

    def load_detection_result(self):
        input_path = "datasets/MOT17/Tracktor_v2_MOT17"
        _, _, directories = next(walk(input_path))

        results = []
        for directory in directories:
            file = os.path.join(input_path, directory)
            with open(file) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            results.append(content)
        dict_names = []
        for directory in directories:
            dict_names.append(directory[:-4])

        output = dict(zip(dict_names, results))

        return output

    # def load_detection_result(self):
    #     _, directories, _ = next(walk(self.input_path))
    #
    #     PATH = self.input_path
    #
    #     results = []
    #     for directory in directories:
    #         # file = os.path.join(PATH, directory, "det", directory+".txt")
    #         file = os.path.join(PATH, directory, "det", "det.txt")
    #         with open(file) as f:
    #             content = f.readlines()
    #         content = [x.strip() for x in content]
    #         results.append(content)
    #
    #     output = dict(zip(directories, results))
    #
    #     return output

    def load_gt_result(self):
        _, directories, _ = next(walk(self.input_path))
        results = []
        for directory in directories:
            file = os.path.join(self.input_path, directory, "gt", "gt.txt")
            with open(file) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            results.append(content)

        output = dict(zip(directories, results))

        return output

    def inference(self):
        self.train_mode = False

    def set_seq_name(self, name):
        self.seq_name = name

    def set_extractor(self, extractor):
        self.extractor = extractor

    def init_source(self, source, train_or_test):
        if train_or_test == "train":
            self.train_mode = True
            self.input_path = INPUT_PATH_TRAIN
        else:
            self.train_mode = False
            self.input_path = INPUT_PATH_TEST
        self.source = source
        if source == "ADL-Rundle-1" or source == "ADL-Rundle-3" or source == "ADL-Rundle-6" or source == "ADL-Rundle-8":
            self.img_w = 1920
            self.img_h = 1080
        if source == "AVG-TownCentre":
            self.img_w = 1920
            self.img_h = 1080
        if source == "ETH-Crossing" or source == "ETH-Jelmoli" or source == "ETH-Linthescher" or source == "ETH-Bahnhof" or source == "ETH-Pedcross2":
            self.img_w = 640
            self.img_h = 480
        if source == "KITTI-13":
            self.img_w = 1242
            self.img_h = 375
        if source == "KITTI-16":
            self.img_w = 1224
            self.img_h = 370
        if source == "KITTI-17":
            self.img_w = 1224
            self.img_h = 370
        if source == "KITTI-19":
            self.img_w = 1238
            self.img_h = 374
        if source == "PETS09-S2L1" or source == "PETS09-S2L2":
            self.img_w = 720
            self.img_h = 576
        if source == "TUD-Crossing" or source == "TUD-Campus" or source == "TUD-Stadtmitte":
            self.img_w = 640
            self.img_h = 480
        if source == "Venice-1" or source == "Venice-2":
            self.img_w = 1920
            self.img_h = 1080
        seqfile = os.path.join(self.input_path, source, 'seqinfo.ini')
        if os.path.isfile(seqfile):
            config = configparser.ConfigParser()
            config.read(seqfile)
            self.img_w = int(config.get('Sequence', 'imWidth'))
            self.img_h = int(config.get('Sequence', 'imHeight'))
            self.seq_name = config.get('Sequence', 'name')

        if self.train_mode:
            # self.det_result = self.load_detection_result()[self.source]
            self.det_result = self.load_lpc_results(self.seq_name)
            self.gt_result = self.load_gt_result()[self.source]
        else:
            self.det_result = self.load_lpc_results(self.seq_name)
            # self.det_result = self.load_detection_result()[self.source]
            # print("Not implemented inference mode for env.init")
            # raise

    # initiate the objects in env, according to the first frame detection
    def initiate_env(self, start_frame):
        self.tracks = []
        self.objects = []
        self.step_count = 0
        self.start_frame = start_frame
        self.frame = start_frame
        # self.detection_memory = self.get_detection(self.frame)
        # det, feat = self.get_detection(self.frame)
        # obj_count = len(det)
        # feature = feat
        # for i in range(obj_count):
        #     new_obj = RLObject()
        #     new_obj.update(det[i], feature[i])
        #     self.objects.append(new_obj)
        #     track = RLTrack()
        #     track.init_track(self.frame, new_obj)
        #     self.tracks.append(track)

        # return self.gen_obs()

    # def update_env_objects(self, src_list, tgt_list, type='block'):
    #     for obj in tgt_list:
    #         obj.set_block(True)
    #     update_list = [box[0] for box in src_list]
    #     obj_list = [obj.get_location() for obj in tgt_list]
    #
    #     src_feat = [box[1] for box in src_list]
    #     tgt_feat = [obj.get_feature() for obj in tgt_list]
    #
    #     src = torch.Tensor(update_list)
    #     tgt = torch.Tensor(obj_list)
    #
    #     # src_feat = torch.Tensor(update_list_feature)
    #     # tgt_feat = torch.Tensor(obj_list_feature)
    #
    #     src = {'bbox': src,
    #            'feature': src_feat}
    #     tgt = {'bbox': tgt,
    #            'feature': tgt_feat}
    #
    #     if src['bbox'].shape[0] != 0 and tgt['bbox'].shape[0] != 0:
    #         ind_row, ind_col = _matcher(tgt, src, type)
    #         for i, j in zip(ind_row, ind_col):
    #             obj = tgt_list[i]
    #             obj.update(src_list[j][0], src_list[j][1], self.frame)
    #             obj.set_block(False)

    # def update(self, action):
    #     action = action.detach().tolist()[0]
    #     boxes, feats = self.get_detection_memory()
    #
    #     action = action[:len(boxes)]
    #
    #     assert len(action) == len(boxes), "action list length not equal to detection numbers!"
    #     assert len(action) == len(feats), "action list length not equal to detection feature numbers!"
    #
    #     add_list = []
    #     update_current_list = []
    #     update_old_list = []
    #     for act, box, feat in zip(action, boxes, feats):
    #         # new object
    #         if act == 0:
    #             # print("new object")
    #             add_list.append((box, feat))
    #         elif act == 1:
    #             # print("update current")
    #             update_current_list.append((box, feat))
    #         elif act == 2:
    #             # print("update old")
    #             update_old_list.append((box, feat))
    #
    #     reveal_objects = []
    #     blocked_objects = []
    #     for obj in self.objects:
    #         if obj.is_blocked():
    #             blocked_objects.append(obj)
    #         else:
    #             reveal_objects.append(obj)
    #
    #     # update current object
    #     self.update_env_objects(update_current_list, reveal_objects, type='reveal')
    #
    #     # update old object, turn them into current object
    #     self.update_env_objects(update_old_list, blocked_objects, type='block')
    #
    #     # add new object
    #     for obj, feat in add_list:
    #         new_obj = RLObject()
    #         new_obj.update(obj, feat, self.frame)
    #         self.objects.append(new_obj)

    def calculate_dist(self, feat_dist, bbox_dist, time_dist):
        feat_dist = feat_dist[0][0]
        bbox_dist = bbox_dist.item()
        if feat_dist < 0.1:
            return feat_dist
        elif 0.1 <= feat_dist <= 0.6:
            if time_dist < 30:
                if bbox_dist <= 100:
                    return 0.1
                elif 100 < bbox_dist <= 500:
                    return feat_dist
                elif bbox_dist > 500:
                    return 1
            else:
                return feat_dist
        elif feat_dist > 0.6:
            return 1
        return feat_dist

    def update_env_objects2(self, src_list, tgt_list, type='block'):
        for obj in tgt_list:
            obj.set_block(True)
        update_list = [box[0] for box in src_list]
        obj_list = [obj.get_location() for obj in tgt_list]

        src_feat = [box[1] for box in src_list]
        tgt_feat = [obj.get_feature() for obj in tgt_list]

        src = torch.Tensor(update_list)
        tgt = torch.Tensor(obj_list)

        # src_feat = torch.Tensor(update_list_feature)
        # tgt_feat = torch.Tensor(obj_list_feature)

        src = {'bbox': src,
               'feature': src_feat}
        tgt = {'bbox': tgt,
               'feature': tgt_feat}

        # print("detection number: ", src['bbox'].shape[0])
        # print("env obj number", tgt['bbox'].shape[0])
        # print("self.no_detection ", self.no_detection)

        # src is detection, tgt is env objects
        if src['bbox'].shape[0] != 0 and tgt['bbox'].shape[0] != 0 and not self.no_detection:
            ind_row, ind_col = _matcher(src, tgt, type)
            # print("matching result: ", ind_row, ind_col)
            # add unmatched detection as new object
            for i, det in enumerate(src['bbox']):
                if i not in ind_row:
                    new_obj = RLObject(src_list[i][0])
                    new_obj.update(src_list[i][0], src_list[i][1], self.frame)
                    self.objects.append(new_obj)
            # match env obj with detection, if distance > 300, set to new obj
            for i, j in zip(ind_row, ind_col):
                bbox = src_list[i][0]
                feat = src_list[i][1]

                env_bbox = tgt_list[j].get_location()
                env_feat = tgt_list[j].get_feature()

                src_box = torch.Tensor([bbox])
                tgt_box = torch.Tensor([env_bbox])

                bbox_dist = torch.cdist(src_box, tgt_box, p=1)
                feat_dist = _cosine_distance([feat], [env_feat])
                time_dist = tgt_list[j].get_time_since_update()

                # dist = self.calculate_dist(feat_dist, bbox_dist, time_dist)
                dist = feat_dist
                # if bbox_dist < 500 and feat_dist < 0.5:
                if dist < 0.6:
                    obj = tgt_list[j]
                    obj.update(src_list[i][0], src_list[i][1], self.frame)
                    obj.add_feat_dist(feat_dist[0][0])
                    obj.set_block(False)
                else:
                    new_obj = RLObject(src_list[i][0])
                    new_obj.update(src_list[i][0], src_list[i][1], self.frame)
                    self.objects.append(new_obj)
        # predict movement of non-updated env object, if time_since_update > 8, leave the object still
        for obj in tgt_list:
            if obj.is_blocked():
                obj.not_updated()
                time_threshold = 200
                if self.seq_name[:8] in MOVING_SEQUENCE:
                    time_threshold = 100
                if obj.get_time_since_update() >= time_threshold:
                    obj.deactivate()
                # if obj.get_time_since_update() <= 8:
                #     obj.kalman_predict(self.frame)

    def update_no_action(self):
        try:
            detections = self.det_result[self.frame]
            update_list = []
            for det in detections:
                box, feat = det
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                update_list.append((box, feat))
            # for box, feat in zip(boxes, feats):
            #     update_list.append((box, feat))

            env_objs = []
            for obj in self.objects:
                if obj.is_active():
                    env_objs.append(obj)

            if len(env_objs) == 0:
                for obj, feat in update_list:
                    new_obj = RLObject(obj)
                    new_obj.update(obj, feat, self.frame)
                    self.objects.append(new_obj)

            # update old object, turn them into current object
            self.update_env_objects2(update_list, env_objs, type='reveal')

        except KeyError:
            print("No detection found in this frame")

    def resize_roi(self, roi):
        w = self.img_w
        h = self.img_h
        resize_roi = roi.clone()
        normalized_roi = roi.clone()

        resize_roi[:, 0] = roi[:, 0] * 1066 / w
        resize_roi[:, 2] = roi[:, 2] * 1066 / w
        resize_roi[:, 1] = roi[:, 1] * 800 / h
        resize_roi[:, 3] = roi[:, 3] * 800 / h

        normalized_roi[:, 0] = roi[:, 0] / w
        normalized_roi[:, 2] = roi[:, 2] / w
        normalized_roi[:, 1] = roi[:, 1] / h
        normalized_roi[:, 3] = roi[:, 3] / h

        return normalized_roi, resize_roi

    # def get_detection(self, frame):
    #     def get_index(frame):
    #         result = ""
    #         digit = len(str(frame))
    #         max = 6
    #         for i in range(max - digit):
    #             result += "0"
    #         result += str(frame)
    #         result += ".jpg"
    #
    #         return result
    #
    #     boxes = []
    #     feat = []
    #     result = self.det_result
    #
    #     for line in result:
    #         line = line.split(',')
    #         if int(line[0]) == frame:
    #             temp = []
    #             for i in line[2:6]:
    #                 if float(i) < 0:
    #                     i = 0
    #                 temp.append(float(i))
    #             # transform detection from (x, y, w, h) to (x1, y1, x2, y2)
    #             temp = box_cxcywh_to_xyxy(temp)
    #             boxes.append(temp)
    #     if len(boxes) > 0:
    #         self.no_detection = False
    #         if self.extract_feature:
    #             img_index = get_index(frame)
    #             img_path = os.path.join(self.input_path, self.source, 'img1', img_index)
    #
    #             img = Image.open(img_path)
    #             transform = T.Compose([
    #                 T.Resize(800),
    #                 T.ToTensor(),
    #                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ])
    #             img = transform(img).unsqueeze(0).cuda()
    #             pad = (1, 0)
    #             box_tensor = torch.Tensor(boxes)
    #             _, box_tensor = self.resize_roi(box_tensor)
    #             box_tensor = F.pad(box_tensor, pad, 'constant', 0).cuda()
    #             feat = self.extractor(box_tensor, img)
    #
    #             copy = feat.detach().cpu().numpy().tolist()
    #             del feat
    #             torch.cuda.empty_cache()
    #         else:
    #             copy = np.empty([len(boxes), 512])
    #     else:
    #         copy = []
    #         boxes.append(np.zeros(4))
    #         copy.append(np.zeros(512))
    #         self.no_detection = True
    #
    #     return boxes, copy

    def get_detection_memory(self):
        return self.detection_memory

    def append_log(self, action):
        self.log.append(str(action))
        self.log.append(str(self.mota))
        self.log.append(str(self.idf1))
        self.log.append("-------------------------------------------")

    def write_log(self):
        mota_log_file = 'motaLog.txt'
        with open(mota_log_file, 'a') as f:
            for line in self.log:
                f.write(line)
                f.write('\n')

    # update objects in env, according to action
    def step(self, action):
        obs = {}

        # self.act_reward = self.action_reward(action)
        # boxes, feat = self.get_detection_memory()
        # self.update_objects(action, boxes, feat)
        # self.update(action)
        self.update_no_action()
        # reward = self.reward()
        self.step_count += 1
        self.frame += 1
        # self.detection_memory_old = self.detection_memory
        # self.detection_memory_old = self.detection_memory
        # self.detection_memory = self.get_detection(self.frame)

        # print("Env obj count: ", len(self.objects))
        # cc = 0
        # for obj in self.objects:
        #     if obj.is_active():
        #         cc += 1
        # print("active object: ", cc)
        # print("Detection count: ", len(self.get_detection_memory()[0]))

        reward = 0
        end = False
        # if self.train_mode:
        #     reward = self.reward()
        #
        #     end = bool(len(self.objects) <= 0
        #                # or np.mean(self.mota) < 0.4
        #                or np.mean(self.idf1) < 0.7
        #                or self.step_count >= self.train_length)
        #
        #     if not end:
        #         self.append_log(action)
        #         obs = self.gen_obs()
        #         # reward = np.mean(self.mota) + np.mean(self.idf1)
        #         reward = 7 - self.IDs
        #     elif self.step_count >= self.train_length:
        #         self.write_log()
        #         obs = self.gen_obs()
        #         # reward = np.mean(self.mota) + np.mean(self.idf1)
        #         reward = 7 - self.IDs
        #     else:
        #         self.write_log()
        #         reward = 0
        #         # os.remove('gym_rltracking/envs/rltrack/gt/' + str(self.id) + '.txt')
        #         # os.remove('gym_rltracking/envs/rltrack/' + str(self.id) + '.txt')
        # else:
        #     obs = self.gen_obs()

        return obs, reward, end, {}

    def output_result(self):
        node_features, node_labels, node_topologies = self.generate_track_from_objects()
        self.write_to_file(self.track_for_reward)
        return node_features, node_labels, node_topologies

    def output_gt(self):
        self.gt_result = self.load_gt_result()[self.source]
        gt = self.generate_gt_for_reward()
        self.write_gt_to_file(gt)

    def reset(self):
        self.id = uuid.uuid4()
        self.step_count = 0
        self.frame = 0
        self.objects = []
        self.tracks = []
        self.mota = []
        self.idf1 = []
        self.log = []

    def compare_dataframes(self, gts, ts):
        """Builds accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)
            else:
                print('No ground truth for %s, skipping.', k)

        return accs, names

    def construct_det_dataframe(self, list):
        list = np.transpose(list)
        index = list[:2]
        index = pd.MultiIndex.from_arrays(index, names=('FrameId', 'Id'))
        col_dict = {'FrameId': np.array(list[0]),
                    'Id': np.array(list[1]),
                    'X': np.array(list[2]),
                    'Y': np.array(list[3]),
                    'Width': np.array(list[4]),
                    'Height': np.array(list[5]),
                    'Confidence': np.array(list[6]),
                    'ClassId': np.array(list[7]),
                    'Visibility': np.array(list[8]),
                    'unused': np.array(list[9])}

        columns = ['X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused']
        df = DataFrame(col_dict, columns=columns, index=index)
        # Account for matlab convention.
        df[['X', 'Y']] -= (1, 1)
        del df['unused']
        min_confidence = 1
        return df[df['Confidence'] >= min_confidence]

    def construct_dataframe(self, list):
        list = np.transpose(list)
        index = list[:2]
        index = pd.MultiIndex.from_arrays(index, names=('FrameId', 'Id'))
        col_dict = {'FrameId': np.array(list[0]),
                    'Id': np.array(list[1]),
                    'X': np.array(list[2]),
                    'Y': np.array(list[3]),
                    'Width': np.array(list[4]),
                    'Height': np.array(list[5]),
                    'Confidence': np.array(list[6]),
                    'ClassId': np.array(list[7]),
                    'Visibility': np.array(list[8])}

        columns = ['X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']
        df = DataFrame(col_dict, columns=columns, index=index)
        # Account for matlab convention.
        df[['X', 'Y']] -= (1, 1)

        # Removed trailing column
        # del df['unused']

        # Remove all rows without sufficient confidence
        min_confidence = 1
        df = df[df['ClassId'] == 1]
        df = df[df['Visibility'] >= 0.5]
        return df[df['Confidence'] >= min_confidence]

    def calculate_mota(self):
        # gtfiles = glob.glob(os.path.join('gym_rltracking/envs/rltrack/gt', 'gt.txt'))
        # tsfiles = [f for f in glob.glob(os.path.join('gym_rltracking/envs/rltrack', '*.txt')) if
        #            not os.path.basename(f).startswith('eval')]
        # gtfile = 'gym_rltracking/envs/rltrack/gt/' + str(self.id) + '.txt'
        # tsfile = 'gym_rltracking/envs/rltrack/' + str(self.id) + '.txt'
        # dataframe = mm.io.loadtxt(gtfile, fmt='mot15-2D', min_confidence=1)
        if len(self.track_for_reward) > 0:
            gt_df = self.construct_dataframe(self.gt_for_reward)
            track_df = self.construct_det_dataframe(self.track_for_reward)
            gt = OrderedDict([(str(self.id), gt_df)])
            ts = OrderedDict([(str(self.id), track_df)])

            mh = mm.metrics.create()
            accs, names = self.compare_dataframes(gt, ts)

            metrics = list(mm.metrics.motchallenge_metrics)
            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
            # print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
            # print(summary.values[0][13])
        else:
            return None, False

        return summary.values[0], True

    def generate_gt_for_reward(self):
        result = []
        for gt in self.gt_result:
            line = gt.split(',')
            if self.start_frame <= int(line[0]) < self.frame:
                result.append(line)
        output = []
        for res in result:
            res = [float(x) for x in res]
            output.append(res)
        self.gt_for_reward = output
        return output
        # self.write_gt_to_file(output)

    # def generate_track_result(self):
    #     start_idx = 2001
    #     result = []
    #     for track in self.tracks:
    #         start_frame = track.start_frame
    #         for t in track.track:
    #             line = [start_frame, start_idx, t[0], t[1], round(t[2] - t[0], 2), round(t[3] - t[1], 2), 1, -1, -1, -1]
    #             result.append(line)
    #             start_frame += 1
    #
    #         start_idx += 1
    #     self.write_to_file(result)

    def interpolate1d(self, t1, t2, loc1, loc2):
        X = [t1, t2]
        Y = [loc1, loc2]
        inter_x = list(range(t1, t2 - 1))
        interp = interp1d(X, Y)
        return interp(inter_x)

    def interpolate_track(self, obj):
        gap_ind = []
        l = obj.get_frames()
        if not sorted(l) == list(range(min(l), max(l) + 1)):
            for ind, frame in enumerate(obj.get_frames()):
                if ind == 0:
                    continue
                last_frame = obj.get_frames()[ind - 1]
                if frame != last_frame + 1:
                    gap_ind.append(ind - 1)

        for ind in gap_ind:
            t_1 = obj.get_frames()[ind]
            t_2 = obj.get_frames()[ind + 1]
            loc_1 = obj.get_history()[ind]
            loc_2 = obj.get_history()[ind + 1]

            interp_x1 = self.interpolate1d(t_1, t_2, loc_1[0], loc_2[0])
            interp_x2 = self.interpolate1d(t_1, t_2, loc_1[2], loc_2[2])
            interp_y1 = self.interpolate1d(t_1, t_2, loc_1[1], loc_2[1])
            interp_y2 = self.interpolate1d(t_1, t_2, loc_1[3], loc_2[3])

            for i in range(len(interp_x1)):
                timestamp = t_1 + i + 1
                coord = [interp_x1[i], interp_y1[i], interp_x2[i], interp_y2[i]]
                obj.add_interp(timestamp, coord)

        return obj

    def get_average_feat(self, obj_list):
        feat_values = []
        for obj in obj_list:
            feat_values.append(obj.get_feature().tolist())

        return feat_values

    def reconcat_object(self, obj1, obj2):
        for frame, loc, feature in zip(obj2.get_frames(), obj2.get_history(), obj2.get_features()):
            obj1.update(loc, feature, frame)
        return obj1

    def reconcat_split_objects(self, objects):
        if len(objects) < 3:
            return objects

        concated_index = []
        for i, object in enumerate(objects):

            if i == 0 or i == 1:
                continue
            source_obj = [object]
            target_obj_list = objects[:i-1]

            source_feat = self.get_average_feat(source_obj)
            target_feats = self.get_average_feat(target_obj_list)

            for j, target_feat in enumerate(target_feats):
                feat_dist = _cosine_distance(source_feat, [target_feat])[0][0]
                if feat_dist < 0.3:
                    target_obj_list[j] = self.reconcat_object(target_obj_list[j], source_obj[0])
                    concated_index.append(i)
        processed = []
        for i, obj in enumerate(objects):
            if i not in concated_index:
                processed.append(obj)

        return processed

    def split_object_2(self, obj, labels):
        if labels.count(1) == 0 or labels.count(0) == 0:
            return [obj]

        objects = []
        new_obj = RLObject([0, 0, 0, 0])
        new_obj.update(obj.get_history()[0], obj.get_features()[0], obj.get_frames()[0])
        objects.append(new_obj)
        for i, (label, frame, loc, feature) in enumerate(zip(labels[1:], obj.get_frames()[1:], obj.get_history()[1:], obj.get_features()[1:])):
            change_flag = False

            cur_label = labels[i + 1]
            prev_label = labels[i]

            if cur_label != prev_label:
                change_flag = True

            if not change_flag:
                obj = objects[-1]
                obj.update(loc, feature, frame)
            else:
                obj = RLObject([0, 0, 0, 0])
                obj.update(loc, feature, frame)
                objects.append(obj)

        # objects = self.reconcat_split_objects(objects)

        for obj in objects[1:]:
            obj.set_is_separated(True)

        return objects

    def split_object(self, obj, labels):
        pure_object = RLObject([0, 0, 0, 0])
        impure_object = RLObject([0, 0, 0, 0])

        if labels.count(1) == 0 or labels.count(0) == 0:
            return obj, None

        for label, frame, loc, feature in zip(labels, obj.get_frames(), obj.get_history(), obj.get_features()):
            if label == 0:
                pure_object.update(loc, feature, frame)
            else:
                impure_object.update(loc, feature, frame)

        pure_object.set_is_separated(True)
        return pure_object, impure_object


    def keep_separate_track(self, gat, objects):
        processed_features, node_topologies = extract_tracklet_features_from_object(objects)
        model_outputs = []
        with torch.no_grad():
            for in_nodes_features, edge_index in zip(processed_features, node_topologies):
                if in_nodes_features.shape[0] == 1:
                    output = [0]
                    output = torch.Tensor(output).to(torch.int64).to(self.device)
                    model_outputs.append(output)
                    continue
                graph_data = (in_nodes_features, edge_index)
                outputs = gat(graph_data)[0].softmax(-1)
                predicted = torch.argmax(outputs, dim=-1)
                model_outputs.append(predicted)

        objs_need_process = []
        for i, labels in enumerate(model_outputs):
            pure_object, impure_object = self.split_object(objects[i], labels.tolist())
            self.processed_objects.append(pure_object)
            if impure_object is not None:
                objs_need_process.append(impure_object)

        if len(objs_need_process) != 0:
            self.keep_separate_track(gat, objs_need_process)


    def separate_track(self, objects):
        model_state = torch.load(MODEL_PATH)
        gat = GAT(
            num_of_layers=model_state['num_of_layers'],
            num_heads_per_layer=model_state['num_heads_per_layer'],
            num_features_per_layer=model_state['num_features_per_layer'],
            add_skip_connection=model_state['add_skip_connection'],
            bias=model_state['bias'],
            dropout=model_state['dropout'],
            layer_type=name_to_layer_type(model_state['layer_type']),
            log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
        ).to(self.device)
        gat.load_state_dict(model_state["state_dict"], strict=True)
        gat.eval()

        processed_features, node_topologies = extract_tracklet_features_from_object(objects)

        model_outputs = []
        with torch.no_grad():
            for in_nodes_features, edge_index in zip(processed_features, node_topologies):
                if in_nodes_features.shape[0] == 1:
                    output = [0]
                    output = torch.Tensor(output).to(torch.int64).to(self.device)
                    model_outputs.append(output)
                    continue
                graph_data = (in_nodes_features, edge_index)
                outputs = gat(graph_data)[0].softmax(-1)
                _, predicted = torch.max(outputs, 1)
                model_outputs.append(predicted)
                if predicted.tolist().count(1) != 0:
                    print(predicted)

        objs_need_process = []

        for i, labels in enumerate(model_outputs):
            splited_objects = self.split_object_2(objects[i], labels.tolist())
            for obj in splited_objects:
                self.processed_objects.append(obj)


        #     self.processed_objects.append(pure_object)
        #     if impure_object is not None:
        #         objs_need_process.append(impure_object)
        #
        # if len(objs_need_process) != 0:
        #     self.keep_separate_track(gat, objs_need_process)


        return model_outputs


    def generate_track_from_objects(self):
        idx = 2001
        result = []
        node_features, node_labels, node_topologies = [], [], []
        if self.train_mode:
            w_h = self.img_w, self.img_h
            gts = self.generate_gt_for_reward()
            node_features, node_labels, node_topologies = load_graph_data(self.objects, gts, w_h)

        self.separate_track(self.objects)
        # visualize_graph(node_features, node_labels, node_topologies)
        print(len(self.objects))
        print(len(self.processed_objects))
        for obj in self.processed_objects:
            # print(obj.get_history())
            # print(obj.get_frames())
            obj = self.interpolate_track(obj)
            for frame, loc in zip(obj.get_frames(), obj.get_history()):
                if obj.get_is_separated():
                    flag = 1
                else:
                    flag = 0
                # if is_detection:
                line = [frame, idx, loc[0], loc[1], round(loc[2] - loc[0], 2), round(loc[3] - loc[1], 2), -1, -1, -1, flag]
                # line = [frame, idx, loc[0], loc[1], loc[2], loc[3], 1, -1, -1, -1]
                result.append(line)
            if len(obj.get_interps()[0]) != 0 and self.seq_name not in NO_INTERP:
                timestamps = obj.get_interps()[0]
                coords = obj.get_interps()[1]
                for timestamp, coord in zip(timestamps, coords):
                    line_interp = [timestamp, idx, coord[0], coord[1], round(coord[2] - coord[0], 2),
                                   round(coord[3] - coord[1], 2), -1, -1, -1, -1]
                    result.append(line_interp)
            idx += 1
        self.track_for_reward = result
        # self.write_to_file(result)
        return node_features, node_labels, node_topologies

    def write_gt_to_file(self, res):
        print("Storing gt result to file gym_rltracking/envs/rltrack/gt/" + str(self.seq_name) + ".txt")
        with open('gym_rltracking/envs/rltrack/gt/' + str(self.seq_name) + '.txt', 'w') as f:
            for item in res:
                f.write(','.join(map(repr, item)))
                f.write('\n')

    def write_to_file(self, res):
        print("Storing tracking result to file gym_rltracking/envs/rltrack/seq_result/" + str(self.seq_name) + ".txt")
        with open('gym_rltracking/envs/rltrack/seq_result/' + str(self.seq_name) + '.txt', 'w') as f:
            for item in res:
                f.write(','.join(map(repr, item)))
                f.write('\n')

    def get_frame_gt(self, frame):
        result = 0
        for gt in self.gt_result:
            line = gt.split(',')
            if frame == int(line[0]):
                result += 1
        return result

    def compare_objects(self, gt_number):
        env_count = len(self.objects)
        return -abs(gt_number - env_count)
        # if det_count == env_count:
        #     return 1
        # else:
        #     return -1

    def action_reward(self, action):
        action = action.detach().tolist()[0]
        boxes, _ = self.get_detection_memory()
        action = action[:len(boxes)]
        block_act = 0
        reveal_act = 0
        for act in action:
            if act == 1:
                reveal_act += 1
            elif act == 2:
                block_act += 1

        block_obj = 0
        reveal_obj = 0
        for obj in self.objects:
            if obj.is_blocked():
                block_obj += 1
            else:
                reveal_obj += 1

        return -(abs(reveal_act - reveal_obj) + abs(block_act - block_obj))

    def reward(self):
        weight = [1, 1, 1, 1]
        # if self.step_count % 10 == 0:
        self.generate_gt_for_reward()
        self.generate_track_from_objects()
        summary, flag = self.calculate_mota()
        if flag:
            self.idf1.append(summary[0])
            self.mota.append(summary[13])
            self.IDs = summary[11]
            motp = summary[14]
        else:
            self.idf1.append(0)
            self.mota.append(0)

        # if self.mota >= 0.5:
        #     reward_mota = 1
        # elif 0.3 < self.mota < 0.5:
        #     reward_mota = 0
        # else:
        #     reward_mota = -1
        #
        # if self.idf1 >= 0.5:
        #     reward_idf1 = 1
        # elif 0.3 < self.idf1 < 0.5:
        #     reward_idf1 = 0
        # else:
        #     reward_idf1 = -1

        # boxes, _ = self.get_detection_memory()
        # gt_number = self.get_frame_gt(self.frame)
        # reward_obj_count = self.compare_objects(gt_number)

        # reward = weight[0] * self.mota + weight[1] * self.idf1 + weight[2] * reward_obj_count + weight[3] * self.act_reward

        return 0

    def gen_obs(self):
        locations = []
        feats = []
        # for obj in self.objects:
        #     locations.append(obj.get_location())
        #     feats.append(obj.get_feature())
        # locations = torch.Tensor(locations).cuda()
        # feats = torch.Tensor(feats).cuda()

        # frame = self.frame
        det, det_feat = self.get_detection_memory()
        self.detection_memory = (det, det_feat)

        # locations = self.resize_roi(locations)
        # pad_number = self.max_track_number - len(det)
        det = torch.Tensor(det)
        normed_det, det = self.resize_roi(det)
        det_feat = torch.Tensor(det_feat)
        # det_pad = torch.zeros(pad_number, 4)
        # feat_pad = torch.zeros(pad_number, 512)
        # det = torch.cat((det, det_pad), dim=0).cuda()
        # det_feat = torch.cat((det_feat, feat_pad), dim=0).cuda()

        # if self.detection_memory_old is not None:
        #     det_old, det_feat_old = self.detection_memory_old
        # else:
        #     det_old = [np.zeros(4)]
        #     det_feat_old = [np.zeros(512)]
        # pad_number_old = self.max_track_number - len(det_old)
        # det_old = torch.Tensor(det_old)
        # normed_det_old, det_old = self.resize_roi(det_old)
        # det_feat_old = torch.Tensor(det_feat_old)
        # det_pad_old = torch.zeros(pad_number_old, 4)
        # feat_pad_old = torch.zeros(pad_number_old, 512)
        # det_old = torch.cat((det_old, det_pad_old), dim=0).cuda()
        # det_feat_old = torch.cat((det_feat_old, feat_pad_old), dim=0).cuda()

        obs = {
            'det': det,
            'det_feat': det_feat,
        }
        return obs

    def render(self, mode='human'):
        print(self.frame)
        if mode == 'printTrack':
            for track in self.tracks:
                print(track.track)
            print("-------------------")


class RLObject:
    def __init__(self, location):
        self.location = location
        self.feature_history = []
        self.feature = None
        self.block = False
        self.frames = []
        self.history = []
        self.history_is_detection = []
        self.interps = []
        self.interps_coord = []
        self.feat_dist_history = [0]
        self.features = []
        self.is_separated = False

        # self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.kf.F = np.array(
        #     [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
        #      [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # self.kf.H = np.array(
        #     [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        #
        # self.kf.R[2:, 2:] *= 10.
        # self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        # self.kf.P *= 10.
        # self.kf.Q[-1, -1] *= 0.01
        # self.kf.Q[4:, 4:] *= 0.01

        # self.kf.x[:4] = convert_bbox_to_z(np.array(self.location).reshape((4, 1)))

        self.time_since_update = 0
        self.kalman_history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.active = True

    # def kalman_update(self, bbox):
    #     """
    # Updates the state vector with observed bbox.
    # """
    #     self.time_since_update = 0
    #     self.kalman_history = []
    #     self.hits += 1
    #     self.hit_streak += 1
    #     self.kf.update(convert_bbox_to_z(bbox))

    def update(self, location, feature, frame):
        self.location = location

        if len(self.feature_history) == 30:
            self.feature_history.pop(0)
        self.feature_history.append(feature)
        self.features.append(feature)
        feature_list = np.array(self.feature_history)
        self.feature = np.mean(feature_list, axis=0)

        # self.feature = feature
        self.frames.append(frame)
        self.history.append(location)
        self.history_is_detection.append(True)
        # self.kf.predict()
        # self.kalman_history = []
        # self.kf.update(convert_bbox_to_z(location))
        self.time_since_update = 0

    # def kalman_predict(self, frame):
    #     """
    #     Advances the state vector and returns the predicted bounding box estimate.
    #     :return:
    #     """
    #     if (self.kf.x[6] + self.kf.x[2]) <= 0:
    #         self.kf.x[6] *= 0.0
    #     self.kf.predict()
    #     self.age += 1
    #     if self.time_since_update > 0:
    #         self.hit_streak = 0
    #     # self.time_since_update += 1
    #
    #     location = convert_x_to_bbox(self.kf.x)[0]
    #     self.kalman_history.append(location)
    #     self.frames.append(frame)
    #     self.history.append(location)
    #     self.history_is_detection.append(False)
    #     return self.kalman_history[-1]
    def set_is_separated(self, flag):
        self.is_separated = flag

    def get_is_separated(self):
        return self.is_separated

    def add_feat_dist(self, dist):
        self.feat_dist_history.append(dist)

    def get_feat_dist_history(self):
        return self.feat_dist_history

    def add_interp(self, timestamp, coord):
        self.interps.append(timestamp)
        self.interps_coord.append(coord)

    def get_interps(self):
        return self.interps, self.interps_coord

    def not_updated(self):
        self.time_since_update += 1

    def get_time_since_update(self):
        return self.time_since_update

    def deactivate(self):
        self.active = False

    def is_active(self):
        return self.active

    def get_state(self):
        return self.location, self.feature

    def get_location(self):
        return self.history[-1]

    def get_feature(self):
        return self.feature

    def get_frames(self):
        return self.frames

    def get_history(self):
        return self.history

    def get_features(self):
        return self.features

    def get_detection_history(self):
        return self.history_is_detection

    def is_blocked(self):
        return self.block

    def set_block(self, block):
        self.block = block


class RLTrack:
    def __init__(self):
        self.start_frame = 0
        self.end_frame = -1
        self.object = None
        self.track = []
        self.frames = []

    def init_track(self, frame, obj):
        self.object = obj
        self.start_frame = frame
        self.frames.append(frame)
        self.track.append(obj.get_location())

    def get_object(self):
        return self.object

    def update_track(self, frame):
        assert self.object is not None, "no object in this track!"
        self.track.append(self.object.get_location())
        self.frames.append(frame)

    def end_track(self, frame):
        self.end_frame = frame
