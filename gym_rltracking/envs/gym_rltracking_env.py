import gym
import torch
import os
from os import walk
from gym.spaces import Discrete, Tuple
from PIL import Image
from utils.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as T
import torch.nn.functional as F

import motmetrics as mm
import glob
from collections import OrderedDict
from pathlib import Path

INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"
INPUT_PATH_TEST = "datasets/2DMOT2015/test"


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [x_c, y_c,
         (x_c + w), (y_c + h)]
    return b


def _matcher(src_boxes, tgt_boxes):
    cost_bbox = torch.cdist(src_boxes, tgt_boxes, p=1)
    cost_giou = -generalized_box_iou(src_boxes, tgt_boxes)

    C = 2 * cost_bbox + 5 * cost_giou
    C = C.cpu()

    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind


def load_detection_result():
    _, directories, _ = next(walk(INPUT_PATH_TRAIN))
    results = []
    for directory in directories:
        file = os.path.join(INPUT_PATH_TRAIN, directory, "det", "det.txt")
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(directories, results))

    return output


def load_gt_result():
    _, directories, _ = next(walk(INPUT_PATH_TRAIN))
    results = []
    for directory in directories:
        file = os.path.join(INPUT_PATH_TRAIN, directory, "gt", "gt.txt")
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(directories, results))

    return output


class GymRltrackingEnv(gym.Env):
    def __init__(self):
        self.source = "PETS09-S2L1"
        self.train_mode = True
        self.obj_count = 0
        self.step_count = 1
        self.img_w = 720
        self.img_h = 576
        action_list = []

        for i in range(self.obj_count):
            action = Discrete(4)
            action_list.append(action)

        self.action_space = Tuple(action_list)
        self.objects = []
        self.tracks = []
        self.detection_memory = None

        self.extractor = None

        if self.train_mode:
            self.det_result = load_detection_result()[self.source]
            self.gt_result = load_gt_result()[self.source]
        else:
            print("Not implemented inference mode for env.init")
            raise

    def inference(self):
        self.train_mode = False

    def set_extractor(self, extractor):
        self.extractor = extractor

    def init_source(self, source):
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

    # initiate the objects in env, according to the first frame detection
    def initiate_env(self, start_frame):
        self.tracks = []
        self.objects = []
        self.step_count = start_frame
        det, feat = self.get_detection(self.step_count)
        obj_count = len(det)
        feature = feat
        for i in range(obj_count):
            new_obj = RLObject()
            new_obj.update(det[i], feature[i])
            self.objects.append(new_obj)
            track = RLTrack()
            track.init_track(self.step_count, new_obj)
            self.tracks.append(track)

        return self.gen_obs()

    def update_objects(self, action, det_boxes, det_feat):
        det_box_copy = det_boxes.copy()
        det_feat_copy = det_feat.copy()
        action = action.detach().tolist()[0]
        assert len(action) >= len(self.objects), "action list is shorter than objects in the environment!"
        update_list = []
        update_list_object = []
        remove_list = []
        block_list = []
        for i, obj in enumerate(self.objects):
            if action[i] == 0:
                update_list.append(obj.get_location())
                update_list_object.append(obj)
            if action[i] == 2:
                remove_list.append(obj)
            if action[i] == 1 or action[i] == 3:
                block_list.append(obj)

        # remove object
        for obj in remove_list:
            self.objects.remove(obj)

        # update object
        src = torch.Tensor(update_list).cuda()
        tgt = torch.Tensor(det_boxes).cuda()
        if src.shape[0] != 0:
            ind_row, ind_col = _matcher(src, tgt)
            for i, j in zip(ind_row, ind_col):
                obj = self.objects[i]
                obj.update(det_boxes[j], det_feat[j])
                det_box_copy.remove(det_boxes[j])
                det_feat_copy.remove(det_feat[j])

        # add object, select from the rest of det_box_copy list
        for i, act in enumerate(action[len(self.objects):]):
            if act == 1 and len(det_box_copy) > 0:
                new_obj = RLObject()
                new_obj.update(det_box_copy[0], det_feat_copy[0])
                self.objects.append(new_obj)
                track = RLTrack()
                track.init_track(self.step_count, new_obj)
                self.tracks.append(track)
                det_box_copy.pop(0)
                det_feat_copy.pop(0)

        # modify tracklets
        for track in self.tracks:
            obj = track.get_object()
            if obj in update_list_object:
                track.update_track()
            if obj in block_list:
                track.update_track()
            if obj in remove_list:
                track.end_track(self.step_count)

    def resize_roi(self, roi):
        w = self.img_w
        h = self.img_h
        roi[:, 0] = roi[:, 0] * 1066 / w
        roi[:, 2] = roi[:, 2] * 1066 / w
        roi[:, 1] = roi[:, 1] * 800 / h
        roi[:, 3] = roi[:, 3] * 800 / h
        return roi

    def get_detection(self, frame):
        def get_index(frame):
            result = ""
            digit = len(str(frame))
            max = 6
            for i in range(max - digit):
                result += "0"
            result += str(frame)
            result += ".jpg"

            return result

        boxes = []
        if self.train_mode:
            result = self.det_result

            for line in result:
                line = line.split(',')
                if int(line[0]) == frame:
                    temp = []
                    for i in line[2:6]:
                        if float(i) < 0:
                            i = 0
                        temp.append(float(i))
                    # transform detection from (x, y, w, h) to (x1, y1, x2, y2)
                    temp = box_cxcywh_to_xyxy(temp)
                    boxes.append(temp)

            img_index = get_index(frame)
            img_path = os.path.join(INPUT_PATH_TRAIN, self.source, 'img1', img_index)

            img = Image.open(img_path)
            transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = transform(img).unsqueeze(0).cuda()
            pad = (1, 0)
            box_tensor = torch.Tensor(boxes)
            box_tensor = self.resize_roi(box_tensor)
            box_tensor = F.pad(box_tensor, pad, 'constant', 0).cuda()
            feat = self.extractor(box_tensor, img)

            copy = feat.detach().cpu().numpy().tolist()
            del feat
            torch.cuda.empty_cache()
        else:
            print("Not implemented inference mode for function env.get_next_detection()")
            raise

        return boxes, copy

    def get_detection_memory(self):
        return self.detection_memory

    # update objects in env, according to action
    def step(self, action):
        self.step_count += 1
        end = False
        obs = {}
        boxes, feat = self.get_detection_memory()
        self.update_objects(action, boxes, feat)
        # reward = self.reward()
        reward = 0
        if len(self.objects) <= 0:
            end = True
        else:
            obs = self.gen_obs()

        return obs, reward, end, {}

    def reset(self):
        self.step_count = 1
        self.objects = []
        self.tracks = []

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

    def calculate_mota(self):
        gtfiles = glob.glob(os.path.join('gym_rltracking/envs/rltrack/gt', 'gt.txt'))
        tsfiles = [f for f in glob.glob(os.path.join('gym_rltracking/envs/rltrack', '*.txt')) if
                   not os.path.basename(f).startswith('eval')]
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D')) for f in tsfiles])

        mh = mm.metrics.create()
        accs, names = self.compare_dataframes(gt, ts)

        metrics = list(mm.metrics.motchallenge_metrics)
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        # print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        # print(summary.values[0][13])

        return summary.values[0][13]

    def generate_gt_for_reward(self):
        result = []
        for gt in self.gt_result:
            line = gt.split(',')
            if int(line[0]) <= self.step_count:
                result.append(line)
        output = []
        for res in result:
            res = [float(x) for x in res]
            output.append(res)
        self.write_gt_to_file(output)

    def generate_track_result(self):
        start_idx = 2001
        result = []
        for track in self.tracks:
            start_frame = track.start_frame
            for t in track.track:
                line = [start_frame, start_idx, t[0], t[1], round(t[2] - t[0], 2), round(t[3] - t[1], 2), 1, -1, -1, -1]
                result.append(line)
                start_frame += 1

            start_idx += 1
        self.write_to_file(result)

    def write_gt_to_file(self, res):
        with open('gym_rltracking/envs/rltrack/gt/gt.txt', 'w') as f:
            for item in res:
                f.write(','.join(map(repr, item)))
                f.write('\n')

    def write_to_file(self, res):
        with open('gym_rltracking/envs/rltrack/rltrack.txt', 'w') as f:
            for item in res:
                f.write(','.join(map(repr, item)))
                f.write('\n')

    def compare_objects(self, dets):
        det_count = len(dets)
        env_count = self.obj_count
        return -abs(det_count - env_count)

    def reward(self):
        weight = [5, 3]
        # if self.step_count % 10 == 0:
        self.generate_gt_for_reward()
        self.generate_track_result()
        mota = self.calculate_mota()
        reward_mota = mota * self.step_count

        boxes, _ = self.get_detection_memory()
        reward_obj_count = self.compare_objects(boxes)

        reward = weight[0] * reward_mota + weight[1] * reward_obj_count

        return reward

    def gen_obs(self):
        locations = []
        feats = []
        for obj in self.objects:
            locations.append(obj.get_location())
            feats.append(obj.get_feature())
        locations = torch.Tensor(locations).cuda()
        feats = torch.Tensor(feats).cuda()

        next_frame = self.step_count + 1
        det, det_feat = self.get_detection(next_frame)
        self.detection_memory = (det, det_feat)

        det = torch.Tensor(det).cuda()
        det = self.resize_roi(det)
        det_feat = torch.Tensor(det_feat).cuda()
        locations = self.resize_roi(locations)
        return det, det_feat, locations, feats

    def render(self, mode='human'):
        print(self.step_count)
        if mode =='printTrack':
            for track in self.tracks:
                print(track.track)
            print("-------------------")


class RLObject:
    def __init__(self):
        self.location = 0
        self.feature = None

    def update(self, location, feature):
        self.location = location
        self.feature = feature

    def get_state(self):
        return self.location, self.feature

    def get_location(self):
        return self.location

    def get_feature(self):
        return self.feature


class RLTrack:
    def __init__(self):
        self.start_frame = 0
        self.end_frame = -1
        self.object = None
        self.track = []

    def init_track(self, frame, obj):
        self.object = obj
        self.start_frame = frame
        self.track.append(obj.get_location())

    def get_object(self):
        return self.object

    def update_track(self):
        assert self.object is not None, "no object in this track!"
        self.track.append(self.object.get_location())

    def end_track(self, frame):
        self.end_frame = frame
