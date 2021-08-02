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
import configparser
import motmetrics as mm
import glob
from collections import OrderedDict
from pathlib import Path

# INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"
INPUT_PATH_TRAIN = "datasets/MOT17/train"
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
        self.source = None
        self.train_mode = True
        self.obj_count = 0
        self.step_count = 0
        self.start_frame = -1
        self.frame = -1
        self.img_w = -1
        self.img_h = -1
        action_list = []

        for i in range(self.obj_count):
            action = Discrete(4)
            action_list.append(action)

        self.action_space = Tuple(action_list)
        self.objects = []
        self.tracks = []
        self.detection_memory = None

        self.extractor = None
        self.act_reward = 0

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

        seqfile = os.path.join(INPUT_PATH_TRAIN, source, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqfile)
        self.img_w = int(config.get('Sequence', 'imWidth'))
        self.img_h = int(config.get('Sequence', 'imHeight'))

        if self.train_mode:
            self.det_result = load_detection_result()[self.source]
            self.gt_result = load_gt_result()[self.source]
        else:
            print("Not implemented inference mode for env.init")
            raise

    # initiate the objects in env, according to the first frame detection
    def initiate_env(self, start_frame):
        self.tracks = []
        self.objects = []
        self.step_count = 0
        self.start_frame = start_frame
        self.frame = start_frame
        self.detection_memory = self.get_detection(self.frame)
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

        return self.gen_obs()

    def update_env_objects(self, src_list, tgt_list):
        for obj in tgt_list:
            obj.set_block(True)
        update_list = [box[0] for box in src_list]
        obj_list = [obj.get_location() for obj in tgt_list]
        src = torch.Tensor(update_list).cuda()
        tgt = torch.Tensor(obj_list).cuda()

        if src.shape[0] != 0 and tgt.shape[0] != 0:
            ind_row, ind_col = _matcher(tgt, src)
            for i, j in zip(ind_row, ind_col):
                obj = tgt_list[i]
                obj.update(src_list[j][0], src_list[j][1], self.frame)
                obj.set_block(False)

    def update(self, action):
        action = action.detach().tolist()[0]
        boxes, feats = self.get_detection_memory()

        assert len(action) == len(boxes), "action list length not equal to detection numbers!"
        assert len(action) == len(feats), "action list length not equal to detection feature numbers!"

        add_list = []
        update_current_list = []
        update_old_list = []
        for act, box, feat in zip(action, boxes, feats):
            # new object
            if act == 0:
                # print("new object")
                add_list.append((box, feat))
            elif act == 1:
                # print("update current")
                update_current_list.append((box, feat))
            elif act == 2:
                # print("update old")
                update_old_list.append((box, feat))

        reveal_objects = []
        blocked_objects = []
        for obj in self.objects:
            if obj.is_blocked():
                blocked_objects.append(obj)
            else:
                reveal_objects.append(obj)

        # update current object
        self.update_env_objects(update_current_list, reveal_objects)

        # update old object, turn them into current object
        self.update_env_objects(update_old_list, blocked_objects)

        # add new object
        for obj, feat in add_list:
            new_obj = RLObject()
            new_obj.update(obj, feat, self.frame)
            self.objects.append(new_obj)

    def update_objects(self, action, det_boxes, det_feat):
        det_box_copy = det_boxes.copy()
        det_feat_copy = det_feat.copy()
        action = action.detach().tolist()[0]
        assert len(action) >= len(self.objects), "action list is shorter than objects in the environment!"
        update_list = []
        update_list_object = []
        add_list = []
        remove_list = []
        block_list = []
        for i, obj in enumerate(det_boxes):
            if action[i] == 0:
                # update_list.append(obj.get_location())
                # update_list_object.append(obj)
                update_list.append(obj)
            if action[i] == 1:
                add_list.append(obj)
            if action[i] == 2:
                remove_list.append(obj)
            if action[i] == 3:
                block_list.append(obj)

        # env_obj = []
        # for obj in self.objects:
        #     env_obj.append(obj.get_location())

        # remove object
        for obj in remove_list:
            self.objects.remove(obj)

        # update object
        src = torch.Tensor(update_list).cuda()
        tgt = torch.Tensor(det_boxes).cuda()
        if src.shape[0] != 0 and tgt.shape[0] != 0:
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
                track.init_track(self.frame, new_obj)
                self.tracks.append(track)
                det_box_copy.pop(0)
                det_feat_copy.pop(0)

        # modify tracklets
        for track in self.tracks:
            obj = track.get_object()
            if obj in update_list_object:
                track.update_track(self.frame)
            if obj in block_list:
                track.update_track(self.frame)
            if obj in remove_list:
                track.end_track(self.frame)

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
            _, box_tensor = self.resize_roi(box_tensor)
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
        end = False
        obs = {}

        self.act_reward = self.action_reward(action)
        # boxes, feat = self.get_detection_memory()
        # self.update_objects(action, boxes, feat)
        self.update(action)
        # reward = self.reward()
        self.step_count += 1
        self.frame += 1
        self.detection_memory = self.get_detection(self.frame)
        reward = self.reward()

        if len(self.objects) <= 0:
            end = True
        else:
            obs = self.gen_obs()

        return obs, reward, end, {}

    def reset(self):
        self.step_count = 0
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

        return summary.values[0]

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

    def generate_track_from_objects(self):
        idx = 2001
        result = []
        for obj in self.objects:
            for frame, loc in zip(obj.get_frames(), obj.get_history()):
                line = [frame, idx, loc[0], loc[1], round(loc[2] - loc[0], 2), round(loc[3] - loc[1], 2), 1, -1, -1, -1]
                result.append(line)
            idx += 1
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
        weight = [1, 1, 0, 0]
        # if self.step_count % 10 == 0:
        self.generate_gt_for_reward()
        self.generate_track_from_objects()
        summary = self.calculate_mota()
        idf1 = summary[0]
        mota = summary[13]
        motp = summary[14]

        if mota >= 0.5:
            reward_mota = 1
        elif 0.3 < mota < 0.5:
            reward_mota = 0
        else:
            reward_mota = -1

        if idf1 >= 0.5:
            reward_idf1 = 1
        elif 0.3 < idf1 < 0.5:
            reward_idf1 = 0
        else:
            reward_idf1 = -1

        boxes, _ = self.get_detection_memory()
        gt_number = self.get_frame_gt(self.frame)
        reward_obj_count = self.compare_objects(gt_number)

        reward = weight[0] * mota + weight[1] * idf1 + weight[2] * reward_obj_count + weight[3] * self.act_reward

        return reward

    def gen_obs(self):
        locations = []
        feats = []
        # for obj in self.objects:
        #     locations.append(obj.get_location())
        #     feats.append(obj.get_feature())
        # locations = torch.Tensor(locations).cuda()
        # feats = torch.Tensor(feats).cuda()

        frame = self.frame
        det, det_feat = self.get_detection_memory()
        self.detection_memory = (det, det_feat)

        det = torch.Tensor(det).cuda()
        normed_det, det = self.resize_roi(det)
        det_feat = torch.Tensor(det_feat).cuda()
        # locations = self.resize_roi(locations)
        obs = {
            'det': det,
            'det_feat': det_feat,
        }
        return obs

    def render(self, mode='human'):
        print(self.frame)
        if mode =='printTrack':
            for track in self.tracks:
                print(track.track)
            print("-------------------")


class RLObject:
    def __init__(self):
        self.location = 0
        self.feature = None
        self.block = False
        self.frames = []
        self.history = []


    def update(self, location, feature, frame):
        self.location = location
        self.feature = feature
        self.frames.append(frame)
        self.history.append(location)

    def get_state(self):
        return self.location, self.feature

    def get_location(self):
        return self.location

    def get_feature(self):
        return self.feature

    def get_frames(self):
        return self.frames

    def get_history(self):
        return self.history

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
