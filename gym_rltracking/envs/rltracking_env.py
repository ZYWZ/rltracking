import gym
import os
from os import walk
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.spaces import Box, Tuple
import torch
from torch import nn
import cv2
import unittest

from utils.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

BASEPATH = "datasets/PETS09/View_"
INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"
VIEW = "001"


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


def _get_detection(view="001"):
    result = []
    filepath = BASEPATH + view + "/View_" + view + "_input.txt"
    with open(filepath) as f:
        content = f.read().splitlines()
    for c in content:
        c_list = c.split(",")
        result.append(c_list)
    return result


# def _get_gt(view="001"):
#     gt = []
#     filepath = BASEPATH + view + "/View_" + view + ".txt"
#     with open(filepath) as f:
#         content = f.read().splitlines()
#     for c in content:
#         c_list = c.split(" ")
#         gt.append(c_list)
#     return gt

def get_gt():
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


def _exist(obj):
    return sorted(obj) == sorted(np.array([0, 0, 0, 0]))


class RltrackingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.device = torch.device("cpu")
        self.step_count = 0
        self.frame_count = 0
        self.img_w = 720
        self.img_h = 576
        self.obj_count = 16
        self.active_object = 0
        self.source = "PETS09-S2L1"
        action_list = []
        for i in range(self.obj_count):
            action = Box(np.array([0, 0, 0, 0]), np.array([+1, +1, +1, +1]), dtype=np.float32)
            action_list.append(action)
        # action space for 16 object
        self.action_space = Tuple(action_list)

        # observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 516), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'next_frame': self.observation_space
        })

        self.rescaled_location = []
        for i in range(self.obj_count):
            self.rescaled_location.append(np.array([0, 0, 0, 0]))

        self.obj_locations = []
        self.obj_memories = []

        self.last_bboxes = []

        self.number = 2
        self.gt_all = get_gt()
        self.det_all = load_detection_result()
        self.gt = []

        self.obs_memory = []

    def init_device(self, use_gpu):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def init_view(self, view):
        # self.gt_all = _get_gt(view)
        # self.det_all = _get_detection(view)

    def init_source(self, source):
        self.source = source
        if source == "PETS09-S2L1":
            self.img_w = 720
            self.img_h = 576

    def initiate_obj(self, start_frame, view="001"):
        self.obj_locations = []
        for i in range(self.obj_count):
            self.obj_locations.append(np.array([0, 0, 0, 0]))

        self.obj_memories = []
        for i in range(self.obj_count):
            self.obj_memories.append(0)

        self.step_count = 0
        self.frame_count = start_frame
        obs = self._get_obs()

        return obs

    def update_object(self, bbox_action, op_action):
        for i, act in enumerate(op_action):
            # add
            if act == 1 and (self.obj_memories[i] == 0 or self.obj_memories[i] == 3):
                box = self.rescale_bboxes(bbox_action[i], (self.img_w, self.img_h))
                self.obj_locations[i] = box
            # keep
            if act == 2 and self.obj_memories[i] != 0 and sorted(self.obj_locations[i]) != sorted(
                    np.array([0, 0, 0, 0])):
                box = self.rescale_bboxes(bbox_action[i], (self.img_w, self.img_h))
                self.obj_locations[i] = box
            # remove
            elif act == 3:
                del self.obj_locations[i]
                self.obj_locations.append([0, 0, 0, 0])
                # self.obj_locations[i] = [0, 0, 0, 0]

        for i, mem in enumerate(self.obj_memories):
            self.obj_memories[i] = op_action[i]

    def step(self, action):
        # directly update the objects' location according to the actions
        bbox_action = action['pred_boxes'].cpu().detach().numpy()
        op_action = action['operations'].cpu().detach().numpy()

        self.update_object(bbox_action, op_action)

        # update ground truth of current frame
        # self.gt = self._get_current_gt()

        # reward = self._tracking_reward(op_action, bbox_action)
        reward = 0
        done = False

        obs = self._get_obs()

        self.step_count += 1
        self.frame_count += 1

        return obs, reward, done, {}

    def reset(self):
        self.step_count = 0
        self.number = 2
        self.gt = self._get_current_gt()
        obs = self._get_obs()

        return obs

    def _tracking_reward(self, op_action, bbox_action):
        reward = 0
        gt = self._get_current_gt()
        gts = []
        for g in gt:
            gts.append([g[1], g[2], g[3], g[4]])

        src = torch.Tensor(self.obj_locations)
        tgt = torch.Tensor(gts)
        ind_row, ind_col = _matcher(src, tgt)

        op_count = 0
        for op in op_action:
            if op == 1 or op == 2:
                op_count += 1

        if len(ind_row) != op_count:
            reward = -1
        else:
            ious = []
            for i, j in zip(ind_row, ind_col):
                ious.append(self.get_iou(self.obj_locations[i], gts[j]))

            if sum(ious) / len(ind_row) > 0.7:
                reward = 1
            else:
                reward = 0

        return reward

    # calculate operation reward for each object
    """
        Not using for now. Only use bbox reward
    """

    # def _op_reward(self, op_action):
    #     reward = 0
    #
    #     keep_count = 0
    #     add_count = 0
    #     remove_count = 0
    #     ignore_count = 0
    #
    #     gt2 = self._get_gt_label(self._get_current_gt())
    #     gt1 = self._get_gt_label(self._get_last_gt())
    #
    #     for i in range(max(len(gt1), len(gt2))):
    #         if len(gt1) > len(gt2):
    #             gt = gt1[i]
    #         else:
    #             gt = gt2[i]
    #
    #         if gt in gt1 and gt not in gt2:
    #             remove_count += 1
    #         elif gt not in gt1 and gt in gt2:
    #             add_count += 1
    #         elif gt in gt1 and gt in gt2:
    #             keep_count += 1
    #
    #     for i, act in enumerate(op_action):
    #         if act == 1:
    #             if add_count > 0:
    #                 add_count -= 1
    #                 reward += 1
    #
    #     return reward

    # match next frame detection to last frame detection, give model a update of all objects
    def _match_bbox(self, new_boxes):
        ind_row, ind_col = _matcher(torch.Tensor(self.last_bboxes), torch.Tensor(new_boxes)[:, :4])
        result = []

        """
            Problem exist! when last frame obj -1, current frame obj +1, the matcher will wrongly assign box
        """
        for i in range(max(len(new_boxes), len(self.last_bboxes))):
            result.append(np.zeros(516).tolist())
        # for i, ind in enumerate(ind_row):
        #     result.append(new_boxes[ind_col[i]])
        for i, j in zip(ind_row, ind_col):
            result[i] = new_boxes[j]

        unstored = []
        for i, box in enumerate(new_boxes):
            if i not in ind_col:
                unstored.append(box)

        for i in range(len(result)):
            if i >= len(self.last_bboxes):
                result[i] = unstored[0]
                unstored = unstored[1:]

        return result

    def _IoU(self, gt_labels, pred_box):
        """
            modified from: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
            calculate the iou multiple gt_boxes and 1 pred_box (the same one)
            pred_boxes: multiple predict  boxes coordinate
            gt_box: ground truth bounding  box coordinate
            return: the max overlaps about pred_boxes and gt_box
        """
        # 1. calculate the inters coordinate
        gt_boxes = []
        for label in gt_labels:
            gt_box = np.array([label[1], label[2], label[3], label[4]])
            gt_boxes.append(gt_box)
        gt_boxes = np.array(gt_boxes)

        # print(gt_boxes, pred_box)

        if gt_boxes.shape[0] > 0:
            ixmin = np.maximum(gt_boxes[:, 0], pred_box[0])
            ixmax = np.minimum(gt_boxes[:, 2], pred_box[2])
            iymin = np.maximum(gt_boxes[:, 1], pred_box[1])
            iymax = np.minimum(gt_boxes[:, 3], pred_box[3])

            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)

            # 2.calculate the area of inters
            inters = iw * ih

            # 3.calculate the area of union
            uni = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) +
                   (pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) -
                   inters)

            # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
            iou = inters / uni
            iou_max = np.max(iou)
            nmax = np.argmax(iou)
            return iou, iou_max, nmax

    def get_iou(self, pred_box, gt_box):
        """
        source code from: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        """
        # 1.get the coordinate of inters
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # 2. calculate the area of inters
        inters = iw * ih

        # 3. calculate the area of union
        uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni

        return iou

    def _get_obs(self):
        """
        read next frame's detection result, save current obj location.
        :return: observation for next iteration
        """

        next_frame = self.frame_count + 1
        result = self.det_all[self.source]
        obs = []
        new_boxes = []
        mask = []

        for line in result:
            line = line.split(',')
            if int(line[0]) == next_frame:
                temp = []
                for i in line[2:6]:
                    temp.append(float(i))
                new_boxes.append(temp)
                mask.append(1)

        result = new_boxes
        for box in result:
            if box[0] < 0:
                box[0] = 0
            box[0] = box[0] / self.img_w * 1000
            box[1] = box[1] / self.img_h * 1000
            box[2] = box[2] / self.img_w * 1000
            box[3] = box[3] / self.img_h * 1000

        for i in range(self.obj_count - len(result)):
            result.append([0, 0, 0, 0])
            mask.append(0)

        # next_frame = torch.as_tensor(result)

        obs = {
            'next_frame': torch.Tensor(result).long().to(self.device),
            'mask': torch.Tensor(mask).to(self.device) > 0,
            'locations': torch.Tensor(self.obj_locations).long().to(self.device),
        }

        # self.obs_memory = next_frame
        return obs

    # def _get_gt_label(self, gt):
    #     labels = []
    #     for g in gt:
    #         labels.append(g[0])
    #     return labels
    #
    # def _get_last_gt(self):
    #     result = []
    #     if self.step_count == 0:
    #         return result
    #     else:
    #         gt_all = self.gt_all
    #         for gt in gt_all:
    #             if int(gt[5]) == self.frame_count - 1 and int(gt[6]) == 0:
    #                 for i in range(1, 5):
    #                     gt[i] = int(gt[i])
    #                 result.append(gt)
    #         return result

    # def _get_current_gt(self):
    #     result = []
    #     gt_all = self.gt_all
    #     for gt in gt_all:
    #         if int(gt[5]) == self.frame_count and int(gt[6]) == 0:
    #             for i in range(1, 5):
    #                 gt[i] = int(gt[i])
    #             result.append(gt)
    #     return result

    def _get_current_gt(self, source="PETS09-S2L1"):
        gt_all = self.gt_all[source]
        result = []
        for gt in gt_all:
            gt = gt.split(',')
            if int(gt[0]) == self.frame_count:
                for i in range(len(gt)):
                    gt[i] = int(float(gt[i]))
                result.append(gt)
        return result


    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x
        b = [x_c, y_c,
             (x_c + w), (y_c + h)]
        return b

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * np.array([img_w, img_h, img_w, img_h])
        # b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def render_img(self, action):
        bbox_action = action['pred_boxes'].detach().numpy()
        op_action = action['operations'].detach().numpy()

        img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        gts = self._get_current_gt("PETS09-S2L1")

        # show gt of current frame
        for gt in gts:
            start_point = (gt[2], gt[3])
            end_point = (gt[2]+gt[4], gt[3]+gt[5])
            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), cv2.FILLED)

        # show next frame det result
        # result = self.det_all
        # next_frame = self.frame_count + 1
        # new_boxes = []
        # for line in result:
        #     if int(line[0]) == next_frame:
        #         temp = []
        #         for i in line[1:]:
        #             temp.append(float(i))
        #         new_boxes.append(temp)

        # match next frame result to last frame result, if
        # result = self._match_bbox(new_boxes)
        #
        # count = 0
        # for det in result:
        #     start_point = (int(det[0]), int(det[1]))
        #     end_point = (int(det[2]), int(det[3]))
        #     img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), cv2.FILLED)
        #     count += 1
        #     cv2.putText(img, str(count),
        #                 start_point,
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1,
        #                 (255, 255, 255),
        #                 2)

        # show current det (self.obs_memory)
        # for det in self.obs_memory:
        #     start_point = (int(det[0]*self.img_w), int(det[1]*self.img_h))
        #     end_point = (int(det[2]*self.img_w), int(det[3]*self.img_h))
        #     img = cv2.rectangle(img, start_point, end_point, (255, 255, 0), cv2.FILLED)

        # show output of the model
        # for i, obj_rescaled in enumerate(self.rescaled_location):
        #     start_point = (int(obj_rescaled[0]), int(obj_rescaled[1]))
        #     end_point = (int(obj_rescaled[2]), int(obj_rescaled[3]))
        #     img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), cv2.FILLED)

        for i, act in enumerate(bbox_action):
            if op_action[i] != 0:
                box = self.rescale_bboxes(act, (self.img_w, self.img_h))
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[2]), int(box[3]))
                img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), cv2.FILLED)

        # show frame count top left corner
        cv2.putText(img, str(self.frame_count),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        cv2.imshow("Image", img)
        cv2.waitKey(-1)

    def render(self, mode='human'):
        print("render frame", self.frame_count)
