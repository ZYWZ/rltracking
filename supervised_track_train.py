"""
pre-train the agent model with supervised learning before reinforcement learning

"""
import os
from os import walk
import argparse
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.distributions.normal import Normal
from model import RLTRdemo
from rltr import RLTR
from utils.criterion import SetCriterion
from utils.rltrCriterion import RltrCriterion
import gym
import random
import time
import datetime
import numpy as np

from utils.logger import MetricLogger, SmoothedValue
from utils.simple_matcher import build_matcher
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

FINE_TUNING = False
total_frames = 5000

INIT_MODEL_PATH = "models/state_dict_rltr_init.pt"
MODEL_PATH = "models/state_dict_rltr.pt"
SORT_PATH = "datasets/SORT_Output"
INPUT_PATH_TRAIN = "datasets/2DMOT2015/train"

TRACKING_NUMBER = 16

def get_args_parser():
    parser = argparse.ArgumentParser('RLTR tracker', add_help=False)

    # * Matcher
    parser.add_argument('--set_cost_class', default=0, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--view', default="001", type=str,
                        help="view of current scene")

    parser.add_argument('--gpu_training', default=True, type=bool,
                        help='whether to use gpu for training')
    parser.add_argument('--load_pretrained_model', default=False, type=bool,
                        help='whether to load pretrained model')
    parser.add_argument('--keep_training', default=True, type=bool,
                        help='keep train the last model')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='set the training batch size')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='learning rate of the optimizer')

    parser.add_argument('--loss_operation', default=True, type=bool,
                        help='train operation loss')
    parser.add_argument('--loss_bbox', default=True, type=bool,
                        help='train bbox loss')

    return parser


def plot_loss(losses, lr):
    epochs = range(0, int(len(losses)/10))
    sample_loss = []
    for i, loss in enumerate(losses):
        if i % 10 == 0:
            sample_loss.append(loss)

    plt.plot(epochs, sample_loss, 'b', label='Training loss')
    plt.title('Training loss of supervised learning, lr=' + str(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def compare_two_frame(frame1, frame2):
    result = []
    f1_obj = []
    f2_obj = []
    for x in frame1:
        f1_obj.append(x[1])
    for y in frame2:
        f2_obj.append(y[1])
    # first check keep and remove
    for obj in f1_obj:
        if obj in f2_obj:
            result.append(2)
        else:
            result.append(3)
    # then check for add
    for obj in f2_obj:
        if obj not in f1_obj:
            result.append(1)
    # lastly check for ignore
    obj_count = len(result)
    for i in range(TRACKING_NUMBER - obj_count):
        result.append(0)

    return result


def gen_operation(sub_trajectory):
    operations = []
    length = len(sub_trajectory)
    for i in range(length):
        temp = np.zeros(TRACKING_NUMBER)
        operations.append(temp)

    for i, output in enumerate(sub_trajectory):
        if i != 0:
            operations[i] = compare_two_frame(sub_trajectory[i-1], sub_trajectory[i])
        else:
            # print("first frame")
            obj_count = len(sub_trajectory[0])
            operations[0][:obj_count] = [1] * obj_count

    return operations


def gen_bbox_output(sub_trajectory, operations):
    result = []
    for i in range(len(sub_trajectory)):
        frame_result = []
        if i != 0:
            idx = 0
            operation = operations[i]
            for op in operation:
                if op == 0:
                    frame_result.append([0, 0, 0, 0])
                elif op == 1 or op == 2:
                    line = sub_trajectory[i][idx]
                    x = line[2]
                    y = line[3]
                    w = line[4]
                    h = line[5]
                    frame_result.append([x, y, w, h])
                    idx += 1
                elif op == 3:
                    frame_result.append([0, 0, 0, 0])
        else:
            for line in sub_trajectory[i]:
                x = line[2]
                y = line[3]
                w = line[4]
                h = line[5]
                frame_result.append([x, y, w, h])

            obj_count = len(frame_result)
            for i in range(TRACKING_NUMBER - obj_count):
                frame_result.append([0, 0, 0, 0])
        result.append(frame_result)

    return result


def get_trajectory(tracking_results, source, frame, length):
    results = tracking_results[source]
    sub_trajectory = []
    for line in results:
        line = [int(float(x)) for x in line.split(',')]
        if line[0] in range(frame, frame + length):
            sub_trajectory.append(line)

    sub_trajectory_by_frame = []
    for i in range(length):
        sub_trajectory_by_frame.append([])

    for line in sub_trajectory:
        idx = line[0] - frame
        sub_trajectory_by_frame[idx].append(line)

    operations = gen_operation(sub_trajectory_by_frame)
    bbox_outputs = gen_bbox_output(sub_trajectory_by_frame, operations)

    return operations, bbox_outputs


def sample_random_batch(tracking_results, batch_size, source, length):
    trajectories = []
    frames = []
    while len(trajectories) < batch_size:
        frame = random.randint(0, 500)
        trajectory = get_trajectory(tracking_results, source, frame, length)
        trajectories.append(trajectory)
        frames.append(frame)
    return trajectories, frames


def load_tracking_result():
    path = SORT_PATH
    names = []
    _, _, filenames = next(walk(path))
    for file in filenames:
        names.append(file[:-4])

    results = []

    for filename in filenames:
        file = os.path.join(SORT_PATH, filename)
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(names, results))

    return output


def load_detection_result():
    path = INPUT_PATH_TRAIN
    _, directories, _ = next(walk(path))
    results = []
    for directory in directories:
        file = os.path.join(INPUT_PATH_TRAIN, directory, "det", "det.txt")
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        results.append(content)

    output = dict(zip(directories, results))

    return output



class Learner:
    """
    Learner that update agent parameters based on supervised transTrack approaches
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        if args.gpu_training:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RLTR()
        if args.load_pretrained_model:
            self.model.load_state_dict(torch.load(INIT_MODEL_PATH))
        if args.keep_training:
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.matcher = build_matcher(args)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        _losses = ['boxes']
        self.criterion = SetCriterion(losses=_losses, matcher=self.matcher, device=self.device)
        self.trajectories = ""

        self.w = 720
        self.h = 576

    def bbox_rescale(self, input):
        for inp in input:
            for frame in inp:
                for box in frame:
                    box[0] = box[0] / self.w
                    box[1] = box[1] / self.h
                    box[2] = box[2] / self.w
                    box[3] = box[3] / self.h
        return input


    def train_one_batch(self, model, criterion, trajectories, frames, optimizer, device):
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        length = len(trajectories[0][0])
        batch_outputs = []
        obss = []
        envs = []
        for frame in frames:
            env = gym.make('gym_rltracking:rltracking-v0')
            # self.env.init_view(args.view)
            env.init_device(args.gpu_training)
            obs = env.initiate_obj(frame)
            obss.append(obs)
            envs.append(env)

        # CrossEntropyLoss for output operations and sort output
        # and l1 loss and giou loss for bbox prediction
        op_targets = []
        bbox_targets = []
        for trajectory in trajectories:
            op, bbox = trajectory
            op_targets.append(op)
            bbox_targets.append(bbox)
        op_targets = torch.Tensor(op_targets).long().permute(1, 0, 2).to(self.device)
        bbox_targets = torch.Tensor(bbox_targets).permute(1, 0, 2, 3).to(self.device)

        bbox_targets = self.bbox_rescale(bbox_targets)

        for i in range(length):
            policy_logits = model(obss)
            # batch_outputs.append(policy_logits)
            obss = []
            for j, env in enumerate(envs):
                op_action = Categorical(logits=policy_logits['operations'][j]).sample()
                bbox_action = policy_logits['pred_boxes'][j]
                action = {'pred_boxes': bbox_action,
                              'operations': op_action}
                obs, reward, done, _ = env.step(action)
                obss.append(obs)

            # crossentropy loss
            input = policy_logits['operations'].clone().permute(0, 2, 1)
            op_target = op_targets[i]
            crossentropyloss = CrossEntropyLoss()
            loss = crossentropyloss(input, op_target)

            # l1 and giou loss
            input = policy_logits['pred_boxes']
            bbox_target = bbox_targets[i]
            loss_dict = criterion(input, bbox_target)

            # loss = loss_dict['loss_bbox']
            loss = loss + loss_dict['loss_bbox'] + loss_dict['loss_giou']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss)

        return metric_logger.loss.value

    def run(self):
        print("Loading tracking result from SORT...")
        tracking_results = load_tracking_result()
        # detection_results = load_detection_result()

        self.model.to(self.device)
        epoch = 0
        frame_count = 0
        sources = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
        source = 'PETS09-S2L1'
        losses = []
        BATCH_SIZE = self.args.batch_size

        while frame_count < total_frames:
            length = 10
            trajectories, frames = sample_random_batch(tracking_results, BATCH_SIZE, source, length)
            loss = self.train_one_batch(self.model, self.criterion, trajectories, frames, self.optimizer, self.device)
            losses.append(loss)

            print("train step: ", int(frame_count / length), ", Averaged stats: ", str(loss))
            frame_count += length

        print("Saving model...")
        torch.save(self.model.state_dict(), MODEL_PATH)

        plot_loss(losses, self.optimizer.param_groups[0]['lr'])


def main(args):
    learner = Learner(args)
    learner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RLTR supervised training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)