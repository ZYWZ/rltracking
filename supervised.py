"""
pre-train the agent model with supervised learning before reinforcement learning

"""
import argparse
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
from model import RLTRdemo
from rltr import RLTR
from utils.criterion import SetCriterion
from utils.rltrCriterion import RltrCriterion
import gym
import random
import time
import datetime

from utils.logger import MetricLogger, SmoothedValue
from utils.simple_matcher import build_matcher
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

FINE_TUNING = False
total_frames = 1000

INIT_MODEL_PATH = "models/state_dict_rltr_init.pt"
MODEL_PATH = "models/state_dict_rltr.pt"

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
    parser.add_argument('--lr', default=0.00001, type=float,
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


def _get_detection(view="001"):
    result = []
    with open('datasets/PETS09/View_'+view+'/View_'+view+'_input.txt') as f:
        content = f.read().splitlines()
    for c in content:
        c_list = c.split(",")
        result.append(c_list)
    return result


def _get_gt(view="001"):
    gt = []
    with open('datasets/PETS09/View_'+view+'/View_'+view+'.txt') as f:
        content = f.read().splitlines()
    for c in content:
        c_list = c.split(" ")
        gt.append(c_list)
    return gt


class Learner:
    """
    Learner that update agent parameters based on supervised transTrack approaches
    """

    def __init__(self, args):
        self.args = args
        if args.gpu_training:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = RLTR()
        if args.load_pretrained_model:
            self.model.load_state_dict(torch.load(INIT_MODEL_PATH))
        if args.keep_training:
            self.model.load_state_dict(torch.load(MODEL_PATH))
        self.matcher = build_matcher(args)
        _losses = []
        if args.loss_operation:
            _losses.append('operations')
        if args.loss_bbox:
            _losses.append('boxes')
        self.criterion = RltrCriterion(losses=_losses, matcher=self.matcher, device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.trajectories = ""
        self.gt = _get_gt(args.view)
        self.detection = _get_detection(args.view)
        self.env = gym.make('gym_rltracking:rltracking-v0')
        self.env.init_view(args.view)
        self.env.init_device(args.gpu_training)
        self.last_detection = []

    def get_policy(self, obs, model):
        outputs = model(obs)
        return Normal(outputs['pred_boxes'], torch.Tensor([0.005]))

    def get_action(self, obs, model):
        return self.get_policy(obs, model).sample()

    def load_data(self, num_frames):
        # load data as x,y,w,h type
        gt = []
        det = []
        gt_all = self.gt
        for i in range(num_frames, num_frames + 10):

            sub_gt = []
            sub_det = []
            for line in gt_all:
                if int(line[5]) == i and int(line[6]) == 0:
                    label = int(line[0])
                    w = int(line[3]) - int(line[1])
                    h = int(line[4]) - int(line[2])
                    xx = int(line[1]) + w / 2
                    yy = int(line[2]) + h / 2
                    ww = w
                    hh = h
                    sub_gt.append([label, xx, yy, ww, hh])

            gt.append(sub_gt)

            for line in self.detection:
                if int(line[0]) == num_frames + 1:
                    w = int(line[3]) - int(line[1])
                    h = int(line[4]) - int(line[2])
                    xx = int(line[1]) + w/2
                    yy = int(line[2]) + h/2
                    ww = w
                    hh = h
                    sub_det.append([xx, yy, ww, hh])

            det.append(sub_det)

        return gt, det

    def _gen_target(self, gt1, gt2):
        int_gt = []
        labels = []
        if len(gt1) != 0:
            for g in gt1:
                int_gt.append([int(g[1]), int(g[2]), int(g[3]), int(g[4])])
                labels.append(int(g[0]))
        result1 = {
            "labels": torch.Tensor(labels).to(self.device),
            "boxes": torch.Tensor(int_gt).to(self.device)
        }

        int_gt = []
        labels = []
        for g in gt2:
            int_gt.append([int(g[1]), int(g[2]), int(g[3]), int(g[4])])
            labels.append(int(g[0]))
        result2 = {
            "labels": torch.Tensor(labels).to(self.device),
            "boxes": torch.Tensor(int_gt).to(self.device)
        }
        return [result1, result2]

    def _gen_output(self, output):
        img_w, img_h = (720, 576)
        output['pred_boxes'] = output['pred_boxes'] * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(self.device)
        return output

    def train_one_epoch(self, env, model, frame, criterion, data, optimizer, device, epoch):
        model.train()
        criterion.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
        gt, det = data

        obs = env.initiate_obj(frame)

        for i in range(len(gt)):
            output = model(obs)
            op_action = Categorical(logits=output['operations']).sample()
            bbox_action = output['pred_boxes']
            action = {'pred_boxes': bbox_action,
                      'operations': op_action}

            obs, reward, done, _ = self.env.step(action)
            if i != 0:
                target = self._gen_target(gt[i-1], gt[i])
            else:
                target = self._gen_target([], gt[i])
            output = self._gen_output(output)

            loss_dict = criterion(output, target)

            losses = 10 * loss_dict['operations']
            if self.args.loss_bbox:
                losses += 0.1 * loss_dict['loss_bbox'] + 20 * loss_dict['loss_giou']

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses)

        print("Epoch: " + str(epoch) +", Averaged stats:" + str(metric_logger))
        return metric_logger.loss.value

    def run(self):
        self.model.to(self.device)
        epoch = 0
        frame_count = 0
        view = ["001"]
        # view = ["001", "005", "006", "007", "008"]
        losses = []
        mut = (10 / 0.0000001) ** (1 / total_frames)

        while frame_count < total_frames:
            frame = random.randint(0, 500)
            # frame = 15
            view_index = int(frame_count / 1000)
            if view_index >= len(view):
                view_index = view_index % len(view)
            if frame_count % 1000 == 0:
                self.env.init_view(view[view_index])
                self.gt = _get_gt(view[view_index])
                self.detection = _get_detection(view[view_index])
                print("Change to view: ", view[view_index])
            # frame = frame_count % 500
            data = self.load_data(frame)

            loss = self.train_one_epoch(self.env, self.model, frame, self.criterion, data, self.optimizer, self.device, epoch)
            losses.append(loss)

            # if self.optimizer.param_groups[0]['lr'] < 10:
            #     self.optimizer.param_groups[0]['lr'] *= mut
            if frame_count == 5000:
                self.optimizer.param_groups[0]['lr'] = 0.00001
            epoch += 1
            frame_count += 10

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
