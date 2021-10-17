import gym
from models.rltracker import build_agent
import argparse
import numpy as np
from torch.distributions.categorical import Categorical
import torch
import os
from os import walk
import configparser

from proto import tracking_results_pb2

train_test = "train"
MODEL_PATH = "models/state_dict_rltr_RL.pt"
basePath ="datasets/MOT17/"+train_test
def get_args_parser():
    parser = argparse.ArgumentParser('RLTracker args', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # PyTorch checkpointing for saving memory (torch.utils.checkpoint.checkpoint)
    parser.add_argument('--checkpoint_enc_ffn', default=False, action='store_true')
    parser.add_argument('--checkpoint_dec_ffn', default=False, action='store_true')

    return parser

def test():
    _, directories, _ = next(walk(basePath))

    for seq_name in directories[8:9]:
        output_file = os.path.join("saved_results", seq_name+".pb")
        env = gym.make('gym_rltracking:rltracking-v1')
        env.inference()
        flag = False

        seqfile = os.path.join(basePath, seq_name, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqfile)
        seq_length = int(config.get('Sequence', 'seqLength'))

        env.reset()
        env.init_source(seq_name, train_test)
        extractor, agent = build_agent(args)
        extractor.eval()
        env.set_extractor(extractor)

        obs = env.initiate_env(1)
        for i in range(seq_length):
            print(seq_name + " " + str(i) + "/" + str(seq_length))
            action = [0]
            obs, reward, end, _ = env.step(action)
        node_features, node_labels, node_topologies = env.output_result()
        node_app_features, node_st_features = node_features
        tracking_result_pb = tracking_results_pb2.Tracklets()

        for app_features, st_features, labels, topologies in zip(node_app_features, node_st_features, node_labels, node_topologies):
            _tracklet = tracking_result_pb.tracklet.add()
            for d in app_features:
                _af = _tracklet.ap_feature_list.features.add()
                for k in d:
                    _af.feats.append(k)

            for d in st_features:
                _sf = _tracklet.st_features.features.add()
                _sf.x = d[0]
                _sf.y = d[1]
                _sf.w = d[2]
                _sf.h = d[3]
                _sf.frame = d[4]

            for d in labels:
                _lb = _tracklet.label_list.label.add()
                _lb.label.append(0 if d else 1)

            for s, t in zip(topologies[0], topologies[1]):
                _tp = _tracklet.topology.edges.add()
                _tp.source = s
                _tp.target = t

        with open(output_file, 'wb') as f:
            f.write(tracking_result_pb.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RLTracker args', parents=[get_args_parser()])
    args = parser.parse_args()

    test()