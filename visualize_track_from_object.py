# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import os
import numpy as np
import enum
import torch
from scipy.optimize import linear_sum_assignment

device = "cuda"

# Networkx is not precisely made with drawing as its main feature but I experimented with it a bit
class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1

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


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [x_c, y_c,
         (x_c + w), (y_c + h)]
    return b


def _matcher(srcs, tgt, type='block'):
    src_boxes = torch.Tensor(srcs)
    tgt_box = torch.Tensor(tgt).unsqueeze(0)

    cost_bbox = torch.cdist(src_boxes, tgt_box, p=1)

    C = cost_bbox

    C = C.cpu()

    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind


def build_edge_index(object):
    source_nodes_ids, target_nodes_ids = [], []
    index_list = object.get_frames()
    if len(index_list) > 1:
        for idx, obj in enumerate(index_list):
            source_nodes_ids.append(idx)
            target_nodes_ids.append(idx - 1)

            source_nodes_ids.append(idx)
            target_nodes_ids.append(idx + 1)

    edge_index = np.row_stack((source_nodes_ids[1:-1], target_nodes_ids[1:-1]))

    return edge_index

def process_features_for_model(node_app_features, node_st_features):
    processed_ap_feature, processed_st_feature = [], []

    for i in range(len(node_app_features)):
        node_app_features[i] = np.array(node_app_features[i])
        if i == 0:
            # processed_ap_feature.append(node_app_features[i].tolist())
            processed_ap_feature.append(np.zeros(2048).tolist())
            # processed_ap_feature.append([0])
        if i > 0:
            processed_ap_feature.append((node_app_features[i] - node_app_features[i-1]).tolist())
            # feat_dist = _cosine_distance([node_app_features[i]], [node_app_features[i - 1]])
            # processed_ap_feature.append(feat_dist[0].tolist())

    for i in range(len(node_st_features)):
        if i == 0:
            processed_st_feature.append([1, 0, 0, 0, 0])
        else:
            x1, y1, w1, h1, t1 = node_st_features[i-1]
            x2, y2, w2, h2, t2 = node_st_features[i]
            x = 2*(x2-x1)/(w1+w2)
            y = 2*(y2-y1)/(h1+h2)
            w = np.log(w2/w1)
            h = np.log(h2/h1)
            t = t2-t1
            processed_st_feature.append([x, y, w, h, t])

    processed_ap_feature = torch.Tensor(processed_ap_feature).to(device)
    processed_st_feature = torch.Tensor(processed_st_feature).to(device)
    processed_features = torch.cat([processed_ap_feature, processed_st_feature], 1)
    # processed_features = processed_ap_feature
    return processed_features


def extract_tracklet_features_from_object(objects):
    node_topologies = []
    processed_features = []

    for obj in objects:
        topology = build_edge_index(obj)
        topology = torch.Tensor(topology).to(torch.int64).to(device)
        node_topologies.append(topology)
        feature_list = []
        st_feature_list = []
        for frame, loc, feature in zip(obj.get_frames(), obj.get_history(), obj.get_features()):
            feature_list.append(feature)
            st_feature_list.append([loc[0], loc[1], loc[2] - loc[0], loc[3] - loc[1], frame])
            # feature_list.append(frame)

        processed_features.append(process_features_for_model(feature_list, st_feature_list))
    return processed_features, node_topologies


def load_graph_data(objects, gts, w_h):
    node_topologies = []
    node_app_features = []
    node_st_features = []
    node_labels = load_graph_labels(objects, gts, w_h)
    for obj in objects:
        num_of_nodes = len(obj.get_frames())
        topology = build_edge_index(obj)
        node_topologies.append(topology)
        feature_list = []
        st_feature_list = []
        for frame, loc, feature in zip(obj.get_frames(), obj.get_history(), obj.get_features()):
            feature_list.append(feature)
            st_feature_list.append([loc[0], loc[1], loc[2] - loc[0], loc[3] - loc[1], frame])
            # feature_list.append(frame)
        node_app_features.append(feature_list)
        node_st_features.append(st_feature_list)

    node_features = node_app_features, node_st_features
    return node_features, node_labels, node_topologies


def process_gts(gts, w_h):
    img_w, img_h = w_h
    gts_dict = {}
    for gt in gts:
        obj_index = gt[1]
        frame = gt[0]
        box = gt[2], gt[3], gt[4], gt[5]
        box = box_cxcywh_to_xyxy(box)

        # check for out-bounded gt
        if box[0] < 0:
            box[0] = 0
        if box[2] > img_w:
            box[2] = img_w
        if box[1] < 0:
            box[1] = 0
        if box[3] > img_h:
            box[3] = img_h

        if obj_index not in gts_dict:
            gts_dict.setdefault(obj_index, {}).setdefault(frame, box)
        else:
            gts_dict[obj_index].setdefault(frame, box)

    return gts_dict


def load_graph_labels(objects, gts, w_h):
    processed_gt = process_gts(gts, w_h)
    labels = []
    for obj in objects:
        start_frame = obj.get_frames()[0]
        tgt_box = obj.get_history()[0]
        src_boxes = []
        src_indices = []
        for gt in gts:
            if gt[0] == start_frame and int(gt[7]) <= 2 and int(gt[6]) == 1 and float(gt[8]) > 0.3:
                box = gt[2], gt[3], gt[4], gt[5]
                box = box_cxcywh_to_xyxy(box)
                src_boxes.append(box)
                src_indices.append(gt[1])
        row_ind = _matcher(src_boxes, tgt_box)[0]
        start_index = src_indices[row_ind]

        obj_labels = []
        # compare the following bboxes with the gts with the specific start_index, define their labels
        for frame, loc in zip(obj.get_frames(), obj.get_history()):
            if frame not in processed_gt[start_index]:
                obj_labels.append(False)
            else:
                iou = bb_intersection_over_union(loc, processed_gt[start_index][frame])
                if iou > 0:
                    obj_labels.append(True)
                else:
                    obj_labels.append(False)
        labels.append(obj_labels)

    # if False takes more than 80%, revert labels
    # for obj_labels in labels:
    #     false_count = obj_labels.count(False)
    #     result = false_count / len(obj_labels)
    #     if result >= 0.8:
    #         for i in range(len(obj_labels)):
    #             obj_labels[i] = not obj_labels[i]

    return labels


def plot_graph_distributions(node_features, node_labels, node_topologies):
    pass


def visualize_graph(node_features, node_labels, node_topologies):
    # visualization_tool = GraphVisualizationTool.IGRAPH
    label_to_color_map = {True: "blue", False: "red"}
    for i in range(len(node_labels)):
        topology = node_topologies[i]
        labels = node_labels[i]
        edge_topology = list(zip(topology[0, :], topology[1, :]))  # igraph requires this format

        g = ig.Graph()
        g.add_vertices(len(labels))
        g.add_edges(edge_topology)
        visual_style = {}

        visual_style["vertex_color"] = [label_to_color_map[label] for label in labels]
        visual_style["layout"] = g.layout_kamada_kawai()

        ig.plot(g, "track visualization/Track "+str(i)+".png", **visual_style)

