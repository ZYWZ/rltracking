from multiprocessing import Pool
import json
import cv2
from proto import tracking_results_pb2
import os
from os import walk
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

basePath = "saved_results"
_SPLIT_random_min_track_length = 5
_PURE_TRACK_SPLIT_SELECT_THRESHOLD = 1

moving_dataset = ['MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13']

global tracklet_count

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


def save_visualize_graph_data_splited(result, filename, tracklet_index):
    print("visualizing graph data")
    filename = filename[:-3]
    img_base_path = "datasets/MOT17/train/"
    img_path = os.path.join(img_base_path, filename, "img1")
    save_path = os.path.join(basePath, "visualize", filename)
    _, result_st_feature, result_label, _ = result
    for i, result in enumerate(result_st_feature):
        x, y, w, h, frame = result
        img_name = '0' * (6 - len(str(frame))) + str(frame) + '.jpg'
        img_file = os.path.join(img_path, img_name)
        img = cv2.imread(img_file)
        crop_img = img[y:y + h, x:x + w]
        # cv2.imshow('Img', crop_img)
        # cv2.waitKey(0)
        if not os.path.isdir('saved_results/visualize/%03d/' % tracklet_index):
            os.mkdir('saved_results/visualize/%03d/' % tracklet_index)  # make sure the directory exists
        if result_label[i] == 0:
            if not cv2.imwrite('saved_results/visualize/%03d/%03d_True.jpg' % (tracklet_index, frame), crop_img):
                raise Exception("Could not write image")
        else:
            if not cv2.imwrite('saved_results/visualize/%03d/%03d_False.jpg' % (tracklet_index, frame), crop_img):
                raise Exception("Could not write image")


def save_visualize_graph_data(result, filename):
    print("visualizing graph data")
    filename = filename[:-3]
    img_base_path = "datasets/MOT17/train/"
    img_path = os.path.join(img_base_path, filename, "img1")
    save_path = os.path.join(basePath, "visualize", filename)
    _, result_st_feature, result_label, _ = result
    for i, result in enumerate(result_st_feature):
        for j,res in enumerate(result):
            x, y, w, h, frame = res
            img_name = '0'*(6-len(str(frame))) + str(frame) + '.jpg'
            img_file = os.path.join(img_path, img_name)
            img = cv2.imread(img_file)
            crop_img = img[y:y + h, x:x + w]
            # cv2.imshow('Img', crop_img)
            # cv2.waitKey(0)
            if not os.path.isdir('saved_results/visualize/%03d/' % i):
                os.mkdir('saved_results/visualize/%03d/' % i)  # make sure the directory exists
            if result_label[i][j] == 0:
                if not cv2.imwrite('saved_results/visualize/%03d/%03d_True.jpg' % (i, frame), crop_img):
                    raise Exception("Could not write image")
            else:
                if not cv2.imwrite('saved_results/visualize/%03d/%03d_False.jpg' % (i, frame), crop_img):
                    raise Exception("Could not write image")

def read_tracklet(filename):
    file = os.path.join(basePath, filename)
    tracklets = tracking_results_pb2.Tracklets()
    f = open(file, "rb")
    tracklets.ParseFromString(f.read())
    seq_ap_features = []
    seq_st_features = []
    seq_labels = []
    seq_topology = []
    for tracklet in tracklets.tracklet[:]:
        ap_features = []
        st_features = []
        labels = []
        topology = []

        ap_list = tracklet.ap_feature_list.features
        for ap in ap_list:
            ap_features.append([a for a in ap.feats])
        st_list = tracklet.st_features.features
        for st in st_list:
            st_features.append([st.x, st.y, st.w, st.h, st.frame])
        label_list = tracklet.label_list.label
        for label in label_list:
            labels.append(label.label[0])
        tp_list = tracklet.topology
        for tp in tp_list.edges:
            topology.append([tp.source, tp.target])
        seq_ap_features.append(ap_features)
        seq_st_features.append(st_features)
        seq_labels.append(labels)
        seq_topology.append(topology)
    seq_results = seq_ap_features, seq_st_features, seq_labels, seq_topology
    return seq_results


def process_features(results):
    result_ap_feature, result_st_feature, result_label, result_topology = results

    processed_ap_feature, processed_st_feature, processed_labels = [], [], []
    for i in range(len(result_ap_feature)):
        result_ap_feature[i] = np.array(result_ap_feature[i])
        if i == 0:
            # processed_ap_feature.append(result_ap_feature[i].tolist())
            processed_ap_feature.append(np.zeros(2048).tolist())
            # processed_ap_feature.append([0])
        if i > 0:
            processed_ap_feature.append((result_ap_feature[i] - result_ap_feature[i-1]).tolist())
            # feat_dist = _cosine_distance([result_ap_feature[i]], [result_ap_feature[i-1]])
            # processed_ap_feature.append(feat_dist[0].tolist())

    for i in range(len(result_st_feature)):
        if i == 0:
            processed_st_feature.append([0, 0, 0, 0, 0])
        else:
            x1, y1, w1, h1, t1 = result_st_feature[i-1]
            x2, y2, w2, h2, t2 = result_st_feature[i]
            x = 2*(x2-x1)/(w1+w2)
            y = 2*(y2-y1)/(h1+h2)
            w = np.log(w2/w1)
            h = np.log(h2/h1)
            t = t2-t1
            processed_st_feature.append([x, y, w, h, t])
    ind = result_topology[0][0]
    for i in range(len(result_topology[0])):
        result_topology[0][i] = result_topology[0][i] - ind
        result_topology[1][i] = result_topology[1][i] - ind

    #process label, from two class to normal point/break point
    processed_labels.append(0)
    for i in range(len(result_label[1:])):
        cur_label = result_label[i+1]
        prev_label = result_label[i]
        if cur_label == prev_label:
            processed_labels.append(0)
        else:
            processed_labels.append(1)

    processed_results = processed_ap_feature, processed_st_feature, processed_labels, result_topology.tolist()
    return processed_results


def random_split_tracklet_pure(labels, split_num):
    totalNum = len(labels)
    numList = sorted(list(set([np.random.randint(1, totalNum) for ii in range(split_num - 1)])))
    split_objList = []
    result_index = []
    for i, num in enumerate(numList):
        if i == 0:
            split_objList.append(labels[:num])
            result_index.append([0, num])
        else:
            split_objList.append(labels[numList[i - 1]:num])
            result_index.append([numList[i - 1], num])
    if len(labels[split_num:]) > 0:
        split_objList.append(labels[split_num:])
        result_index.append([split_num, totalNum-1])

    results = []
    indexs = []
    for index, split_tracklet in zip(result_index, split_objList):
        if len(split_tracklet) > 30:
            continue
        results.append(split_tracklet)
        indexs.append(index)

    return result_index


def random_split_tracklet(labels, split_num):
    totalNum = len(labels)
    numList = sorted(list(set([np.random.randint(1, totalNum) for ii in range(split_num - 1)])))
    split_objList = []
    result_index = []
    for i, num in enumerate(numList):
        if i == 0:
            split_objList.append(labels[:num])
            result_index.append([0, num])
        else:
            split_objList.append(labels[numList[i - 1]:num])
            result_index.append([numList[i - 1], num])
    if len(labels[split_num:]) > 0:
        split_objList.append(labels[split_num:])
        result_index.append([split_num, totalNum-1])

    results = []
    indexs = []
    for index, split_tracklet in zip(result_index, split_objList):
        if split_tracklet.count(0) == 0:
            continue
        if split_tracklet.count(1) == 0:
            continue
        if split_tracklet[0] == 1:
            continue
        if len(split_tracklet) > 30:
            continue
        results.append(split_tracklet)
        indexs.append(index)

    return indexs


def split_tracklet_pure(seq_results):
    seq_ap_features, seq_st_features, seq_labels, seq_topology = seq_results
    track_lengths = []
    results = []
    for i, labels in enumerate(seq_labels):
        if np.shape(labels)[0] < 5:
            continue
        if labels.count(1) > 0:
            continue
        track_lengths.append(np.shape(labels)[0])
        split_num = np.random.randint(0, len(labels) // _SPLIT_random_min_track_length)
        tracklet_indexs = random_split_tracklet_pure(labels, split_num)
        for index in tracklet_indexs[:_PURE_TRACK_SPLIT_SELECT_THRESHOLD]:
            start, end = index[0], index[1]
            result_ap_feature = seq_ap_features[i][start:end+1]
            result_st_feature = seq_st_features[i][start:end+1]
            result_label = seq_labels[i][start:end+1]
            result_topology = seq_topology[i][2 * start:2 * end]
            result_topology = np.array(result_topology).T

            result = result_ap_feature, result_st_feature, result_label, result_topology
            result = process_features(result)
            results.append(result)
    return results


def split_tracklet_impure(seq_results, filename):
    global tracklet_count
    seq_ap_features, seq_st_features, seq_labels, seq_topology = seq_results
    track_lengths = []
    results = []
    for i, labels in enumerate(seq_labels):
        if np.shape(labels)[0] < 5:
            continue
        if labels.count(0) == 0:
            continue
        if labels.count(1) == 0:
            continue
        track_lengths.append(np.shape(labels)[0])
        split_num = np.random.randint(0, len(labels)//_SPLIT_random_min_track_length)
        tracklet_indexs = random_split_tracklet(labels, split_num)


        for index in tracklet_indexs:
            start, end = index[0], index[1]
            result_ap_feature = seq_ap_features[i][start:end+1]
            result_st_feature = seq_st_features[i][start:end+1]
            result_label = seq_labels[i][start:end+1]
            result_topology = seq_topology[i][2*start:2*end]
            result_topology = np.array(result_topology).T

            result = result_ap_feature, result_st_feature, result_label, result_topology

            # save_visualize_graph_data_splited(result, filename, tracklet_count)
            tracklet_count += 1

            result = process_features(result)
            results.append(result)
    return results


def gen_traindata_one_seq(filename):
    IMPURE_THRESHOLD = 1500
    PURE_THRESHOLD = 1500
    # if filename[:8] in moving_dataset:
    #     IMPURE_THRESHOLD = 500
    #     PURE_THRESHOLD = 2000
    global tracklet_count
    tracklet_count = 0
    seq_results = read_tracklet(filename)
    # seq_ap_features, seq_st_features, seq_labels, seq_topology = seq_results
    # del seq_ap_features[64] # remove bad training tracklet
    # del seq_st_features[64]
    # del seq_labels[64]
    # del seq_topology[64]
    # save_visualize_graph_data(seq_results, filename)
    pure_tracklets_all, impure_tracklets_all, tracklets_all = [], [], []
    while len(impure_tracklets_all) < IMPURE_THRESHOLD:
        impure_tracklets = split_tracklet_impure(seq_results, filename)
        impure_tracklets_all += impure_tracklets

    while len(pure_tracklets_all) < PURE_THRESHOLD:
        pure_tracklets = split_tracklet_pure(seq_results)
        pure_tracklets_all += pure_tracklets

    # ensure ratio
    tracklets_all = pure_tracklets_all + impure_tracklets_all
    # tracklets_all = impure_tracklets_all
    mlog.info("{} produced {} pure and {} impure training tracklets.".format(filename, len(pure_tracklets_all), len(impure_tracklets_all)))
    output_file_name = os.path.join("D:/training_data", "{}.json".format(filename))
    json.dump(tracklets_all, open(output_file_name, "w"))


def gen_traindata_no_split(filename):
    seq_results = read_tracklet(filename)
    results = []
    seq_ap_features, seq_st_features, seq_labels, seq_topology = seq_results
    for i, (result_label, result_ap_feature, result_st_feature, result_topology) in enumerate(zip(seq_labels, seq_ap_features, seq_st_features, seq_topology)):
        if len(result_label) < 2:
            continue
        if i == 64:
            print("skipping bad train tracklet")
            continue
        result_topology = np.array(result_topology).T
        result = result_ap_feature, result_st_feature, result_label, result_topology

        result = process_features(result)
        results.append(result)
    mlog.info("{} produced {} training tracklets.".format(filename, len(results)))
    output_file_name = os.path.join("D:/training_data", "{}.json".format(filename))
    json.dump(results, open(output_file_name, "w"))
if __name__ == "__main__":
    _, _, files = next(walk(basePath))
    # for filename in files[:]:
    #     gen_traindata_one_seq(filename)
    p = Pool(2)
    p.map(gen_traindata_one_seq,
          [filename for i, filename in enumerate(files[:])])
    p.close()
    p.join()
        # gen_traindata_no_split(filename)

# track_lengths = []
# for labels in all_labels:
#     for lb in labels:
#         if np.shape(lb)[0] < 100:
#             continue
#         if lb.count(0) == 0:
#             continue
#         if lb.count(1) == 0:
#             continue
#         track_lengths.append(np.shape(lb)[0])
#         print(lb)
# print(len(track_lengths))
# plt.hist(track_lengths, bins=200)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
# plt.show()
#
# num = 5
# totalNum = 500
# numList = sorted(list(set([np.random.randint(1, totalNum) for ii in range(num-1)])))
# print(numList)
#
# split_number = np.random.randint(0, 500//5)
# print(500//5)
# print(split_number)
