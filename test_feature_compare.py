from models.rltracker import build_agent
import gym
import numpy as np

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

args = ""
extractor, _ = build_agent(args)

env = gym.make('gym_rltracking:rltracking-v1')
env.init_source('MOT17-02-FRCNN', "train")
env.set_extractor(extractor)

boxes, feats = env.get_detection(1)
boxes2, feats2 = env.get_detection(2)

cost_feat = _cosine_distance(feats, feats2)
print(cost_feat)