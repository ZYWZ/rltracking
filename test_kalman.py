from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.common import Q_discrete_white_noise

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
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class RLObject:
    def __init__(self, location):
        self.location = location
        self.feature = None
        self.block = False
        self.frames = []
        self.history = [location]

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(np.array(self.location).reshape((4, 1)))
        self.time_since_update = 0
        self.kalman_history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

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
        self.feature = feature
        self.frames.append(frame)
        self.history.append(location)

        self.kf.predict()
        # self.kf.x[:4] = convert_bbox_to_z(location)

        self.kf.update(convert_bbox_to_z(location))
        self.kalman_history = []

    def kalman_predict(self, frame):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        :return:
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        location = convert_x_to_bbox(self.kf.x)[0]
        self.kalman_history.append(location)
        self.frames.append(frame)
        self.history.append(location)
        return self.kalman_history[-1]

    def get_location(self):
        return self.history[-1]

location = [1, 1, 20, 20]
obj = RLObject(location)
print(obj.get_location())
location2 = [2, 2, 21, 21]
feat = np.zeros(512)
obj.update(location2, feat, 1)
print(obj.get_location())
obj.kalman_predict(2)
print(obj.get_location())
obj.kalman_predict(3)
print(obj.get_location())

