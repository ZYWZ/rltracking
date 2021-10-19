import os
import json
import numpy as np

seq = "MOT17-05-FRCNN"
labels = []
# 1 - 10
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(609).tolist())
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(169).tolist()+np.ones(387).tolist())
labels.append(np.zeros(1039).tolist())
labels.append(np.zeros(459).tolist())
labels.append(np.zeros(402).tolist())
labels.append(np.zeros(64).tolist())
labels.append(np.zeros(583).tolist())


# 11 - 20
labels.append(np.zeros(608).tolist())
labels.append(np.zeros(442).tolist())
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(952).tolist())
labels.append(np.zeros(213).tolist()+np.ones(641).tolist())
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(1050).tolist())
labels.append(np.zeros(504).tolist())
labels.append(np.zeros(875).tolist())
labels.append(np.zeros(619).tolist())

# 21 - 30
labels.append(np.zeros(32).tolist())
labels.append(np.zeros(635).tolist())
labels.append(np.zeros(14).tolist())
labels.append(np.zeros(13).tolist())
labels.append(np.zeros(534).tolist())
labels.append(np.zeros(783).tolist())
labels.append(np.zeros(17).tolist())
labels.append(np.zeros(542).tolist())
labels.append(np.zeros(381).tolist()+np.ones(28).tolist())
labels.append(np.zeros(436).tolist())

# 31 - 40
labels.append(np.zeros(1010).tolist())
labels.append(np.zeros(261).tolist())
labels.append(np.zeros(636).tolist())
labels.append(np.zeros(935).tolist())
labels.append(np.zeros(452).tolist())
labels.append(np.zeros(170).tolist())
labels.append(np.zeros(167).tolist())
labels.append(np.zeros(785).tolist())
labels.append(np.zeros(217).tolist())
labels.append(np.zeros(675).tolist())

# 41 - 50
labels.append(np.zeros(282).tolist())
labels.append(np.zeros(250).tolist())
labels.append(np.zeros(647).tolist())
labels.append(np.zeros(192).tolist())
labels.append(np.zeros(575).tolist())
labels.append(np.zeros(605).tolist())
labels.append(np.zeros(513).tolist())
labels.append(np.zeros(12).tolist())
labels.append(np.zeros(438).tolist())
labels.append(np.zeros(427).tolist())

# 51 - 60
labels.append(np.zeros(411).tolist())
labels.append(np.zeros(75).tolist())
labels.append(np.zeros(402).tolist())
labels.append(np.zeros(385).tolist())
labels.append(np.zeros(255).tolist())
labels.append(np.zeros(258).tolist())
labels.append(np.zeros(218).tolist())
labels.append(np.zeros(199).tolist())
labels.append(np.zeros(195).tolist())
labels.append(np.zeros(176).tolist())

# 61 - 66
labels.append(np.zeros(167).tolist())
labels.append(np.zeros(164).tolist())
labels.append(np.zeros(126).tolist())
labels.append(np.zeros(110).tolist())
labels.append(np.zeros(90).tolist())
labels.append(np.zeros(88).tolist())

output_file_name = os.path.join("manual_label/", "{}.json".format(seq))
json.dump(labels, open(output_file_name, "w"))


seq2 = seq = "MOT17-05-SDP"
labels2 = []

# 1 - 10
labels2.append(np.zeros(1050).tolist())
labels2.append(np.zeros(1050).tolist())
labels2.append(np.zeros(1050).tolist())
labels2.append(np.zeros(609).tolist())
labels2.append(np.zeros(1044).tolist())
labels2.append(np.zeros(203).tolist()+np.ones(711).tolist())
labels2.append(np.zeros(1050).tolist())
labels2.append(np.zeros(459).tolist())
labels2.append(np.zeros(404).tolist()+np.ones(470).tolist())
labels2.append(np.zeros(213).tolist()+np.ones(657).tolist())


# 11 - 20
labels2.append(np.zeros(587).tolist())
labels2.append(np.zeros(1049).tolist())
labels2.append(np.zeros(1041).tolist())
labels2.append(np.zeros(68).tolist())
labels2.append(np.zeros(1039).tolist())
labels2.append(np.zeros(980).tolist())
labels2.append(np.zeros(762).tolist())
labels2.append(np.zeros(442).tolist()+np.ones(1).tolist())
labels2.append(np.zeros(728).tolist()+np.ones(144).tolist())
labels2.append(np.zeros(1004).tolist())


# 21 - 30
labels2.append(np.zeros(765).tolist())
labels2.append(np.zeros(1050).tolist())
labels2.append(np.zeros(608).tolist())
labels2.append(np.zeros(849).tolist())
labels2.append(np.zeros(16).tolist())
labels2.append(np.zeros(32).tolist())
labels2.append(np.zeros(30).tolist())
labels2.append(np.zeros(637).tolist())
labels2.append(np.zeros(18).tolist())
labels2.append(np.zeros(620).tolist())

# 31 - 40
labels2.append(np.zeros(564).tolist())
labels2.append(np.zeros(309).tolist())
labels2.append(np.zeros(782).tolist())
labels2.append(np.zeros(340).tolist())
labels2.append(np.zeros(465).tolist())
labels2.append(np.zeros(637).tolist())
labels2.append(np.zeros(910).tolist())
labels2.append(np.zeros(453).tolist())
labels2.append(np.zeros(1).tolist()+np.ones(887).tolist())
labels2.append(np.zeros(454).tolist())

# 41 - 50
labels2.append(np.zeros(170).tolist())
labels2.append(np.zeros(167).tolist())
labels2.append(np.zeros(747).tolist())
labels2.append(np.zeros(754).tolist())
labels2.append(np.zeros(232).tolist()+np.ones(422).tolist())
labels2.append(np.zeros(292).tolist()+np.ones(105).tolist())
labels2.append(np.zeros(237).tolist())
labels2.append(np.zeros(274).tolist()+np.ones(3).tolist())
labels2.append(np.zeros(645).tolist())
labels2.append(np.zeros(206).tolist()+np.ones(285).tolist())


# 51 - 60
labels2.append(np.zeros(583).tolist())
labels2.append(np.zeros(12).tolist())
labels2.append(np.zeros(479).tolist())
labels2.append(np.zeros(88).tolist())
labels2.append(np.zeros(309).tolist())
labels2.append(np.zeros(408).tolist())
labels2.append(np.zeros(396).tolist())
labels2.append(np.zeros(239).tolist())
labels2.append(np.zeros(228).tolist())
labels2.append(np.zeros(223).tolist())


# 61 - 70
labels2.append(np.zeros(200).tolist())
labels2.append(np.zeros(24).tolist())
labels2.append(np.zeros(181).tolist())
labels2.append(np.zeros(169).tolist())
labels2.append(np.zeros(166).tolist())
labels2.append(np.zeros(142).tolist())
labels2.append(np.zeros(126).tolist())
labels2.append(np.zeros(89).tolist())
labels2.append(np.zeros(60).tolist())


output_file_name = os.path.join("manual_label/", "{}.json".format(seq2))
json.dump(labels2, open(output_file_name, "w"))