from proto import detection_results_pb2
file = "datasets/MOT17-01-DPM.pb"

detections = detection_results_pb2.Detections()
f = open(file, "rb")
detections.ParseFromString(f.read())
dets = detections.tracked_detections
type = detections.detection_type
result = []
print(len(dets))
for det in dets[:1]:
    result.append(det)
    feature = list(det.features.features[0].feats)
    print(det.features)


