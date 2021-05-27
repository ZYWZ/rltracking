"""
    Simplified version of the matcher
    match the predicted bbox to ground truth

"""
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy


class SimpleMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, ind):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list contains box assignment from gt to prediction. For example [1, 0] means 2nd gt match 1st prediction,
            1st gt match 2nd prediction
        """
        with torch.no_grad():
            # bs, num_queries = outputs["pred_logits"].shape[:2]
            # obj_number = 3
            # second gt is current gt
            targets = [targets[1]]

            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            out_bbox = out_bbox[ind]
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            # C = C.view(bs, num_queries, -1).cpu()[0]
            C = C.cpu()
            # print(C)
            row_ind, col_ind = linear_sum_assignment(C)
            return row_ind, col_ind

def build_matcher(args):
    return SimpleMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)