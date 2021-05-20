import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.categorical import Categorical

from utils import box_ops

class RltrCriterion(nn.Module):
    def __init__(self, losses, matcher, device):
        super().__init__()
        self.device = device
        self.losses = losses
        self.matcher = matcher


    def create_gt_operation(self, outputs, gts):
        gt1, gt2 = gts
        gt1 = gt1['labels']
        gt2 = gt2['labels']
        add = [0, 10, 0, 0]
        keep = [0, 0, 10, 0]
        remove = [0, 0, 0, 10]
        ignore = [10, 0, 0, 0]

        op_gt = []

        for i in range(max(len(gt1), len(gt2))):
            if len(gt1) > len(gt2):
                gt = gt1[i]
            else:
                gt = gt2[i]

            if gt in gt1 and gt not in gt2:
                op_gt.append(remove)
            elif gt not in gt1 and gt in gt2:
                op_gt.append(add)
            elif gt in gt1 and gt in gt2:
                op_gt.append(keep)

        length = len(op_gt)
        for j in range(len(outputs['operations'][0]) - length):
            op_gt.append(ignore)

        return op_gt

    def loss_operation(self, outputs, targets, indices, ind, **kwargs):
        sources = outputs['operations'][0]
        targets = self.create_gt_operation(outputs, targets)
        targets = torch.Tensor(targets).to(self.device)

        loss_operations = F.l1_loss(sources, targets, reduction='none')
        losses = {'operations': loss_operations.sum() / len(outputs['operations'][0])}

        return losses

    def loss_boxes(self, outputs, targets, indices, ind):
        row_ind, col_ind = indices
        sources = outputs['pred_boxes'][0]
        targets = targets[1]['boxes']
        num_boxes = len(row_ind)
        """
            Problem exist! not comparing correctly!
        """
        if len(row_ind) != 0:
            src_boxes = torch.cat([sources[i].unsqueeze(0) for i in row_ind], dim=0)
        else:
            src_boxes = torch.cat([sources[i].unsqueeze(0) for i in range(len(targets))], dim=0)
            num_boxes = len(targets)
            print("The predicted operations are all zero!")

            # empty_src = torch.Tensor([0, 0, 0, 0])
            # src_boxes = torch.cat([empty_src.unsqueeze(0) for i in col_ind], dim=0)
        if len(row_ind) != 0:
            target_boxes = torch.cat([targets[i].unsqueeze(0) for i in col_ind], dim=0)
        else:
            target_boxes = torch.cat([targets[i].unsqueeze(0) for i in range(len(targets))], dim=0)

        assert src_boxes.shape == target_boxes.shape, "src_boxes shape not equals to target_boxes shape, " + str(src_boxes.shape) + str(target_boxes.shape) + str(indices)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, indices, ind, **kwargs):
        loss_map = {
            # 'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'operations': self.loss_operation,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, ind, **kwargs)

    def forward(self, outputs, targets):
        """
        perform loss computation
        :return:
        """
        operations = outputs['operations'].cpu()[0]
        act = Categorical(logits=operations).sample()
        ind = np.where(act != 0)[0]

        # TO-DO this is set to fixed number 3 for now
        # num_boxes = outputs['pred_boxes'][0].shape[0]
        # num_boxes = 3
        # Retrieve the matching between the outputs of the last layer and the targets
        # with matcher
        indices = self.matcher(outputs, targets, ind)
        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, ind, **kwargs))

        return losses