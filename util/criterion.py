"""
The class compute loss for supervised learning

The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

"""
import torch
from torch import nn
import torch.nn.functional as F

from utils import box_ops


class SetCriterion(nn.Module):
    def __init__(self, losses, matcher, device):
        super().__init__()
        self.losses = losses
        self.matcher = matcher
        self.device = device
        self.w = 720
        self.h = 576



    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        bbox_loss = F.l1_loss(outputs, targets, reduction='mean')

        losses = {}
        losses['loss_bbox'] = bbox_loss
        losses_giou = []
        for i,target in enumerate(targets):
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(outputs[i]),
                box_ops.box_cxcywh_to_xyxy(target)))

            loss_giou = loss_giou.sum() / 16
            losses_giou.append(loss_giou)

        losses_giou = torch.mean(torch.stack(losses_giou))
        losses['loss_giou'] = losses_giou

        # row_ind, col_ind = indices
        # sources = outputs['pred_boxes'][0]
        # targets = targets[0]['boxes']
        #
        # src_boxes = torch.cat([sources[i].unsqueeze(0) for i in row_ind], dim=0)
        # target_boxes = torch.cat([targets[i].unsqueeze(0) for i in col_ind], dim=0)
        #
        # assert src_boxes.shape == target_boxes.shape, "src_boxes shape not equals to target_boxes shape, " + str(src_boxes.shape) + str(target_boxes.shape) + str(indices)
        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        #
        # losses = {}
        # losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        #
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        #
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            # 'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        perform loss computation
        :return:
        """
        # TO-DO this is set to fixed number 3 for now
        num_boxes = 1
        # num_boxes = 3
        # Retrieve the matching between the outputs of the last layer and the targets
        # with matcher
        indices = (0,1)
        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        return losses
