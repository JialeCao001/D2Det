import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target

@HEADS.register_module
class D2DetHead(nn.Module):

    def __init__(self,
                 num_convs=8,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=36),
                 MASK_ON=False):
        super(D2DetHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = 576
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.MASK_ON = MASK_ON
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0


        self.convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            groups = 1 if i == 0 else 36
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.convs = nn.Sequential(*self.convs)

        self.D2Det_reg = nn.Conv2d(self.conv_out_channels, 4, 3, padding=1)
        self.D2Det_mask = nn.Conv2d(self.conv_out_channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if self.MASK_ON:
            self.deconv1 = nn.ConvTranspose2d(
                self.conv_out_channels,
                256,
                2,
                stride=2)
            self.norm1 = nn.GroupNorm(16, 256)
            self.deconv2 = nn.ConvTranspose2d(
                256,
                256,
                2,
                stride=2)
            self.D2Det_instance = nn.Conv2d(256, 81, 3, padding=1)
            self.fcs = nn.ModuleList()
            for i in range(2):
                in_channels = 577*7*7 if i == 0 else 1024
                self.fcs.append(nn.Linear(in_channels, 1024))
            self.fc_instance_iou = nn.Linear(1024,81)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # TODO: compare mode = "fan_in" or "fan_out"
                kaiming_init(m)
        normal_init(self.D2Det_reg, std=0.001)
        normal_init(self.D2Det_mask, std=0.001)

        if self.MASK_ON:
            nn.init.kaiming_normal_(
                self.deconv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.deconv1.bias, 0)
            nn.init.kaiming_normal_(
                self.deconv2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.deconv2.bias, 0)

            for fc in self.fcs:
                kaiming_init(
                    fc,
                    a=1,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='uniform')
            normal_init(self.fc_instance_iou, std=0.01)
            
    def forward(self, x, idx=None):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x0 = self.convs(x)
        x_m = self.D2Det_mask(x0)
        x_r = self.relu(self.D2Det_reg(x0))
        
        if idx is not None:
            x2 = self.deconv1(x0)
            x2 = F.relu(self.norm1(x2), inplace=True)
            x2 = self.deconv2(x2)
            x_s = self.D2Det_instance(F.relu(x2, inplace=True))

            xs = x_s[idx > 0, idx].detach()
            xs = F.max_pool2d(xs.unsqueeze(1), 4, 4, 1)
            xi = torch.cat([x0, xs.sigmoid()], dim=1)
            xi = xi.view(xi.size(0), -1)
            for fc in self.fcs:
                xi = self.relu(fc(xi))
            x_i = self.fc_instance_iou(xi)
            return x_r, x_m, x_s, x_i
        return x_r, x_m

    def get_target(self, sampling_results):
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results],
                               dim=0).cpu()
        pos_gt_bboxes = torch.cat(
            [res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape
        num_rois = pos_bboxes.shape[0]
        map_size = 7
        targets = torch.zeros((num_rois, 4, map_size, map_size),dtype=torch.float)
        points = torch.zeros((num_rois, 4, map_size, map_size),dtype=torch.float)
        masks = torch.zeros((num_rois, 1, map_size, map_size), dtype=torch.float)

        for j in range(map_size):
            y = pos_bboxes[:, 1] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / map_size * (j+0.5)

            dy = (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / (map_size - 1)
            for i in range(map_size):
                x = pos_bboxes[:, 0] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / map_size * (i+0.5)

                dx = (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / (map_size - 1)

                targets[:, 0, j, i] = x - pos_gt_bboxes[:, 0]
                targets[:, 1, j, i] = pos_gt_bboxes[:, 2] - x
                targets[:, 2, j, i] = y - pos_gt_bboxes[:, 1]
                targets[:, 3, j, i] = pos_gt_bboxes[:, 3] - y

                idx = ((x-pos_gt_bboxes[:,0]>=dx) & (pos_gt_bboxes[:, 2] - x>=dx) & (y - pos_gt_bboxes[:, 1]>=dy) & (pos_gt_bboxes[:, 3] - y>=dy))

                masks[idx, 0, j, i] = 1

                points[:, 0, j, i] = x
                points[:, 1, j, i] = y
                points[:, 2, j, i] = pos_bboxes[:, 2] - pos_bboxes[:, 0]
                points[:, 3, j, i] = pos_bboxes[:, 3] - pos_bboxes[:, 1]

        targets = targets.cuda()
        points = points.cuda()
        masks = masks.cuda()
        return points, targets, masks

    def get_target_mask(self, sampling_results, gt_masks, rcnn_train_cfg):
        # mix all samples (across images) together.
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results],
                               dim=0)
        pos_gt_bboxes = torch.cat(
            [res.pos_gt_bboxes for res in sampling_results], dim=0)

        assert pos_bboxes.shape == pos_gt_bboxes.shape


        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        instances = mask_target([res.pos_bboxes for res in sampling_results], pos_assigned_gt_inds, gt_masks,
                            rcnn_train_cfg)
        masks = F.interpolate(instances.unsqueeze(0), scale_factor=1 / 4, mode='bilinear', align_corners=False).squeeze(0)
        masks = masks.gt(0.5).float()

        num_rois = pos_bboxes.shape[0]
        map_size = 7
        targets = pos_bboxes.new_zeros((num_rois, 4, map_size, map_size), dtype=torch.float)
        points = pos_bboxes.new_zeros((num_rois, 4, map_size, map_size), dtype=torch.float)

        for j in range(map_size):
            y = pos_bboxes[:, 1] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / map_size * (j+0.5)

            for i in range(map_size):
                x = pos_bboxes[:, 0] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / map_size * (i+0.5)

                targets[:, 0, j, i] = x - pos_gt_bboxes[:, 0]
                targets[:, 1, j, i] = pos_gt_bboxes[:, 2] - x
                targets[:, 2, j, i] = y - pos_gt_bboxes[:, 1]
                targets[:, 3, j, i] = pos_gt_bboxes[:, 3] - y

                points[:, 0, j, i] = x
                points[:, 1, j, i] = y
                points[:, 2, j, i] = pos_bboxes[:, 2] - pos_bboxes[:, 0]
                points[:, 3, j, i] = pos_bboxes[:, 3] - pos_bboxes[:, 1]

        return points, targets, masks, instances


    def get_bboxes_avg(self, det_bboxes, D2Det_pred, D2Det_pred_mask, img_meta):
        # TODO: refactoring
        assert det_bboxes.shape[0] == D2Det_pred.shape[0]

        det_bboxes = det_bboxes
        D2Det_pred = D2Det_pred
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]

        map_size = 7
        targets = torch.zeros((det_bboxes.shape[0], 4, map_size, map_size), dtype=torch.float, device=D2Det_pred.device)

        idx = (torch.arange(0, map_size).float() + 0.5).cuda() / map_size

        h = (det_bboxes[:, 3] - det_bboxes[:, 1]).view(-1, 1, 1)
        w = (det_bboxes[:, 2] - det_bboxes[:, 0]).view(-1, 1, 1)
        y = det_bboxes[:, 1].view(-1, 1, 1) + h * idx.view(1, map_size, 1)
        x = det_bboxes[:, 0].view(-1, 1, 1) + w * idx.view(1, 1, map_size)

        targets[:, 0, :, :] = x - D2Det_pred[:, 0, :, :] * w
        targets[:, 2, :, :] = x + D2Det_pred[:, 1, :, :] * w
        targets[:, 1, :, :] = y - D2Det_pred[:, 2, :, :] * h
        targets[:, 3, :, :] = y + D2Det_pred[:, 3, :, :] * h

        targets = targets.permute(0, 2, 3, 1).view(targets.shape[0], -1, 4)
        ious = (D2Det_pred_mask.view(-1, map_size * map_size, 1) > 0.0).float()

        targets = torch.sum(targets * ious, dim=1) / (torch.sum(ious, dim=1) + 0.00001)

        aa = torch.isnan(targets)
        if aa.sum() != 0:
            print('nan error...')

        bbox_res = torch.cat([targets, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_meta[0]['img_shape'][1] - 1)
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_meta[0]['img_shape'][0] - 1)

        return bbox_res

    def get_target_maskiou(self, sampling_results, gt_masks, mask_pred, mask_targets, sample_idx):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (list[ndarray]): Gt masks (the whole instance) of each
                image, binary maps with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        # compute the area ratio of gt areas inside the proposals and
        # the whole instance
        area_ratios = map(self._get_area_ratio, pos_proposals,
                          pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        area_ratios = area_ratios[sample_idx]
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_pred = (mask_pred > 0.5).float()
        mask_pred_areas = mask_pred.sum((-1, -2))

        # mask_pred and mask_targets are binary maps
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (torch.abs(
            mask_pred_areas + gt_full_areas - overlap_areas)+1e-7)
        mask_iou_targets = mask_iou_targets.clamp(min=0)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance"""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances (batch processing for speedup)
            gt_instance_mask_area = gt_masks.sum((-1, -2))
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask[y1:y2 + 1, x1:x2 + 1]

                ratio = gt_mask_in_proposal.sum() / (
                    gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0, ))
        return area_ratios
