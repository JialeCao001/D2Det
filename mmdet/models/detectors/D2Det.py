import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multiclass_nms1, bbox2roi_expand
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector
import numpy as np
import mmcv
import pycocotools.mask as mask_util

@DETECTORS.register_module
class D2Det(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 reg_roi_extractor,
                 D2Det_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        assert D2Det_head is not None
        super(D2Det, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if reg_roi_extractor is not None:
            self.reg_roi_extractor = builder.build_roi_extractor(
                reg_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.reg_roi_extractor = self.bbox_roi_extractor
        self.D2Det_head = builder.build_head(D2Det_head)

        self.loss_roi_reg = builder.build_loss(dict(type='IoULoss', loss_weight=2.0))
        self.loss_roi_mask = builder.build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.MASK_ON = D2Det_head.MASK_ON
        if self.MASK_ON:
            self.loss_roi_instance = builder.build_loss(dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
            self.loss_iou = builder.build_loss(dict(type='MSELoss', loss_weight=0.5))

        self.init_extra_weights()

    def init_extra_weights(self):
        self.D2Det_head.init_weights()
        if not self.share_roi_extractor:
            self.reg_roi_extractor.init_weights()

    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_bboxes = new_bboxes
        return sampling_results


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_bbox:
            # assign gts and sample proposals
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

            #####dense local regression head ################################
            sampling_results = self._random_jitter(sampling_results, img_meta)
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

            reg_feats = self.reg_roi_extractor(
                x[:self.reg_roi_extractor.num_inputs], pos_rois)


            if self.with_shared_head:
                reg_feats = self.shared_head(reg_feats)
            # Accelerate training
            max_sample_num_reg = self.train_cfg.rcnn.get('max_num_reg', 192)
            sample_idx = torch.randperm(
                reg_feats.shape[0])[:min(reg_feats.shape[0], max_sample_num_reg)]
            reg_feats = reg_feats[sample_idx]
            pos_gt_labels = torch.cat([
                res.pos_gt_labels for res in sampling_results
            ])
            pos_gt_labels = pos_gt_labels[sample_idx]
            if self.MASK_ON == False:
                #####################instance segmentation############################
                reg_pred, reg_masks_pred = self.D2Det_head(reg_feats)
                reg_points, reg_targets, reg_masks = self.D2Det_head.get_target(sampling_results)
                reg_targets = reg_targets[sample_idx]
                reg_points = reg_points[sample_idx]
                reg_masks = reg_masks[sample_idx]
                x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :]*reg_points[:, 2, :, :]
                x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :]*reg_points[:, 2, :, :]
                y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :]*reg_points[:, 3, :, :]
                y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :]*reg_points[:, 3, :, :]

                pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)

                x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
                x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
                y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
                y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

                pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)
                loss_reg = self.loss_roi_reg(
                    pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    weight=reg_masks.reshape(-1, 1))

                loss_mask = self.loss_roi_mask(
                    reg_masks_pred.reshape(-1,  reg_masks.shape[2]*reg_masks.shape[3]),
                    reg_masks.reshape(-1,  reg_masks.shape[2]*reg_masks.shape[3]))
                losses.update(dict(loss_reg=loss_reg, loss_mask = loss_mask))
                #############################################
            else:
                #####################object detection############################
                reg_pred, reg_masks_pred, reg_instances_pred, reg_iou = self.D2Det_head(reg_feats, pos_gt_labels)
                reg_points, reg_targets, reg_masks, reg_instances = self.D2Det_head.get_target_mask(sampling_results, gt_masks, self.train_cfg.rcnn)

                reg_targets = reg_targets[sample_idx]
                reg_points = reg_points[sample_idx]
                reg_masks = reg_masks[sample_idx]
                reg_instances = reg_instances[sample_idx]

                x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :]*reg_points[:, 2, :, :]
                x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :]*reg_points[:, 2, :, :]
                y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :]*reg_points[:, 3, :, :]
                y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :]*reg_points[:, 3, :, :]

                pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)

                x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
                x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
                y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
                y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

                pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)
                loss_reg = self.loss_roi_reg(
                    pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    weight=reg_masks.reshape(-1, 1))
                loss_mask = self.loss_roi_mask(
                    reg_masks_pred.reshape(-1,  reg_masks.shape[1]*reg_masks.shape[2]),
                    reg_masks.reshape(-1,  reg_masks.shape[1]*reg_masks.shape[2]))

                loss_instance = self.loss_roi_instance(reg_instances_pred, reg_instances, pos_gt_labels)
                reg_iou_targets = self.D2Det_head.get_target_maskiou(sampling_results, gt_masks, reg_instances_pred[pos_gt_labels>0, pos_gt_labels], reg_instances, sample_idx)
                reg_iou_weights = ((reg_iou_targets > 0.1) & (reg_iou_targets <=1.0)).float()
                loss_iou = self.loss_iou(
                    reg_iou[pos_gt_labels>0,pos_gt_labels],
                    reg_iou_targets,
                    weight=reg_iou_weights)

                losses.update(dict(loss_reg=loss_reg,loss_mask=loss_mask,loss_instance=loss_instance,loss_iou=loss_iou))
                #############################################


        return losses


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=False)


        if det_bboxes.shape[0] != 0:
            reg_rois = bbox2roi([det_bboxes[:, :4]])
            reg_feats = self.reg_roi_extractor(
                x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
            self.D2Det_head.test_mode = True

            ####dense local regression predection############
            reg_pred, reg_pred_mask = self.D2Det_head(reg_feats)
            det_bboxes = self.D2Det_head.get_bboxes_avg(det_bboxes,
                                                   reg_pred,
                                                   reg_pred_mask,
                                                   img_meta)

            if self.MASK_ON:
                reg_rois = bbox2roi([det_bboxes[:, :4]])
                reg_feats = self.reg_roi_extractor(
                    x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
                reg_pred, reg_pred_mask, reg_pred_instance, reg_iou = self.D2Det_head(reg_feats, det_labels + 1)
                mask_scores = self.get_mask_scores(reg_iou, det_bboxes, det_labels)
                segm_result = self.get_seg_masks(
                    reg_pred_instance, det_bboxes[:, :4], det_labels, img_meta[0]['ori_shape'],
                    img_meta[0]['scale_factor'], rescale=rescale)
            else:
                det_bboxes, det_labels = multiclass_nms1(det_bboxes[:, :4], det_bboxes[:, 4], det_labels + 1,
                                                         0.0, dict(type='soft_nms', iou_thr=0.5), 100)
            
            if rescale:
                det_bboxes[:, :4] /= img_meta[0]['scale_factor']
        else:
            det_bboxes = torch.Tensor([])
            segm_result = [[] for _ in range(81 - 1)]
            mask_scores = [[] for _ in range(81 - 1)]

        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        if self.MASK_ON:
            return bbox_results, (segm_result, mask_scores)
        return bbox_results

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(80)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            bbox[0] = max(bbox[0], 0)
            bbox[1] = max(bbox[1], 0)
            bbox[2] = min(bbox[2], img_w-1)
            bbox[3] = min(bbox[3], img_h-1)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not None:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > 0.5).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms
    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = 0.3*mask_iou_pred[inds, det_labels +
                                    1] +det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [
            mask_scores[det_labels == i] for i in range(81 - 1)
        ]
