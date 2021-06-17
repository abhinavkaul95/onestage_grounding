import numpy as np
import torch
from utils.utils import bbox_iou


def build_target(img_size, anchor_imsize, anchors_full, raw_coord, pred):
    coord_list, bbox_list = [], []
    for scale_ii in range(len(pred)):
        coord = torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda()
        batch, grid = raw_coord.size(0), img_size//(32//(2**scale_ii))
        coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2])/(2*img_size)
        coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3])/(2*img_size)
        coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0])/(img_size)
        coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1])/(img_size)
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))
    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch):
        anch_ious = []
        for scale_ii in range(len(pred)):
            batch, grid = raw_coord.size(0), img_size//(32//(2**scale_ii))
            gi = coord_list[scale_ii][ii, 0].long()
            gj = coord_list[scale_ii][ii, 1].long()
            tx = coord_list[scale_ii][ii, 0] - gi.float()
            ty = coord_list[scale_ii][ii, 1] - gj.float()
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]
            anchor_idxs = [x + 3*scale_ii for x in [0, 1, 2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (anchor_imsize/grid),
                               x[1] / (anchor_imsize/grid)) for x in anchors]
            ## Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw.item(), gh.item()])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate(
                (np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n//3
        batch, grid = raw_coord.size(0), img_size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0, 1, 2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (anchor_imsize/grid),
                           x[1] / (anchor_imsize/grid)) for x in anchors]
        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)
        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)
    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].cuda()
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=size_average)
    celoss = torch.nn.CrossEntropyLoss(size_average=size_average)
    batch = input[0].size(0)

    pred_bbox = torch.zeros(batch, 4).cuda()
    gt_bbox = torch.zeros(batch, 4).cuda()
    for ii in range(batch):
        pred_bbox[ii, 0:2] = torch.sigmoid(
            input[best_n_list[ii]//3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = input[best_n_list[ii]//3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])
    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(len(input)):
        pred_conf_list.append(input[scale_ii][:, :, 4, :, :].contiguous().view(batch, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x+loss_y+loss_w+loss_h)*w_coord + loss_conf
