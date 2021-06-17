import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import xywh2xyxy, bbox_iou
from functions import build_target


class GroundingFunc:
    def __init__(self, img_size, anchors_full, anchors_imsize):
        self.returns = ["loss", "iou", "accu", "accu_center", "total"]
        self.train = True
        self.img_size = img_size
        self.anchors_full = anchors_full
        self.anchor_imsize = anchors_imsize

    def get_bbox_train(self, batch_size, pred_anchor, best_n_list, gi, gj):
        ## offset eval: if correct with gt center loc
        ## convert offset pred to boxes
        pred_bbox = torch.zeros(batch_size, 4)
        for ii in range(batch_size):
            best_scale_ii = best_n_list[ii]//3
            grid, grid_size = self.img_size//(32//(2**best_scale_ii)), 32//(2**best_scale_ii)
            anchor_idxs = [x + 3*best_scale_ii for x in [0, 1, 2]]
            anchors = [self.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (self.anchor_imsize/grid),
                               x[1] / (self.anchor_imsize/grid)) for x in anchors]
            pred_bbox[ii, 0] = F.sigmoid(
                pred_anchor[best_scale_ii]
                [ii, best_n_list[ii] % 3, 0, gj[ii], gi[ii]]) + gi[ii].float()
            pred_bbox[ii, 1] = F.sigmoid(
                pred_anchor[best_scale_ii]
                [ii, best_n_list[ii] % 3, 1, gj[ii], gi[ii]]) + gj[ii].float()
            pred_bbox[ii, 2] = (torch.exp(pred_anchor[best_scale_ii]
                                          [ii, best_n_list[ii] % 3, 2, gj[ii], gi[ii]]) *
                                scaled_anchors[best_n_list[ii] % 3][0])
            pred_bbox[ii, 3] = (torch.exp(pred_anchor[best_scale_ii]
                                          [ii, best_n_list[ii] % 3, 3, gj[ii], gi[ii]]) *
                                scaled_anchors[best_n_list[ii] % 3][1])
            pred_bbox[ii, :] = pred_bbox[ii, :] * grid_size
        return xywh2xyxy(pred_bbox)

    def get_bbox_eval(self, batch_size, pred_anchor, pred_conf_list,
                      max_loc, max_conf, gi, gj):
        pred_bbox = torch.zeros(batch_size, 4)
        pred_gi, pred_gj, pred_best_n = [], [], []
        for ii in range(batch_size):
            if max_loc[ii] < 3*(self.img_size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(self.img_size//32)**2 + 3*(self.img_size//16)**2:
                best_scale = 1
            else:
                best_scale = 2
            grid, grid_size = self.img_size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0, 1, 2]]
            anchors = [self.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (self.anchor_imsize/grid),
                               x[1] / (self.anchor_imsize/grid)) for x in anchors]
            pred_conf = pred_conf_list[best_scale].view(
                batch_size, 3, grid, grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()
            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii, :, :, :] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)
            pred_bbox[ii, 0] = F.sigmoid(
                pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii, 1] = F.sigmoid(
                pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii, 2] = torch.exp(
                pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii, 3] = torch.exp(
                pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii, :] = pred_bbox[ii, :] * grid_size
        return xywh2xyxy(pred_bbox)

    def __call__(self, batch, criterion, model, optimizer, **kwargs):
        if self.train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        (imgs, word_id, word_mask, target_bbox) = batch
        imgs = model.to_(imgs)
        word_id = model.to_(word_id)
        word_mask = model.to_(word_mask)
        target_bbox = model.to_(target_bbox)
        target_bbox = torch.clamp(target_bbox, min=0, max=self.img_size-1)
        pred_anchor = model(imgs, word_id, word_mask)
        batch_size = imgs.size(0)

        ## flatten anchor dim at each scale
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(
                pred_anchor[ii].size(0), 3, 5, pred_anchor[ii].size(2), pred_anchor[ii].size(3))
        with torch.no_grad():
            ## convert gt box to center+offset format
            gt_param, gi, gj, best_n_list = build_target(self.img_size, self.anchor_imsize,
                                                         self.anchors_full,
                                                         target_bbox,
                                                         pred_anchor)

        ## loss
        loss = criterion(pred_anchor, gt_param, gi, gj, best_n_list)
        if self.train:
            loss.backward()
            optimizer.step()

        ## evaluate if center location is correct
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(
                pred_anchor[ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
            gt_conf_list.append(
                gt_param[ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        accu_center = np.sum(np.array((pred_conf.max(1)[1] == gt_conf.max(1)[1]).cpu(),
                                      dtype=float))
        if self.train:
            pred_bbox = self.get_bbox_train(batch_size, pred_anchor, best_n_list, gi, gj)
        else:
            max_conf, max_loc = torch.max(pred_conf, dim=1)
            pred_bbox = self.get_bbox_eval(batch_size, pred_anchor, pred_conf_list,
                                           max_loc, max_conf, gi, gj)

        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float))

        return {"loss": loss.item(), "iou": torch.mean(iou).item(), "accu": accu,
                "accu_center": accu_center, "total": imgs.size(0)}
