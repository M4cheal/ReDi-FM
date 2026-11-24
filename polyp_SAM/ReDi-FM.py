import os
import random
import torch
import numpy as np
import argparse, sys, datetime
from utils.config import Logger, Seg_loss
from torch.autograd import Variable
from utils.metrics_polyp import calculate_metrics
from networks.segformer.segformer import SegFormer
from torch.utils.data import DataLoader
from dataloaders.POLYP_dataloader import POLYP_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import save_code
from copy import deepcopy
from dataloaders.normalize import normalize_image_to_0_1
from utils.utils import sigmoid_entropy
from skimage.measure import label
import cv2
import math

import torchvision.transforms as transforms
from dataloaders import my_transforms as my_transforms
from PIL import Image

torch.set_num_threads(1)

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

@torch.jit.script
def softmax_entropy(x, x_ema, weight=None):
    x_prob = x.sigmoid()
    x_ema_prob = x_ema.sigmoid()
    if weight is None:
        loss = (- x_prob * torch.log(x_ema_prob + 1e-8)).sum(1)
    else:
        loss = ((- x_prob * torch.log(x_ema_prob + 1e-8))*weight).sum(1)
        
    return loss

def bce_weight(soft_label):
    soft_label = torch.sigmoid(soft_label)
    weight = torch.zeros_like(soft_label).cuda()
    for cls in range(soft_label.size(1)):
        prob = soft_label[:,cls,:,:]
        #initialize Gaussian mean and variance
        u1 = 0.5  
        sigma1 = 0.1  
        left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
        right = np.exp(-(prob.detach().cpu().numpy() - u1)**2 / (2 * sigma1))
        weight_numpy =  1.3 - left*right
        weight[:,cls,:,:] = torch.from_numpy(weight_numpy).cuda()
    return weight

class ReDiFM:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, config.exp_name, config.Source_Dataset)
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        save_code(config.work_root,os.path.join(log_root, time_now, 'code'))
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        self.img_save_path = os.path.join(log_root, time_now, 'mask')
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        self.dataset_root = config.dataset_root

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')

        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = POLYP_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, is_ori=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        # SAM
        self.sam_root = config.sam_root
        self.sam_loss_weight = config.sam_loss_weight
        self.loss_bate_soft = config.loss_bate_soft

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)
        
        # Loss function
        self.lossmap = config.lossmap
        self.seg_cost = Seg_loss(self.lossmap)

        # GPU
        self.device = config.device

        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        self.transform = self.get_tta_transforms()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        self.model = SegFormer(num_classes=self.out_ch, phi=self.backbone, pretrained=False).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-segformer.pth'))
        self.model.load_state_dict(checkpoint, strict=True)
        self.t_model = deepcopy(self.model)
        
        self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_root).to(self.device)
        self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

    def get_tta_transforms(self, gaussian_std: float=0.005, soft=False, clip_inputs=False):
        img_shape = (self.image_size, self.image_size, 3)
        n_pixels = img_shape[0]

        clip_min, clip_max = 0.0, 1.0

        p_hflip = 0.5
    
        tta_transforms = transforms.Compose([
            my_transforms.Clip(0.0, 1.0), 
            my_transforms.ColorJitterPro(
                brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                gamma=[0.85, 1.15] if soft else [0.7, 1.3]
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            my_transforms.Clip(clip_min, clip_max)
        ])
        return tta_transforms
    
    def logit2mask(self, pred_logit, threshold=0.5, on_hot=True):
        pred = torch.sigmoid(pred_logit.clone())
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        if not on_hot:
            pseudo_labels = torch.zeros([pred.size()[0], pred.size()[2], pred.size()[3]])
            if torch.cuda.is_available():
                pseudo_labels = pseudo_labels.cuda()
            pseudo_labels[pred[:, 0] == 1] = 1

            return pseudo_labels.unsqueeze(dim=1)

        return pred
    
    def get_entropy(self, pred_logit):
        entropy = torch.sigmoid(pred_logit)
        entropy = sigmoid_entropy(entropy)
        entropy[torch.isnan(entropy)] = 0
        entropy = ((entropy - entropy.min()) * (1 / (entropy.max() - entropy.min())))
        return entropy
        
    def dice_metric(self, pred, label):
        pred = np.array(pred).astype(np.uint8)
        label = np.array(label).astype(np.uint8)
        batch_size = pred.shape[0]
        dices = []
        smooth = 1e-6

        for batch in range(batch_size):
            intersection = (pred[batch][0] * label[batch][0]).sum()
            dice = (2 * intersection + smooth) / (pred[batch][0].sum() + label[batch][0].sum() + smooth)

            dices.append(dice*100.)

        return dices
    
    def get_box(self, pred_logit, threshold=0.5):
        pred = np.array(torch.sigmoid(pred_logit).detach().cpu())
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        bboxs = []
        empty_index = []
        for i in range(pred.shape[1]):
            non_zero_pixels = np.where(pred[0][i])
            if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
                x_min = np.min(non_zero_pixels[1])
                y_min = np.min(non_zero_pixels[0])
                x_max = np.max(non_zero_pixels[1])
                y_max = np.max(non_zero_pixels[0])

                bbox = np.array([x_min, y_min, x_max, y_max])
                bboxs.append(bbox)
            else:
                bboxs.append(None)
                empty_index.append(i)

        return bboxs, empty_index

    def get_point(self, pred_t, pred_logit_t,label,mask_inputs):
        sam_t_1_hard = pred_t.int()
        center = mask_inputs[sam_t_1_hard==1].mean().cpu().numpy()
        num_samples = 1
        indices_1_hard = np.argwhere(sam_t_1_hard.cpu().numpy()[0][0] == 1)
        if len(indices_1_hard) == 0:
            input_point_1_hard = np.empty((0, 2), dtype=int)
            input_label_1_hard = np.empty((0,), dtype=int)
        else:
            mask_values_hard = mask_inputs[sam_t_1_hard == 1].view(-1).cpu().numpy()
            differences = np.abs(mask_values_hard - center)
            num_hard_samples = min(num_samples, len(differences))
            sorted_indices = np.argsort(-differences)
            top_hard_indices = sorted_indices[:num_hard_samples]
            input_point_1_hard = indices_1_hard[:, [1, 0]][top_hard_indices]
            input_label_1_hard = np.ones(len(input_point_1_hard), dtype=int)

        input_point_1_hard = np.expand_dims(input_point_1_hard,axis=0)
        input_label_1_hard = np.expand_dims(input_label_1_hard,axis=0)
        input_point_1_hard = torch.tensor(input_point_1_hard).to(pred_t.device)
        input_label_1_hard = torch.tensor(input_label_1_hard).to(pred_t.device)
        points_prompt = (input_point_1_hard, input_label_1_hard)
        return points_prompt
    
    def logit2mask_save(self, pred_logit, threshold=0.5, on_hot=True):
        pred = torch.sigmoid(pred_logit.clone())
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        if not on_hot:
            pseudo_labels = torch.zeros([pred.size()[0], pred.size()[2], pred.size()[3]])
            if torch.cuda.is_available():
                pseudo_labels = pseudo_labels.cuda()
            pseudo_labels[pred[:, 0] == 1] = 255

            return pseudo_labels.unsqueeze(dim=1)

        return pred
    
    def prompt_generate_random_fast(self, coarse_mask, img_size, israndom = False):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape
        coarse_mask_np = coarse_mask.detach().cpu().numpy()

        # points: BxNx2 tensor & boxes
        points_prompt = np.zeros([b, num_class, 2])
        points_label = np.zeros([b, num_class])
        points_prompt_random = np.zeros([b, num_class, 2])
        for idx in range(b):  # iterate over each image
            # for cls in num_class_list: # find points for each class
            for cls in range(num_class): # find points for each class
                # obtain the binary mask
                # mask_cls = (coarse_mask_np[idx] == cls).astype(np.uint8)
                mask_cls = coarse_mask_np[idx][cls].astype(np.uint8)
                if mask_cls.max() > 0:
                    
                    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
                    ratio_list, regionid_list = [], []
                    for region_id in range(1, region_ids+1):
                        #find coordinates of points in the region
                        binary_msk = np.where(label_msk==region_id, 1, 0)

                        # clean some region that is abnormally small
                        r = np.sum(binary_msk) / np.sum(mask_cls)
                        # print('curr mask over all mask ratio', r)
                        ratio_list.append(r)
                        regionid_list.append(region_id)

                    ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
                    regionid_list = regionid_list[::-1]

                    binary_msk = np.where(label_msk==regionid_list[0], 1, 0)
                    
                    if israndom:  
                        cY_r, cX_r = np.where(binary_msk==1)
                        random_idx = np.random.randint(0, len(cX_r))
                        points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = int(cX_r[random_idx]), int(cY_r[random_idx])

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
                    dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                    cY, cX = np.where(dist_img==dist_img.max())
                    random_idx = np.random.randint(0, len(cX))
                    points_prompt[idx,cls,0], points_prompt[idx,cls,1] = int(cX[random_idx]), int(cY[random_idx])
                    points_label[idx,cls] = cls+1

        points_prompt = torch.tensor(points_prompt).to(coarse_mask.device)
        points_label = torch.tensor(points_label).to(coarse_mask.device)
        points_prompt = (points_prompt, points_label)

        if israndom:  
            points_prompt_random = torch.tensor(points_prompt_random).to(coarse_mask.device)
            points_prompt_random = (points_prompt_random, points_label)

            return points_prompt, points_prompt_random, None, None

        return points_prompt, None, None

    def get_channel_weights(self, pred, target,cls_mask,eps=1e-6):
        n_channels = target.size(1)
        weights = torch.zeros((n_channels,), device=target.device)
        for ch in range(n_channels):
            fg_count = (target[:, ch] == 1).sum().float()
            bg_count = (target[:, ch] == 0).sum().float()
            w = fg_count / (bg_count+fg_count)

            fn_fg_count = ((target[:, ch] ==1) & (target[:, ch] != pred[:, ch])).sum().float()
            fn_bg_count = (target[:, ch] == 0).sum().float()
            fn_w = fn_fg_count / (bg_count+fg_count)
            w_fg = (w+fn_w)/2
            weights[ch] = 1-w_fg
        weights = weights.view(1, target.size(1), 1, 1)
        weight_map = torch.where(cls_mask == 1, weights, 1 - weights)
        return weight_map
    
    def sig_label_to_hard_cls(self, sig_pls, pseudo_label_s, en_label_threshold=0.5):
        prediction_entropy = sig_pls.clone()
        prediction_entropy = self.get_entropy(prediction_entropy)
        high_entropy = prediction_entropy.clone()
        high_entropy[high_entropy >= en_label_threshold] = 1
        high_entropy[high_entropy < en_label_threshold] = 0

        pseudo_label = pseudo_label_s.int() | high_entropy.int()
        pseudo_label = pseudo_label.float()
        return pseudo_label

    def run(self, epoch):
        metrics_test = [[], [], []]
        metric_dict = ['Dice', 'Enhanced_Align', 'Structure_Measure']

        for batch, data in enumerate(self.target_test_loader):
            x, y, path, x_ori = data

            x_ori, x, y = Variable(x_ori).to(self.device), Variable(x).to(self.device), Variable(y).to(self.device)

            for tr_iter in range(self.iters):
                self.t_model.eval()
                self.model.train()

                with torch.no_grad():
                    pred_logit_t = self.t_model(x)
                    weight = bce_weight(pred_logit_t.detach().clone())
                    pred_logit_t_con = pred_logit_t.detach().clone()
                    pred_t = self.logit2mask(pred_logit_t.detach(), threshold=0.5)
                    pred_t_con = pred_t.clone()
                    pred_t_cls_mask = self.sig_label_to_hard_cls(pred_logit_t.detach(), pred_t.detach())
                    bboxes, empty_index = self.get_box(pred_logit_t)
                    
                    seg_output_list = []
                    seg_logit_list = []
                    for class_index in range(pred_t.size(1)):
                        if class_index not in empty_index:
                            points_prompt_center, points_prompt_random, box_prompt, mask_prompt = self.prompt_generate_random_fast(pred_t[:,class_index].unsqueeze(1), self.image_size, True)
                            points_prompt_hard = self.get_point(pred_t[:,class_index].unsqueeze(1), pred_logit_t[:,class_index].unsqueeze(1),y[:,class_index].unsqueeze(1),x_ori.mean(1).unsqueeze(1).clone())
                            
                            points_prompt = (torch.cat([points_prompt_center[0], points_prompt_hard[0]],dim=1),
                                             torch.cat([points_prompt_center[1], points_prompt_hard[1]],dim=1))

                            batched_input = [{
                                'image': self.resize_transform.apply_image_torch(x_ori)[0],
                                'point_coords': self.resize_transform.apply_coords_torch(torch.from_numpy(np.array(points_prompt[0].cpu())), x_ori.shape[2:]).to(self.device),
                                'point_labels': torch.from_numpy(np.array(points_prompt[1].cpu())).to(self.device),
                                'original_size': x_ori.shape[2:],
                            }]
                            
                            pred_logit_sam = self.sam(batched_input, multimask_output=True)
                            pred_logit_sam_mask = pred_logit_sam[0]["masks"].transpose(0,1).detach()
                            
                            sam_iou_pred = pred_logit_sam[0]["iou_predictions"].detach().cpu().view(-1).tolist()

                            dice_sam_pred_t = self.dice_metric(self.logit2mask(pred_logit_sam_mask, threshold=0.5).detach().cpu(),
                                                               pred_t.detach().cpu().repeat(pred_logit_sam_mask.size(0), 1, 1, 1)[:,class_index].unsqueeze(1))
                            
                            sam_pred_index = dice_sam_pred_t.index(max(dice_sam_pred_t))
                            pred_logit_sam_mask_select = pred_logit_sam_mask[sam_pred_index].unsqueeze(0)
                            seg_logit_list.append(pred_logit_sam_mask_select.clone())
                            pred_logit_sam_mask_select = self.logit2mask(pred_logit_sam_mask_select.detach(), threshold=0.5)
                            seg_output_list.append(pred_logit_sam_mask_select.clone())
                        else:
                            seg_logit_list.append(pred_logit_t[:,class_index].unsqueeze(1).clone())
                            seg_output_list.append(pred_t[:,class_index].unsqueeze(1).clone())

                    seg_logit = torch.cat(seg_logit_list, dim=1)
                    seg_output = torch.cat(seg_output_list, dim=1)
                    sam_weight = bce_weight(seg_logit)

                    pred_bin = pred_t_con.float()
                    sam_bin = seg_output.float()
                    inter = (pred_bin * sam_bin).sum(dim=(1, 2, 3))
                    union = pred_bin.sum(dim=(1, 2, 3)) + sam_bin.sum(dim=(1, 2, 3))
                    dice_agree = (2 * inter + 1e-6) / (union + 1e-6)   # shape: [B]
                    r_gating = dice_agree.clamp(0.0, 1.0)              # Tensor on device

                pred_logit = self.model(self.transform(x))
                pred = self.logit2mask(pred_logit.detach(), threshold=0.5)
                weight_cls_sam = self.get_channel_weights(pred.detach().clone(), seg_output.detach().clone(), pred_t_cls_mask.detach().clone())

                g_map = r_gating.view(-1, 1, 1, 1)              # [B,1,1,1]
                one_map = torch.ones_like(weight_cls_sam)
                weight_cls_sam_gated = g_map * weight_cls_sam + (1.0 - g_map) * one_map

                loss_soft = self.sam_loss_weight * (
                    softmax_entropy(pred_logit, seg_logit, weight_cls_sam_gated)
                ).mean()
                g_scalar = r_gating.mean()   # [B] → scalar
                loss_hard_sam = self.sam_loss_weight * self.seg_cost(pred_logit, seg_output, sam_weight)
                loss_hard_src = self.seg_cost(pred_logit, pred_t_con, weight)
                loss_hard = loss_hard_sam + g_scalar * loss_hard_src

                loss = self.loss_bate_soft*loss_soft + loss_hard

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Inference
            self.model.eval()
            with torch.no_grad():
                pred_logit = self.model(x)
                
            save_pred = self.logit2mask_save(pred_logit.detach().clone(), threshold=0.5, on_hot=False)[0][0].cpu().numpy()
            save_image = Image.fromarray(save_pred.astype(np.uint8))
            save_path = os.path.basename(path[0]).replace(os.path.splitext(path[0])[1], '_'+str(epoch)+os.path.splitext(path[0])[1])
            domain_path = os.path.join(self.img_save_path, path[0].split(self.dataset_root)[-1].split('/')[1])
            if not os.path.exists(domain_path):
                os.makedirs(domain_path) 
            save_image.save(os.path.join(domain_path,save_path))

            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='BKAI',
                        help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--Target_Dataset', type=list)
    parser.add_argument('--exp_name', type=str, default='debug')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=352)

    # Model
    parser.add_argument('--backbone', type=str, default='b0', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Loss function
    parser.add_argument('--lossmap', type=str, default=['dice', 'bce'])

    # Path
    parser.add_argument('--path_save_log', type=str, default='/root/autodl-tmp/save/logs_sam/polyp')
    parser.add_argument('--model_root', type=str, default='/root/autodl-tmp/save/models_sam/polyp_b0')
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/dataset/polyp')
    parser.add_argument('--work_root', type=str, default='/root/code/ReDi-FM/polyp_SAM')
    
    # SAM 
    parser.add_argument('--sam_root', type=str, default='/root/autodl-tmp/save/models/pretrained/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_loss_weight', type=float, default=1.0)
    parser.add_argument('--loss_bate_soft', type=float, default=1.0)

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    config = parser.parse_args()

    config.Target_Dataset = ['BKAI', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']
    config.Target_Dataset.remove(config.Source_Dataset)

        # set random seeds
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)


    TTA = ReDiFM(config)
    for i in range(1):
        TTA.run(i)
    # TTA.run()
