# MSG-VIM inference on video clips with mask guidance
# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F
import random
import utils
from   utils import CONFIG
import networks

random.seed(42)

def tg_re_inference(model, image_dict, post_process=False):
    with torch.no_grad():
        image, tg_mask, re_mask = image_dict['image'], image_dict['tg_mask'], image_dict['re_mask']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda().flatten(0,1)
        tg_mask = tg_mask.cuda().flatten(0,1)
        re_mask = re_mask.cuda().flatten(0,1)
        pred = model(image, tg_mask, re_mask)
        alpha_pred_os1 = pred['alpha_os1'][:,0].unsqueeze(1)
        alpha_pred_os4 = pred['alpha_os4'][:,0].unsqueeze(1)
        alpha_pred_os8 = pred['alpha_os8'][:,0].unsqueeze(1)

        ### refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        h, w = alpha_shape
        T = alpha_pred.shape[0]
        alpha_preds = []
        for i in range(T):
            alpha_pred_f = alpha_pred[i, 0, ...].data.cpu().numpy()
            if post_process:
                alpha_pred_f = utils.postprocess(alpha_pred_f)
            alpha_pred_f = alpha_pred_f * 255
            alpha_pred_f = alpha_pred_f.astype(np.uint8)
            alpha_pred_f = alpha_pred_f[32:h+32, 32:w+32]
            alpha_preds.append(alpha_pred_f)

        return alpha_preds

def generator_tensor_dict(images_path, tg_masks_path, re_masks_path, args):
    # read images
    images = []
    tg_masks = []
    re_masks = []
    sample = {}
    for image_path, tg_mask_path, re_mask_path in zip(images_path, tg_masks_path, re_masks_path):
        image = cv2.imread(image_path)
        tg_mask = cv2.imread(tg_mask_path, 0)
        re_mask = cv2.imread(re_mask_path, 0)

        tg_mask = (tg_mask >= args.guidance_thres).astype(np.float32)
        re_mask = (re_mask >= args.guidance_thres).astype(np.float32) 
        # reshape
        h, w = tg_mask.shape
        sample = {'alpha_shape': tg_mask.shape}
        if h % 32 == 0 and w % 32 == 0:
            padded_image = np.pad(image, ((32,32), (32, 32), (0,0)), mode="reflect")
            padded_tg_mask = np.pad(tg_mask, ((32,32), (32, 32)), mode="reflect")
            padded_re_mask = np.pad(re_mask, ((32,32), (32, 32)), mode="reflect")
        else:
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w
            padded_image = np.pad(image, ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
            padded_tg_mask = np.pad(tg_mask, ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
            padded_re_mask = np.pad(re_mask, ((32,pad_h+32), (32, pad_w+32)), mode="reflect")

        # ImageNet mean & std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        # convert BGR images to RGB
        image, tg_mask, re_mask = padded_image[:,:,::-1], padded_tg_mask, padded_re_mask
        # swap color axis
        image = image.transpose((2, 0, 1)).astype(np.float32)

        tg_mask = np.expand_dims(tg_mask.astype(np.float32), axis=0)
        re_mask = np.expand_dims(re_mask.astype(np.float32), axis=0)

        # normalize image
        image /= 255.

        # to tensor
        images.append(torch.from_numpy(image).sub_(mean).div_(std).unsqueeze(0).unsqueeze(1))
        tg_masks.append(torch.from_numpy(tg_mask).unsqueeze(0).unsqueeze(1))
        re_masks.append(torch.from_numpy(re_mask).unsqueeze(0).unsqueeze(1))
    
    sample['image'], sample['tg_mask'], sample['re_mask'] = torch.cat(images, 1), torch.cat(tg_masks, 1), torch.cat(re_masks, 1)
    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/VIM.toml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/msgvim.pth', help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='~/data/VIM50/', help="input image dir")
    parser.add_argument('--tg-mask-dir', type=str, default='~/tg_masks/', help="tg mask dir")
    parser.add_argument('--re-mask-dir', type=str, default='~/re_masks/', help="refer mask dir")
    parser.add_argument('--image-ext', type=str, default='.png', help="input image ext")
    parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
    parser.add_argument('--output', type=str, default='outputs/', help="output dir")
    parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
    parser.add_argument('--num-frames', type=int, default=10, help="num of frames for inference")
    parser.add_argument('--post-process', action='store_true', default=False, help='post process to keep the largest connected component')

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")
    
    clip_paths = []
    for clip in sorted(os.listdir(args.image_dir)):
        if os.path.isdir(os.path.join(args.image_dir, clip)):
            clip_path = os.path.join(args.image_dir, clip)
            clip_paths.append(clip_path)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder,
                                    dec_T=args.num_frames, dec_B=1)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    for single_clip_path in clip_paths:
        print('processing %s\n'%(single_clip_path))
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    os.listdir(os.path.join(single_clip_path,'com'))),
            key=lambda x: int(x.split('.')[0]))
        
        single_output_path = os.path.join(out_path, single_clip_path.split('/')[-1])
        os.makedirs(single_output_path, exist_ok=True)
        
        images = []
        tg_masks = []
        re_masks = []
        image_names = []
        tg_mask_path = os.path.join(args.tg_mask_dir, single_clip_path.split('/')[-1])
        re_mask_path = os.path.join(args.re_mask_dir, single_clip_path.split('/')[-1])
        for ins_mask in sorted(os.listdir(tg_mask_path)):
            ins_mask_path_ouput = os.path.join(single_output_path, ins_mask)
            os.makedirs(ins_mask_path_ouput, exist_ok=True)
            for image_name in imgs:
                image_path = os.path.join(single_clip_path, 'com', image_name)
                tg_ins_mask_path = os.path.join(tg_mask_path, ins_mask, image_name)
                re_ins_mask_path = os.path.join(re_mask_path, ins_mask, image_name)
                images.append(image_path)
                tg_masks.append(tg_ins_mask_path)
                re_masks.append(re_ins_mask_path)
                image_names.append(image_name)
                if len(images) == args.num_frames:
                    image_dict = generator_tensor_dict(images, tg_masks, re_masks, args)
                    alpha_preds = tg_re_inference(model, image_dict, post_process=args.post_process)
                    for i, image_name_out in enumerate(image_names): 
                        cv2.imwrite(os.path.join(ins_mask_path_ouput, image_name_out), alpha_preds[i])
                    images = []
                    tg_masks = []
                    re_masks = []
                    image_names = []
            
