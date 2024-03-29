import argparse
import logging
import os
import random
from functools import partial

import nibabel as nib
import numpy as np
import surface_distance
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss
from surface_distance import metrics

from dataset.datasets import load_data_volume
from modeling.efficient_3dsam.efficient_3dsam_encoder import ImageEncoderViT_3d
from modeling.SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from modeling.SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
from utils.util import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["bas", "atm", "parse", "imagecas"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--num_prompts",
        default=1,
        type=int,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--split_model", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    seed_value = args.seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["bas", "atm", "parse", "imagecas"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=args.num_worker,
    )
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=192,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16,
    )
    img_encoder.load_state_dict(
        torch.load(os.path.join(args.snapshot_path, file), map_location="cpu")[
            "encoder_dict"
        ],
        strict=True,
    )

    if args.split_model == 0:
        img_encoder.to(device)
    elif args.split_model == 1:
        img_encoder.to("cuda:0")
        for i in img_encoder.blocks[6:12]:
            i.to("cuda:1")
    else:
        raise ValueError("split_model should be 0 or 1")

    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8
            )
        )
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.snapshot_path, file), map_location="cpu")[
                "feature_dict"
            ][i],
            strict=True,
        )
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead(img_size=96).to(device)
    mask_decoder.load_state_dict(
        torch.load(os.path.join(args.snapshot_path, file), map_location="cpu")[
            "decoder_dict"
        ],
        strict=True,
    )
    mask_decoder.to(device)

    dice_loss = DiceLoss(
        include_background=False, softmax=False, to_onehot_y=True, reduction="none"
    )
    img_encoder.eval()
    for i in prompt_encoder_list:
        i.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, prompt, img_encoder, prompt_encoder, mask_decoder):
        out = F.interpolate(
            img.float(), scale_factor=512 / patch_size, mode="trilinear"
        )
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        # feature_list = feature_list[::-1]
        points_torch = prompt.transpose(0, 1)
        new_feature = []
        for i, (feature, feature_decoder) in enumerate(
            zip(feature_list, prompt_encoder)
        ):
            if i == 3:
                new_feature.append(
                    feature_decoder(
                        feature.to(device),
                        points_torch.clone(),
                        [patch_size, patch_size, patch_size],
                    )
                )
            else:
                new_feature.append(feature.to(device))
        img_resize = F.interpolate(
            img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device),
            scale_factor=64 / patch_size,
            mode="trilinear",
        )
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size // 64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks

    if not os.path.exists(os.path.join(args.snapshot_path, "preds")):
        os.mkdir(os.path.join(args.snapshot_path, "preds"))

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for idx, (img, seg, spacing) in enumerate(test_data):
            seg = seg.float()
            prompt = F.interpolate(
                seg[None, :, :, :, :], img.shape[2:], mode="nearest"
            )[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            seg_pred = torch.zeros_like(prompt).to(device)
            l = len(torch.where(prompt == 1)[0])
            # np.random.seed(0)
            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            # sample = sample[:3]
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size // 2
            d_max = x_m + patch_size // 2
            h_min = z_m - patch_size // 2
            h_max = z_m + patch_size // 2
            w_min = y_m - patch_size // 2
            w_max = y_m + patch_size // 2
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])

            points = (
                torch.cat([x - d_min, y - w_min, z - h_min], dim=1).unsqueeze(1).float()
            )
            points_torch = points.to(device)
            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            pred = model_predict(
                img_patch, points_torch, img_encoder, prompt_encoder_list, mask_decoder
            )
            pred = pred[
                :,
                :,
                d_l : patch_size - d_r,
                h_l : patch_size - h_r,
                w_l : patch_size - w_r,
            ]
            pred = F.softmax(pred, dim=1)[:, 1]
            seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

            final_pred = F.interpolate(
                seg_pred.unsqueeze(1), size=seg.shape[2:], mode="trilinear"
            )
            masks = final_pred > 0.5
            # 创建新的Nifti1Image对象
            new_img = nib.Nifti1Image(masks.int().cpu().numpy()[0][0], np.eye(4))

            # 保存新的图像
            save_name = os.path.basename(test_data.dataset.img_dict[idx]).replace(
                "_0000", ""
            )
            nib.save(new_img, os.path.join(args.snapshot_path, "preds", save_name))

            loss = 1 - dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())

            ssd = surface_distance.compute_surface_distances(
                (seg == 1)[0, 0].cpu().numpy(),
                (masks == 1)[0, 0].cpu().numpy(),
                spacing_mm=spacing[0].numpy(),
            )
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
            loss_nsd.append(nsd)
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                )
            )
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()
