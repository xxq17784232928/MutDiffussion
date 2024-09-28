import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import random

def parse_args(): 
    parser = argparse.ArgumentParser(description="")

    # conditions
    parser.add_argument("--mask_path", type=str, default="test_data/512_masks/27007.png", help="path to the segmentation mask")

    # paths
    parser.add_argument("--config_path", type=str, default="configs/512_mask.yaml", help="path to model config")
    parser.add_argument("--ckpt_path", type=str, default="pretrained/512_mask.ckpt", help="path to model checkpoint")
    parser.add_argument("--save_folder", type=str, default="outputs/512_mask2image", help="folder to save synthesis outputs")

    # batch size and ddim steps
    parser.add_argument("--batch_size", type=int, default=4, help="number of images to generate")
    parser.add_argument("--ddim_steps", type=int, default="50", help="number of ddim steps (between 20 to 1000, the larger the slower but better quality)")

    # whether save intermediate outputs
    parser.add_argument("--save_z", type=bool, default=False, help="whether visualize the VAE latent codes and save them in the output folder")
    parser.add_argument("--return_influence_function", type=bool, default=False, help="whether visualize the Influence Functions and save them in the output folder")
    parser.add_argument("--display_x_inter", type=bool, default=False, help="whether display the intermediate DDIM outputs (x_t and pred_x_0) and save them in the output folder")
    parser.add_argument("--save_mixed", type=bool, default=False, help="whether overlay the segmentation mask on the synthesized image to visualize mask consistency")
   
    parser.add_argument("--tensorboard_logdir", type=str, default="runs/generate_256_masks_29980", help="Directory to save TensorBoard logs")

    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    args = parse_args()

    # 设置随机种子，确保生成的一致性
    set_seed(42)  # 你可以根据需要选择一个特定的种子值

    # ========== TensorBoard Setup ==========
    writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    # ========== set up model ==========
    print(f'Set up model')
    config = OmegaConf.load(args.config_path)
    model_config = config['model']
    model = instantiate_from_config(model_config)
    model.init_from_ckpt(args.ckpt_path)
    model = model.cuda()
    model.eval()

    print(model)

    # ========== set output directory ==========
    os.makedirs(args.save_folder, exist_ok=True)

    # ========== prepare seg mask for model ==========
    with open(args.mask_path, 'rb') as f:
        img = Image.open(f)
        resized_img = img.resize((32, 32), Image.NEAREST)  # resize
        flattened_img = list(resized_img.getdata())
    flattened_img_tensor = torch.tensor(flattened_img)  # flatten
    flattened_img_tensor_one_hot = F.one_hot(flattened_img_tensor, num_classes=19)  # one hot
    flattened_img_tensor_one_hot_transpose = flattened_img_tensor_one_hot.transpose(0, 1)
    flattened_img_tensor_one_hot_transpose = torch.unsqueeze(flattened_img_tensor_one_hot_transpose,0).cuda()  # add batch dimension

    # ========== prepare mask for visualization ==========
    mask = Image.open(args.mask_path)
    mask = mask.convert('RGB')
    mask = np.array(mask).astype(np.uint8)  # three channel integer
    input_mask = mask

    # 将分割掩码保存到 TensorBoard
    # 确保图像张量是 [0, 255] 范围内的 uint8 类型
    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    writer.add_image("分割掩码图片mask", mask_tensor)# 使用 `add_image` 时确保张量类型为 uint8

    print(f'================================================================================')
    print(f'mask_path: {args.mask_path}')

    # prepare directories
    mask_name = args.mask_path.split('/')[-1]
    save_sub_folder = os.path.join(args.save_folder, mask_name)
    os.makedirs(save_sub_folder, exist_ok=True)

    # save seg_mask
    save_path_mask = os.path.join(save_sub_folder, mask_name)
    mask_ = Image.fromarray(input_mask)
    mask_.save(save_path_mask)

    # ========== inference ==========
    with torch.no_grad():

        condition = flattened_img_tensor_one_hot_transpose

        with model.ema_scope("Plotting"):

            # encode condition
            condition = model.get_learned_conditioning(condition)  # [1, 96, 640]
            condition = condition.repeat(args.batch_size, 1, 1)  # [B, 96, 640]

            ddim_sampler = DDIMSampler(model)
            z_0_batch, intermediates = ddim_sampler.sample(
                S=args.ddim_steps,
                batch_size=args.batch_size,
                shape=(3, 64, 64),
                conditioning=condition,
                verbose=False,
                eta=1.0,
                log_every_t=1)
            
            # 假设 z_0_batch 是生成的图片张量
            # z_0_batch 的形状为 (batch_size, 3, 64, 64)
            # 如果张量的值范围是 [-1, 1]，我们需要将它转换到 [0, 1]
            z_0_batch1 = (z_0_batch + 1) / 2

            # 记录每一张生成的图片到 TensorBoard
            for i, img_tensor in enumerate(z_0_batch1):
                # 将张量从 GPU 移动到 CPU 并解除梯度追踪
                img_tensor = img_tensor.detach().cpu()
                
                # 使用 `make_grid` 创建网格
                grid = make_grid(img_tensor, normalize=True, scale_each=True)

                # 将生成的图片保存到 TensorBoard，使用 `add_image`
                writer.add_image(f'中间采样的图片z_0_batch_{i}', grid, global_step=i)

        # decode latent z_0 to image x_0
        x_0_batch = model.decode_first_stage(z_0_batch)  # [B, 3, 256, 256]

        for idx in range(args.batch_size):
            writer.add_image(f"采样生成的图片/{idx}", x_0_batch[idx])

    for idx in range(args.batch_size):

        # ========== save synthesized image x_0 ==========
        save_x_0_path = os.path.join(save_sub_folder,
                                     f'{str(idx).zfill(6)}_x_0.png')
        x_0 = x_0_batch[idx, :, :, :].unsqueeze(0)  # [1, 3, 256, 256]
        x_0 = x_0.permute(0, 2, 3, 1).to('cpu').numpy()
        x_0 = (x_0 + 1.0) * 127.5
        np.clip(x_0, 0, 255, out=x_0)  # clip to range 0 to 255
        x_0 = x_0.astype(np.uint8)
        x_0 = Image.fromarray(x_0[0])
        x_0.save(save_x_0_path)

        # 将 x_0 保存到 TensorBoard
        x_0_tensor = torch.from_numpy(np.array(x_0).transpose(2, 0, 1))  # 转换为 [C, H, W] 的格式
        writer.add_image(f'合成的图片x_0/{idx}', x_0_tensor)

        # save intermediate x_t and pred_x_0
        if args.display_x_inter:
            for cond_name in ['x_inter', 'pred_x0']:
                save_conf_path = os.path.join(save_sub_folder, f'{str(idx).zfill(6)}_{cond_name}.png')
                conf = intermediates[f'{cond_name}']
                conf = torch.stack(conf, dim=0)  # 50x8x3x64x64
                conf = conf[:, idx, :, :, :]  #  50x3x64x64
                print('decoding x_inter ......')
                conf = model.decode_first_stage(conf)  # [50, 3, 256, 256]
                conf = make_grid(conf, nrow=10)  # 10 images per row # [3, 256x3, 256x10]
                conf = conf.permute(1, 2,0).to('cpu').numpy()  # cxhxh -> hxhxc
                conf = (conf + 1.0) * 127.5
                np.clip(conf, 0, 255, out=conf)  # clip to range 0 to 255
                conf = conf.astype(np.uint8)
                conf = Image.fromarray(conf)
                conf.save(save_conf_path)

        # save latent z_0
        if args.save_z:
            save_z_0_path = os.path.join(save_sub_folder,f'{str(idx).zfill(6)}_z_0.png')
            z_0 = z_0_batch[idx, :, :, :].unsqueeze(0)  # [1, 3, 64, 64]
            z_0 = z_0.permute(0, 2, 3, 1).to('cpu').numpy()
            z_0 = (z_0 + 40) * 4  # manually tuned denormalization
            np.clip(z_0, 0, 255, out=z_0)  # clip to range 0 to 255
            z_0 = z_0.astype(np.uint8)
            z_0 = Image.fromarray(z_0[0])
            z_0.save(save_z_0_path)

        # overlay the segmentation mask on the synthesized image to visualize mask consistency
        save_mixed_path = os.path.join(save_sub_folder,f'{str(idx).zfill(6)}_mixed.png')
        # Image.blend(x_0, mask_, 0.3).save(save_mixed_path)
        mixed_img = Image.blend(x_0, mask_, 0.3)  # 混合生成图像和分割掩码
        mixed_img.save(save_mixed_path)  # 保存到磁盘

         # 保存到 TensorBoard
        mixed_tensor = transforms.ToTensor()(mixed_img)  # 将 PIL 图像转换为 PyTorch 张量
        writer.add_image(f'混合的图片mixed_img/{idx}', mixed_tensor)

if __name__ == "__main__":
    main()