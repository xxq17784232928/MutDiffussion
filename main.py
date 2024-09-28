import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

from torch.utils.tensorboard import SummaryWriter

#解析参数
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    # 参数配置
    parser.add_argument(
        "-n",  # 参数名简写形式
        "--name",  # 参数名全称
        type=str,  # 参数类型为字符串
        const=True,  # 参数存在即为True，不需要赋值
        default="",  # 默认值为空字符串
        nargs="?",  # 参数可选，存在则需要值，不存在则为默认值
        help="postfix for logdir",  # 参数的帮助信息
    )
    parser.add_argument(
        "-r",  # 参数名简写形式
        "--resume",  # 参数名全称
        type=str,  # 参数类型为字符串
        const=True,  # 参数存在即为True，不需要赋值
        default="",  # 默认值为空字符串
        nargs="?",  # 参数可选，存在则需要值，不存在则为默认值
        help="resume from logdir or checkpoint in logdir",  # 参数的帮助信息
    )

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        # metavar="base_config.yaml",
        metavar="configs/512_codiff_mask_text.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        # default="logs"
         default="outputs/512_codiff_mask_text1",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

#从opt中提取非默认训练参数，并将其作为字典返回
def nondefault_trainer_args(opt):  # 定义函数nondefault_trainer_args，接受一个参数opt
    try:  # 尝试执行以下代码
        parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
        parser = Trainer.add_argparse_args(parser)  # 使用Trainer的add_argparse_args方法添加参数到parser
    except AttributeError as e:  # 如果出现AttributeError异常
        print(e)  # 打印异常信息
    args = parser.parse_args([])  # 解析空列表以获取默认参数
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))  # 返回opt和默认参数args中值不同的参数名称，按字母顺序排序

#用于回调函数的基类
class SetupCallback(Callback):
    #初始化
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()  # 调用父类的初始化方法
        self.resume = resume  # 保存参数 resume，通常用于表示是否从某个检查点恢复训练 ''
        self.now = now  # 保存参数 now，通常用于表示当前时间戳或日期'2024-07-08T07-35-32'
        self.logdir = logdir  # 保存参数 logdir，日志文件的目录'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T07-35-32_512_vae'
        self.ckptdir = ckptdir  # 保存参数 ckptdir，检查点文件的目录'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T07-35-32_512_vae/checkpoints'
        self.cfgdir = cfgdir  # 保存参数 cfgdir，配置文件的目录'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T07-35-32_512_vae/configs'
        self.config = config  # 保存参数 config，一般是模型或训练的配置
        self.lightning_config = lightning_config  # 保存参数 lightning_config，通常是 PyTorch Lightning 的配置

    #用于在训练过程中处理键盘中断事件
    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:  # 检查当前进程是否为全局 rank 为 0 的进程（通常是主进程）
            print("Summoning checkpoint.")  # 输出提示信息
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")  # 构建保存检查点的路径
            trainer.save_checkpoint(ckpt_path)  # 使用 trainer 保存当前检查点到指定路径

    #在训练开始之前被调用，主要用于创建日志目录、检查点目录和配置文件目录，并将配置信息保存到文件中
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:  # 检查当前进程是否为全局 rank 为 0 的进程（通常是主进程）
            # 创建日志目录和保存配置
            os.makedirs(self.logdir, exist_ok=True)  # 创建日志目录，如果目录已存在则不会报错'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T08-03-52_512_vae'
            os.makedirs(self.ckptdir, exist_ok=True)  # 创建检查点目录，如果目录已存在则不会报错'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T08-03-52_512_vae/checkpoints'
            os.makedirs(self.cfgdir, exist_ok=True)  # 创建配置目录，如果目录已存在则不会报错'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T08-03-52_512_vae/configs'

            if "callbacks" in self.lightning_config:  # 检查 lightning_config 中是否包含 "callbacks" 键
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:  # 检查 callbacks 中是否包含 'metrics_over_trainsteps_checkpoint' 键
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)  # 创建 trainstep_checkpoints 目录，如果目录已存在则不会报错

            print("Project config")  # 输出提示信息 "Project config"
            print(OmegaConf.to_yaml(self.config))  # 以 YAML 格式打印配置 self.config
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))  # 将配置保存为 YAML 文件，文件名包含当前时间戳

            print("Lightning config")  # 输出提示信息 "Lightning config"
            print(OmegaConf.to_yaml(self.lightning_config))  # 以 YAML 格式打印 lightning_config 配置
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))  # 将 lightning_config 配置保存为 YAML 文件，文件名包含当前时间戳

        else:
            # ModelCheckpoint 回调创建了日志目录 --- 删除它
            if not self.resume and os.path.exists(self.logdir):  # 如果不是从检查点恢复且日志目录存在
                dst, name = os.path.split(self.logdir)  # 分割日志目录路径，获取父目录路径和目录名
                dst = os.path.join(dst, "child_runs", name)  # 构建新的目标路径，将目录名放在 child_runs 目录下
                os.makedirs(os.path.split(dst)[0], exist_ok=True)  # 创建目标路径的父目录，如果目录已存在则不会报错
                try:
                    os.rename(self.logdir, dst)  # 尝试将日志目录重命名/移动到目标路径
                except FileNotFoundError:  # 如果遇到文件未找到错误，则跳过
                    pass  # 什么都不做，直接跳过

#ImageLogger 类主要用于在训练过程中记录图像数据。
#Callback 类是 Keras 库中的一个回调类，用于在训练过程中执行特定操作
class ImageLogger(Callback):
    #初始化
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
             rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
             log_images_kwargs=None):
        super().__init__()  # 调用父类的初始化方法
        self.rescale = rescale  # 是否重新缩放图像 True
        self.batch_freq = batch_frequency  # 日志记录的批次频率 1000
        self.max_images = max_images  # 最大图像数量 8
        self.logger_log_images = {  # 定义记录器与相应方法的映射
            pl.loggers.NeptuneLogger: self._testtube,  # TestTubeLogger 对应 _testtube 方法
        }
        #[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]  # 计算日志记录的步数
        if not increase_log_steps:  # 如果不增加日志记录步数
            self.log_steps = [self.batch_freq]  # 将日志记录步数设置为批次频率[1000]
        self.clamp = clamp  # 是否启用限制 True
        self.disabled = disabled  # 是否禁用日志记录False
        self.log_on_batch_idx = log_on_batch_idx  # 是否在批次索引上记录日志 False
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}  # 日志图像的额外参数，如果未提供则为空字典{}
        self.log_first_step = log_first_step  # 是否记录第一个步骤 False

    #用于在PyTorch的Lightning模块中添加图像到TensorBoard
    # 装饰器，确保该方法只在全局 rank 为 0 的进程上运行
    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:  # 遍历 images 字典中的所有键
            grid = torchvision.utils.make_grid(images[k])  # 将图像转换为网格
            grid = (grid + 1.0) / 2.0  # 将图像像素值从 [-1, 1] 重新缩放到 [0, 1]

            tag = f"{split}/{k}"  # 构建标签，包含数据集分割和键名
            pl_module.logger.experiment.add_image(  # 使用记录器将图像添加到实验日志中
                tag, grid,  # 标签和网格图像
                global_step=pl_module.global_step)  # 使用 pl_module 的全局步骤作为日志中的步骤编号


    #用于将训练过程中的图像保存到本地
    #将模型生成或处理的图像数据保存为文件，以便后续查看或分析。
    # 它通过一系列步骤将图像数据从张量转换为可视化格式，并保存到指定目录中。
    @rank_zero_only  # 装饰器，确保该方法只在全局 rank 为 0 的进程上运行
    def log_local(self, save_dir, split, images,global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)  # 构建保存图像的根目录路径
        for k in images:  # 遍历 images 字典中的所有键
            grid = torchvision.utils.make_grid(images[k], nrow=4)  # 将图像转换为网格，每行显示 4 张图像
            if self.rescale:  # 如果需要重新缩放图像
                grid = (grid + 1.0) / 2.0  # 将图像像素值从 [-1, 1] 重新缩放到 [0, 1]
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)  # 转换图像的维度顺序
            grid = grid.numpy()  # 将张量转换为 NumPy 数组
            grid = (grid * 255).astype(np.uint8)  # 将像素值从 [0, 1] 转换为 [0, 255] 并转换为 uint8 类型
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(  # 构建保存图像的文件名
                k,  # 图像键名
                global_step,  # 全局步骤编号
                current_epoch,  # 当前纪元编号
                batch_idx)  # 批次索引
            path = os.path.join(root, filename)  # 构建保存图像的完整路径
            os.makedirs(os.path.split(path)[0], exist_ok=True)  # 创建保存图像的目录，如果目录已存在则不会报错
            Image.fromarray(grid).save(path)  # 将图像保存为文件

    #用于在训练过程中记录图像日志的
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step  # 如果 log_on_batch_idx 为 True，则使用 batch_idx 作为检查索引；否则使用 global_step
        if (self.check_frequency(check_idx) and  # 检查索引是否满足记录频率的条件
                hasattr(pl_module, "log_images") and  # 检查 pl_module 是否具有 log_images 方法
                callable(pl_module.log_images) and  # 检查 log_images 是否是可调用的
                self.max_images > 0):  # 确认 max_images 大于 0
            logger = type(pl_module.logger)  # 获取 pl_module.logger 的类型

            is_train = pl_module.training  # 检查 pl_module 是否处于训练模式
            if is_train:
                pl_module.eval()  # 如果处于训练模式，则切换到评估模式

            with torch.no_grad():  # 禁用梯度计算
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)  # 获取 log_images 方法返回的图像

            for k in images:  # 遍历 images 字典中的所有键
                #会返回 images[k].shape[0] 和 self.max_images 两者中的最小值
                N = min(images[k].shape[0], self.max_images)  # 获取要记录的图像数量，最多为 max_images
                images[k] = images[k][:N]  # 截取前 N 张图像
                if isinstance(images[k], torch.Tensor):  # 如果图像是张量类型
                    images[k] = images[k].detach().cpu()  # 分离图像并将其移动到 CPU
                    if self.clamp:  # 如果启用了限制
                        images[k] = torch.clamp(images[k], -1., 1.)  # 将图像像素值限制在 [-1, 1] 范围内

            self.log_local(pl_module.logger.save_dir, split, images,  # 将图像保存到本地
                        pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)  # 获取相应记录器的方法
            logger_log_images(pl_module, images, pl_module.global_step, split)  # 调用记录器方法记录图像

            if is_train:
                pl_module.train()  # 如果最初处于训练模式，则切换回训练模式

    #检查当前的步骤是否满足指定的频率或者是否在指定的步骤列表中，以便在满足条件时执行某些操作
    def check_frequency(self, check_idx):
        # 检查当前索引是否满足记录频率条件
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                # 尝试从log_steps中移除第一个元素
                self.log_steps.pop(0)
            except IndexError as e:
                # 捕捉IndexError并打印错误信息
                print(e)
                pass
            # 返回True表示满足记录频率条件
            return True
        # 返回False表示不满足记录频率条件
        return False

    #在训练过程中处理训练批次结束时的操作
    #trainer：训练  pl_module：当前正在训练的模型   outputs:模型在当前批次上的输出结果
    #batch:当前批次的数据  batch_idx:当前批次的索引，即第几个批次  
    #dataloader_idx=0:如果有多个数据加载器，这是当前数据加载器的索引，默认为0
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,dataloader_idx=0):
        # 检查当前对象是否未被禁用，并且全局步骤数大于0或者记录第一个步骤
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            # 记录训练中的图像
            self.log_img(pl_module, batch, batch_idx, split="train")

    #用于在训练过程中进行验证批次结束时执行的操作
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,dataloader_idx=0):
        # 检查当前对象是否未被禁用，并且全局步骤数大于0
        if not self.disabled and pl_module.global_step > 0:
            # 记录验证中的图像
            self.log_img(pl_module, batch, batch_idx, split="val")
        # 检查pl_module是否具有calibrate_grad_norm属性
        if hasattr(pl_module, 'calibrate_grad_norm'):
            # 如果calibrate_grad_norm为True，并且batch_idx是25的倍数且大于0
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                # 记录梯度信息
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

#CUDACallback类主要用于在训练过程中监控GPU的使用情况，并触发一些操作，例如调整学习率、提前停止训练等
class CUDACallback(Callback):
    #see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # 重置显存使用统计
        # torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        # 同步显卡
        # torch.cuda.synchronize(trainer.root_gpu)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        # 记录当前时间作为开始时间
        self.start_time = time.time()

    #打印训练过程中的最大内存使用情况和每个epoch的训练时间
    def on_train_epoch_end(self, trainer, pl_module):
        # 同步显卡
        # torch.cuda.synchronize(trainer.root_gpu)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        # 获取最大显存使用量，单位为MiB（1 MiB = 2^20 字节）
        # max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        # 计算本轮训练的时间
        epoch_time = time.time() - self.start_time

        try:
            # 通过训练类型插件减少最大显存使用量
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            # 通过训练类型插件减少训练时间
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            # 输出平均每轮训练的时间
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            # 输出平均峰值显存使用量
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            # 如果发生属性错误，则跳过
            pass

#数据加载器（例如 DataLoader）使用
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    # 将任意具有__len__和__getitem__方法的对象包装成pytorch数据集

    def __init__(self, dataset):
        self.data = dataset  # 存储传入的数据集对象 <ldm.data.celebahq.CelebAConditionalDataset object at 0x7f512b4180a0>

    def __len__(self):
        return len(self.data)  # 返回数据集的长度 3000

    def __getitem__(self, idx):
        return self.data[idx]  # 根据索引返回数据集中的元素

#用于初始化数据加载器（DataLoader）中的工作进程的回调函数
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()  # 获取当前worker的信息

    dataset = worker_info.dataset  # 获取数据集对象
    worker_id = worker_info.id  # 获取当前worker的ID

    if isinstance(dataset, Txt2ImgIterableBaseDataset):  # 如果数据集是Txt2ImgIterableBaseDataset类型
        split_size = dataset.num_records // worker_info.num_workers  # 计算每个worker需要处理的数据量
        # 重置num_records为实际的记录数以保留可靠的长度信息
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]  # 为当前worker分配数据样本ID
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)  # 随机选择当前的种子ID
        #如果每次设置随机数种子，那么那么每次生成的随机数就会相同
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)  # 设置随机数种子，确保每个worker的种子不同
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)  # 对于其他类型的数据集，设置随机数种子，确保每个worker的种子不同


#从配置文件中读取数据集的配置信息，并负责加载、预处理和提供数据集给模型训练
class DataModuleFromConfig(pl.LightningDataModule):
    #初始化
    #batch_size：每个批次的大小   train：用于训练的数据集或数据加载器  validation:用于验证的数据集或数据加载器
    #test:用于测试的数据集或数据加载器  predict:用于对新数据进行预测  wrap:是否对数据集或者数据加载器封装
    #num_workers：数据加载器使用的工作线程数  shuffle_test_loader：否在加载测试数据时打乱数据顺序
    #use_worker_init_fn：是否在工作线程初始化时使用特定的初始化函数，通常用于确保每个工作线程具有相同的随机种子，从而使数据加载过程可重复。
    #shuffle_val_dataloader：是否在加载验证数据时打乱数据顺序
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()  # 调用父类的初始化方法
        self.batch_size = batch_size  # 设置批量大小2
        self.dataset_configs = dict()  # 初始化数据集配置字典{}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2  # 设置工作线程数，如果未指定则为批量大小的两倍 4
        self.use_worker_init_fn = use_worker_init_fn  # 是否使用worker初始化函数  False
        if train is not None:
            self.dataset_configs["train"] = train  # 如果提供了训练数据集配置，添加到配置字典中
            self.train_dataloader = self._train_dataloader  # 设置训练数据加载器
        if validation is not None:
            self.dataset_configs["validation"] = validation  # 如果提供了验证数据集配置，添加到配置字典中
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)  # 设置验证数据加载器，并根据传入的参数决定是否打乱数据
        if test is not None:
            self.dataset_configs["test"] = test  # 如果提供了测试数据集配置，添加到配置字典中
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)  # 设置测试数据加载器，并根据传入的参数决定是否打乱数据
        if predict is not None:
            self.dataset_configs["predict"] = predict  # 如果提供了预测数据集配置，添加到配置字典中
            self.predict_dataloader = self._predict_dataloader  # 设置预测数据加载器
        self.wrap = wrap  # 是否使用包裹功能

    #用于准备数据的函数
    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():  # 遍历数据集配置字典中的所有配置
            instantiate_from_config(data_cfg)  # 根据配置实例化数据集

    #主要用于初始化一个对象（例如模型、数据集等）的属性
    def setup(self, stage=None):
        # 初始化数据集字典
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k]))  # 根据数据集配置字典中的配置实例化数据集，并存入数据集字典
            for k in self.dataset_configs)
        if self.wrap:  # 如果启用了包裹功能
            for k in self.datasets:  # 遍历所有数据集
                self.datasets[k] = WrappedDataset(self.datasets[k])  # 将每个数据集用WrappedDataset类进行包裹

    #用于创建一个训练数据加载器的函数
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)  # 检查训练数据集是否为可迭代的数据集
        if is_iterable_dataset or self.use_worker_init_fn:  # 如果是可迭代的数据集或需要使用worker初始化函数
            init_fn = worker_init_fn  # 设置worker初始化函数
        else:
            init_fn = None  # 否则不设置初始化函数
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,  # 返回训练数据加载器
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,  # 根据数据集类型决定是否打乱数据
                          worker_init_fn=init_fn)  # 设置worker初始化函数

    #创建一个验证数据加载器的函数
    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:  # 检查验证数据集是否为可迭代的数据集或是否需要使用worker初始化函数
            init_fn = worker_init_fn  # 设置worker初始化函数 <function worker_init_fn at 0x7f512b62d9d0>
        else:
            init_fn = None  # 否则不设置初始化函数
        return DataLoader(self.datasets["validation"],  # 返回验证数据加载器<main.WrappedDataset object at 0x7fea7841b8e0>
                          batch_size=self.batch_size,  # 设置批量大小2  8 
                          num_workers=self.num_workers,  # 设置工作线程数4  16
                          worker_init_fn=init_fn,  # 设置worker初始化函数  None
                          shuffle=shuffle)  # 根据传入的参数决定是否打乱数据 False

    #创建一个测试数据加载器，用于加载训练集和验证集
    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)  # 检查训练数据集是否为可迭代的数据集
        if is_iterable_dataset or self.use_worker_init_fn:  # 如果是可迭代的数据集或需要使用worker初始化函数
            init_fn = worker_init_fn  # 设置worker初始化函数
        else:
            init_fn = None  # 否则不设置初始化函数

        # 不要为可迭代数据集打乱数据加载器
        shuffle = shuffle and (not is_iterable_dataset)  # 如果数据集是可迭代的，则不打乱数据

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,  # 返回测试数据加载器
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)  # 设置worker初始化函数和数据是否打乱

    #用于创建一个预测数据加载器的函数
    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:  # 检查预测数据集是否为可迭代的数据集或是否需要使用worker初始化函数
            init_fn = worker_init_fn  # 设置worker初始化函数
        else:
            init_fn = None  # 否则不设置初始化函数
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,  # 返回预测数据加载器
                          num_workers=self.num_workers, worker_init_fn=init_fn)  # 设置批量大小、工作线程数和worker初始化函数


if __name__ == "__main__":

    # 设置打印选项
    np.set_printoptions(threshold=np.inf)

    #日期
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    #添加当前目录的路径'/hexp/xxq/myproject/FaceEditing'
    sys.path.append(os.getcwd())

    #解析参数
    parser = get_parser()

    parser = Trainer.add_argparse_args(parser)

    #用于解析命令行参数，并处理训练日志目录和检查点文件
    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:  # 如果同时指定了name和resume参数
        raise ValueError(  # 抛出一个值错误异常
            "-n/--name and -r/--resume cannot be specified both."  # 错误信息：不能同时指定name和resume参数
            "If you want to resume training in a new log folder, "  # 错误信息：如果你想在新的日志文件夹中恢复训练
            "use -n/--name in combination with --resume_from_checkpoint"  # 错误信息：请使用name参数和resume_from_checkpoint参数组合
        )
    if opt.resume:  # 如果指定了resume参数
        if not os.path.exists(opt.resume):  # 如果resume路径不存在
            raise ValueError("Cannot find {}".format(opt.resume))  # 抛出一个值错误异常，并提示找不到指定的路径
        if os.path.isfile(opt.resume):  # 如果resume路径是一个文件
            paths = opt.resume.split("/")  # 将路径按"/"分割成一个列表
            logdir = "/".join(paths[:-2])  # 取路径中倒数第二层目录作为日志目录
            ckpt = opt.resume  # 将resume路径赋值给ckpt
        else:  # 如果resume路径是一个目录
            assert os.path.isdir(opt.resume), opt.resume  # 断言resume路径确实是一个目录，如果不是，则抛出异常
            logdir = opt.resume.rstrip("/")  # 去掉路径末尾的"/"，并将其赋值给logdir
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")  # 在logdir下拼接出"checkpoints/last.ckpt"作为ckpt路径

        opt.resume_from_checkpoint = ckpt  # 将ckpt路径赋值给opt.resume_from_checkpoint
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))  # 使用glob模块找到logdir目录下所有configs目录中的.yaml文件，并排序
        opt.base = base_configs + opt.base  # 将找到的base_configs列表与opt.base列表合并
        _tmp = logdir.split("/")  # 将logdir路径按"/"分割成一个列表
        nowname = _tmp[-1]  # 取分割后列表的最后一个元素作为nowname
 
    else:
       if opt.name:  # 如果指定了name参数
            name = "_" + opt.name  # 将name参数前加上"_"，并赋值给name变量
       elif opt.base:  # 如果没有指定name参数，但指定了base参数 ['configs/512_vae.yaml']
            cfg_fname = os.path.split(opt.base[0])[-1]  # 取base参数列表中第一个文件的文件名 '512_vae.yaml'
            cfg_name = os.path.splitext(cfg_fname)[0]  # 去掉文件的扩展名，取文件名部分 '512_vae'
            name = "_" + cfg_name  # 将文件名部分前加上"_"，并赋值给name变量
       else:  # 如果既没有指定name参数，也没有指定base参数
            name = ""  # 将name变量设为空字符串
       nowname = now + name + opt.postfix  # 将当前时间、name变量和postfix参数组合成nowname '2024-07-03T09-41-44_512_vae'
       #'/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-07-03T09-41-44_512_vae'
       logdir = os.path.join(opt.logdir, nowname)  # 将logdir参数和nowname组合成新的日志目录路径

    ckptdir = os.path.join(logdir, "checkpoints")  # 将logdir路径与"checkpoints"拼接成ckptdir路径 'logs/2024-06-03T21-21-57/checkpoints'
    cfgdir = os.path.join(logdir, "configs")  # 将logdir路径与"configs"拼接成cfgdir路径 'logs/2024-06-03T21-21-57/configs'
    seed_everything(opt.seed)  # 调用seed_everything函数，并使用opt.seed作为种子参数，以保证结果的可重复性 23

    try:
        #初始化和保存配置
        configs = [OmegaConf.load(cfg) for cfg in opt.base]  # 加载opt.base列表中的每个配置文件，并存储在configs列表中
        cli = OmegaConf.from_dotlist(unknown)  # 将命令行参数unknown转换为OmegaConf配置对象
        config = OmegaConf.merge(*configs, cli)  # 将configs列表中的配置对象与cli配置对象合并成一个配置对象
        lightning_config = config.pop("lightning", OmegaConf.create())  # 从合并后的配置对象中提取出"lightning"配置，若不存在则创建一个空的OmegaConf配置对象

        # 将命令行参数与配置文件中的trainer配置合并
        trainer_config = lightning_config.get("trainer", OmegaConf.create())  # 获取lightning_config中的"trainer"配置，如果不存在则创建一个空的OmegaConf配置对象
        # 默认使用ddp（分布式数据并行） {'benchmark': True, 'accumulate_grad_batches': 2, 'accelerator': 'ddp'}
        trainer_config["accelerator"] = "ddp"  # 设置trainer_config中的"accelerator"为"ddp" {'accelerator': 'ddp'}
        for k in nondefault_trainer_args(opt):  # 遍历opt中非默认的trainer参数
            trainer_config[k] = getattr(opt, k)  # 将这些参数的值更新到trainer_config中 '0,1,2,3,'
        if not "gpus" in trainer_config:  # 如果trainer_config中没有设置"gpus"
            del trainer_config["accelerator"]  # 删除trainer_config中的"accelerator"配置
            cpu = True  # 设置cpu变量为True
        else:  # 如果trainer_config中设置了"gpus"
            gpuinfo = trainer_config["gpus"]  # 获取"gpus"配置的值
            print(f"Running on GPUs {gpuinfo}")  # 打印正在使用的GPU信息Running on GPUs 0,1,2,3,
            cpu = False  # 设置cpu变量为False

        #Namespace(accelerator='ddp', benchmark=True, gpus='0,1,2,3,')
        trainer_opt = argparse.Namespace(**trainer_config)  # 使用trainer_config字典中的键值对创建一个Namespace对象，赋值给trainer_opt
        #trainer_config:{'benchmark': True, 'accelerator': 'ddp', 'gpus': '0,1,2,3,'}
        lightning_config.trainer = trainer_config  # 将trainer_config赋值给lightning_config的trainer属性

        #创建模型
        model = instantiate_from_config(config.model)

        trainer_kwargs = dict()  # 创建一个空字典，赋值给变量trainer_kwargs {}

        default_logger_cfgs = {  # 创建一个字典default_logger_cfgs，包含日志记录器的配置
            "wandb": {  # 配置WandbLogger
                "target": "pytorch_lightning.loggers.WandbLogger",  # 目标类为WandbLogger
                "params": {  # 参数配置
                    "name": nowname,  # 日志记录器的名称'2024-07-08T07-35-32_512_vae'
                    "save_dir": logdir,  # 保存日志的目录'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-08T07-35-32_512_vae'
                    "offline": opt.debug,  # 是否离线模式，取决于opt.debug False
                    "id": nowname,  # 日志记录器的ID '2024-07-08T07-35-32_512_vae'
                }
            },
            "tensorboard": {  # 配置TensorBoardLogger
                "target": "pytorch_lightning.loggers.TensorBoardLogger",  # 目标类为TensorBoardLogger
                "params": {  # 参数配置
                    "name": "tensorboard",  # 日志记录器的名称
                    #'/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-07-06T14-46-25_512_vae'
                    "save_dir": logdir,  # 保存日志的目录
                }
            },
        }

        default_logger_cfg = default_logger_cfgs["tensorboard"]  # 从default_logger_cfgs中获取tensorboard的配置，赋值给default_logger_cfg
        if "logger" in lightning_config:  # 如果lightning_config中包含logger配置
            logger_cfg = lightning_config.logger  # 将lightning_config中的logger配置赋值给logger_cfg
        else:  # 否则
            logger_cfg = OmegaConf.create()  # 创建一个新的空配置，赋值给logger_cfg
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)  # 将default_logger_cfg和logger_cfg合并，结果赋值给logger_cfg

        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)  # 使用logger_cfg实例化logger，并将其添加到trainer_kwargs字典中，键为logger

        default_modelckpt_cfg = {  # 创建一个字典default_modelckpt_cfg，包含模型检查点的配置
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",  # 目标类为ModelCheckpoint
            "params": {  # 参数配置
                "dirpath": ckptdir,  # 保存检查点文件的目录 '/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-06-03T22-11-02_512_vae/checkpoints'
                "filename": "{epoch:06}",  # 保存文件的命名格式，其中{epoch:06}表示六位数的epoch编号
                "verbose": True,  # 是否输出详细信息
                "save_last": True,  # 是否保存最后一个epoch的检查点
            }
        }

        if hasattr(model, "monitor"):  # 如果model对象有"monitor"属性
            print(f"Monitoring {model.monitor} as checkpoint metric.")  # 打印监控的检查点指标 'val/rec_loss'
            default_modelckpt_cfg["params"]["monitor"] = model.monitor  # 将model的monitor属性添加到default_modelckpt_cfg的参数中
            default_modelckpt_cfg["params"]["save_top_k"] = 10  # 设置保存最好的k个模型检查点的数量为10

        if "modelcheckpoint" in lightning_config:  # 如果lightning_config中包含"modelcheckpoint"配置
            modelckpt_cfg = lightning_config.modelcheckpoint  # 将lightning_config中的modelcheckpoint配置赋值给modelckpt_cfg
        else:  # 否则
            modelckpt_cfg = OmegaConf.create()  # 创建一个新的空配置，赋值给modelckpt_cfg
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)  # 将default_modelckpt_cfg和modelckpt_cfg合并，结果赋值给modelckpt_cfg
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")  # 打印合并后的modelckpt配置
        if version.parse(pl.__version__) < version.parse('1.4.0'):  # 如果pytorch_lightning的版本小于1.4.0
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)  # 使用modelckpt_cfg实例化checkpoint_callback，并将其添加到trainer_kwargs字典中，键为"checkpoint_callback"

        #添加一个设置日志目录的回调。
        default_callbacks_cfg = {  # 创建一个字典default_callbacks_cfg，包含各种回调函数的配置
            "setup_callback": {  # 配置SetupCallback回调
                "target": "main.SetupCallback",  # 目标类为SetupCallback
                "params": {  # 参数配置
                    "resume": opt.resume,  # 是否从检查点恢复训练 ""
                    "now": now,  # 当前时间 '2024-06-03T22-15-24'
                    "logdir": logdir,  # 日志目录 '/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-06-03T22-15-24_512_vae'
                    "ckptdir": ckptdir,  # 检查点目录 '/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-06-03T22-15-24_512_vae/checkpoints'
                    "cfgdir": cfgdir,  # 配置文件目录'/hexp/xxq/myproject/FaceAging/output/logdir/512_vae/2024-07-06T14-46-25_512_vae/configs'
                    "config": config,  # 配置文件
                    "lightning_config": lightning_config,  # pytorch_lightning配置
                }
            },
            "image_logger": {  # 配置ImageLogger回调
                "target": "main.ImageLogger",  # 目标类为ImageLogger
                "params": {  # 参数配置
                    "batch_frequency": 750,  # 每隔750个batch记录一次图像
                    "max_images": 4,  # 每次记录的最大图像数
                    "clamp": True  # 是否对图像进行clamp操作
                }
            },
            "learning_rate_logger": {  # 配置LearningRateMonitor回调
                "target": "main.LearningRateMonitor",  # 目标类为LearningRateMonitor
                "params": {  # 参数配置
                    "logging_interval": "step",  # 记录学习率的时间间隔，设为每一步
                    # "log_momentum": True  # 是否记录动量，注释掉此行
                }
            }
            ,
            "cuda_callback": {  # 配置CUDACallback回调
                "target": "main.CUDACallback"  # 目标类为CUDACallback
            },
        }

        if version.parse(pl.__version__) >= version.parse('1.4.0'):  # 如果pytorch_lightning的版本大于或等于1.4.0
            #会调用SetupCallback、ImageLogger、LearningRateMonitor、CUDACallback四个类
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})  # 将modelckpt_cfg添加到default_callbacks_cfg中，键为'checkpoint_callback'

        if "callbacks" in lightning_config:  # 如果lightning_config中包含"callbacks"配置
            callbacks_cfg = lightning_config.callbacks  # 将lightning_config中的callbacks配置赋值给callbacks_cfg
        else:  # 否则
            callbacks_cfg = OmegaConf.create()  # 创建一个新的空配置，赋值给callbacks_cfg

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            # 如果callbacks_cfg中存在'metrics_over_trainsteps_checkpoint'，执行以下操作
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
                # 打印警告信息：每n步保存一次检查点而不删除，可能需要一些空闲空间。

            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    # 指定目标回调函数为ModelCheckpoint
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        # 设置检查点保存目录路径为ckptdir下的'trainstep_checkpoints'文件夹
                        "filename": "{epoch:06}-{step:09}",
                        # 设置检查点文件名格式，包括epoch和step信息
                        "verbose": True,
                        # 启用详细信息打印
                        'save_top_k': -1,
                        # 保存所有检查点，而不是仅保存最好的k个
                        'every_n_train_steps': 10000,
                        # 每10000个训练步骤保存一次检查点
                        'save_weights_only': True,
                        # 只保存模型权重，不保存优化器状态等其他信息
                        # 'automatic_optimization ':False   ################6月17日修改
                    }
                    }
            }

            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
            # 将default_metrics_over_trainsteps_ckpt_dict字典更新到default_callbacks_cfg字典中

        #用于配置和初始化PyTorch-Lightning的Trainer类
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        # 将default_callbacks_cfg和callbacks_cfg合并，合并后的结果赋值给callbacks_cfg

        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            # 如果callbacks_cfg中存在'ignore_keys_callback'并且trainer_opt有'resume_from_checkpoint'属性
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
            # 将trainer_opt.resume_from_checkpoint的值赋给callbacks_cfg.ignore_keys_callback.params中的'ckpt_path'键

        elif 'ignore_keys_callback' in callbacks_cfg:
            # 如果callbacks_cfg中存在'ignore_keys_callback'但trainer_opt没有'resume_from_checkpoint'属性
            del callbacks_cfg['ignore_keys_callback']
            # 从callbacks_cfg中删除'ignore_keys_callback'

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # 对callbacks_cfg中的每个回调函数配置调用instantiate_from_config，并将生成的回调函数列表赋给trainer_kwargs["callbacks"]

        #<pytorch_lightning.trainer.trainer.Trainer object at 0x7fd7665bc6a0>
        trainer = Trainer(**trainer_kwargs)  # 直接使用Trainer类进行实例化

        trainer.logdir = logdir #'/hexp/xxq/myproject/FaceEditing/output/logdir/512_vae/2024-07-09T00-42-33_512_vae'
        # # 将logdir的值赋给trainer.logdir

        #获取数据  DataModuleFromConfig <main.DataModuleFromConfig object at 0x7fb2ca5a2d90>
        data = instantiate_from_config(config.data)
        
        data.prepare_data()  # 调用prepare_data方法，准备数据集
        data.setup()  # 调用setup方法，配置数据集
        print("#### Data #####")
        
        for k in data.datasets:  # 遍历 data.datasets 字典中的每一个键 k  train和validation
            # 打印当前键 k、键对应的值的类名以及键对应的值的长度
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
            # f"{k}" 是当前遍历到的键（数据集名称）
            # f"{data.datasets[k].__class__.__name__}" 是当前键对应的值（数据集）的类名（类型）
            # f"{len(data.datasets[k])}" 是当前键对应的值（数据集）的长度（数据集中的数据数量）
            # train, WrappedDataset, 26999
            # validation, WrappedDataset, 3000

        #%% configure learning rate  # 配置学习率
        #bs:2    base_lr:4.5e-06
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate  # 获取批处理大小和基础学习率

        if not cpu:  # 如果不是使用 CPU
            # 获取 GPU 的数量，通过分割 gpus 字符串并计算其长度  4
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:  # 如果使用 CPU
            ngpu = 1  # 设置 GPU 的数量为 1

        if 'accumulate_grad_batches' in lightning_config.trainer:  # 如果 lightning_config.trainer 中包含 'accumulate_grad_batches'
            # 获取累积梯度的批次数 2
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:  # 如果不包含 'accumulate_grad_batches'
            accumulate_grad_batches = 1  # 设置累积梯度的批次数为 1

        # 打印累积梯度的批次数
        print(f"accumulate_grad_batches = {accumulate_grad_batches}") #2  1

        # 设置 lightning_config.trainer 中的 accumulate_grad_batches 属性为 accumulate_grad_batches 的值
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches #2

        # 如果 opt.scale_lr 为 True，则进行学习率的缩放
        if opt.scale_lr:
            # 计算并设置模型的学习率：累积梯度批次数 * GPU 数量 * 批处理大小 * 基础学习率 7.2e-05
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            # 打印设置的学习率以及计算过程中使用的各个参数值
            #Setting learning rate to 1.80e-05 = 2 (accumulate_grad_batches) * 1 (num_gpus) * 2 (batchsize) * 4.50e-06 (base_lr)
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:  # 如果 opt.scale_lr 为 False，则不进行学习率的缩放
            model.learning_rate = base_lr  # 直接设置模型的学习率为基础学习率
            print("++++ NOT USING LR SCALING ++++")  # 打印未使用学习率缩放的提示信息
            # 打印设置的学习率
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1  # 通过 USR1 信号允许检查点的保存
        def melk(*args, **kwargs):  # 定义一个名为 melk 的函数，接受任意数量的参数和关键字参数
            # run all checkpoint hooks  # 运行所有检查点
            if trainer.global_rank == 0:  # 如果当前节点是主节点（global_rank 为 0）
                print("Summoning checkpoint.")  # 打印提示信息 "Summoning checkpoint."
                ckpt_path = os.path.join(ckptdir, "512_codiff_mask_text.ckpt")  # 定义检查点保存路径，将其命名为 "last.ckpt"
                trainer.save_checkpoint(ckpt_path)  # 使用 trainer 保存当前的检查点到指定路径

        def divein(*args, **kwargs):  # 定义一个名为 divein 的函数，接受任意数量的参数和关键字参数
            if trainer.global_rank == 0:  # 如果当前节点是主节点（global_rank 为 0）
                import pudb  # 导入 pudb 调试器
                pudb.set_trace()  # 设置断点，启动 pudb 调试器

        import signal  # 导入 signal 模块，用于处理信号

        # 将 USR1 信号与 melk 函数关联，当接收到 USR1 信号时，melk 函数将被调用
        signal.signal(signal.SIGUSR1, melk)

        # 将 USR2 信号与 divein 函数关联，当接收到 USR2 信号时，divein 函数将被调用
        signal.signal(signal.SIGUSR2, divein)

        #  run  # 开始运行
        if opt.train:  # 如果 opt.train 为 True，则进行训练
            try:  # 尝试以下代码块
                # # TensorBoard的记录
                # writer = SummaryWriter(log_dir="runs/diffussion_main_mask")

                # # 创建一个 Trainer 实例
                # trainer = pl.Trainer()

                # # 将 Trainer 附加到模型上
                # model.trainer = trainer
                # # 记录模型结构
                # writer.add_graph(model, data)
                trainer.fit(model, data)  # 使用 trainer 进行模型训练，传入模型和数据

                # writer.close()
            except Exception:  # 如果发生异常
                melk()  # 调用 melk 函数进行检查点保存
                raise  # 重新抛出异常

        # 如果 opt.no_test 为 False 并且 trainer 没有被中断，则进行测试
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)  # 使用 trainer 进行模型测试，传入模型和数据

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        print("捕捉")
        sys.exit(0)  # 添加这一行

        



# %%
