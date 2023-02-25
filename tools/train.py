# encoding: utf-8
import argparse
import os
import sys
import torch
from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from loguru import logger


def train(cfg, args):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)   # 加载数据集

    # prepare model  模型初始化
    model = build_model(args, num_classes)

    if not args.IF_WITH_CENTER:
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        loss_func = make_loss(cfg, num_classes)     # modified by gu

        if args.pretrain_choice == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(args.pretrain_choice))
        logger.info('ready train...')
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,     # add for using self trained model
            args
        )

    elif args.IF_WITH_CENTER:
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes, args)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)

        arguments = {}
        if args.pretrain_choice == 'imagenet':
            start_epoch = eval('weights/resnet50-19c8e357.pth')
            print('Start epoch:', start_epoch)
            path_to_optimizer = args.weights.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = args.weights.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = args.weights.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)

            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.weights)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict.keys() == pretrained_dict.keys()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            optimizer_dict = optimizer.state_dict()
            pretrained_dict_optimizer = torch.load(path_to_optimizer)
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if
                               k in optimizer_dict.keys() == pretrained_dict_optimizer.keys()}
            optimizer_dict.update(pretrained_dict_optimizer)
            optimizer.load_state_dict(optimizer_dict)

            center_dict = model.state_dict()
            pretrained_dict_center = torch.load(args.weights)
            pretrained_dict_center = {k: v for k, v in pretrained_dict_center.items() if
                               k in center_dict.keys() == pretrained_dict_center.keys()}
            center_dict.update(pretrained_dict_center)
            center_criterion.load_state_dict(center_dict)

            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif args.pretrain_choice == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(args.pretrain_choice))

        do_train_with_center(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,     # add for using self trained model
            args
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(args.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="Yolo v5 with ReID Baseline Training")
    parser.add_argument(
        "--config_file", type=str, default=r"F:/yolov5_reid/configs/softmax_triplet.yml", help="path to config file"
    )
    parser.add_argument('--LAST_STRIDE', type=int, default=1, help='last stride')
    parser.add_argument('--weights', type=str, default='F:/yolov5_reid//weights/r50_ibn_2.pth')
    parser.add_argument('--neck', type=str, default='bnneck', help='If train with BNNeck, options: bnneck or no')
    parser.add_argument('--test_neck', type=str, default='after', help='Which feature of BNNeck to be used for test, '
                                                                       'before or after BNNneck, options: before or '
                                                                       'after')
    parser.add_argument('--model_name', type=str, default='resnet50_ibn_a', help='Name of backbone')
    # Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
    parser.add_argument('--pretrain_choice', type=str, default='imagenet')
    parser.add_argument('--IF_WITH_CENTER', action='store_true', default=False, help="If train loss include center "
                                                                                    "loss, options: 'yes' or 'no'. "
                                                                                    "Loss with center loss has "
                                                                                    "different optimizer "
                                                                                    "configuration")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("yolov5 reid_baseline")
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg, args)


if __name__ == '__main__':
    main()
