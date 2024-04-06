import argparse
import os
import sys
from os import mkdir
from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="yolo v5 with ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default=r"./configs/softmax_triplet.yml", help="path to config file", type=str
    )
    parser.add_argument('--model_name', type=str, default='resnet50_ibn_a', help='backbone')
    parser.add_argument('--LAST_STRIDE', type=int, default=1, help='last stride')
    parser.add_argument('--weights', type=str, default='', help='weight path')
    parser.add_argument('--neck', type=str, default='bnneck', help='If train with BNNeck, options: bnneck or no')
    parser.add_argument('--pretrain_choice', default='')
    parser.add_argument('--test_neck', type=str, default='after', help='Which feature of BNNeck to be used for test, '
                                                                       'before or after BNNneck, options: before or '
                                                                       'after')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print(args)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        try:
            cfg.merge_from_file(args.config_file)
        except Exception as e:
            print(e)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("yolo v5 reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(args, num_classes)
    model.load_param(args.weights)

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
