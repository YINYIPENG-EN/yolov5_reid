import argparse
import random
import time
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')
from reid.data.transforms import build_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadScreenshots, LoadImages
from utils.general import check_img_size, Profile, non_max_suppression, scale_boxes, check_file, increment_path
from utils.plots import plot_one_box
from reid.data import make_data_loader
from pathlib import Path
from reid.modeling import build_model
from reid.config import cfg as reidCfg
import numpy as np
from models.common import DetectMultiBackend
from PIL import Image
import cv2
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def detect(source='0',
           imgsz=(640, 640),
           weights='yolov5s.pt',
           dnn=False,
           data='',
           half=False,
           conf_thres=0.5,
           iou_thres=0.5,
           classes=None,
           agnostic_nms=False,
           vid_stride=1,
           max_det=1000,
           dist_thres=1.0,
           save_res=False,
           project=ROOT / 'runs/detect',
           name='exp',
           exist_ok=False):
    source = str(source)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False  # set False for reproducible results


    # ---------- 行人重识别模型初始化 --------------------------
    query_loader, num_query = make_data_loader(reidCfg)  # 验证集预处理
    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
    reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
    reidModel.to(device).eval()  # 模型测试

    query_feats = []  # 测试特征
    query_pids = []  # 测试ID

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch  # 返回图片，ID，相机ID
            img = img.to(device)  # 将图片放入gpu
            feat = reidModel(img)  # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats.append(feat)  # 获得特征值列表
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = F.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量

    # --------------- yolov5 行人检测模型初始化 -------------------
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # 对于每种类别随机使用一种颜色画框
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # profile()可以记录时间，进入时记录当前时间，退出时输出当前时间与start时间差
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # numpy to tensor
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 就等于im.unsqueeze(0)

        # Inference
        t = time.time()
        with dt[1]:
            pred = model(im)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == 'person':
                        print('%g %ss' % (n, names[int(c)]), end=', ')  # 打印个数和类别

                gallery_img = []
                gallery_loc = []  # 这个列表用来存放框的坐标
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == 'person':
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        w = xmax - xmin
                        h = ymax - ymin
                        # 如果检测到的行人太小了，感觉意义也不大
                        # 这里需要根据实际情况稍微设置下
                        if w * h > 500:
                            gallery_loc.append((xmin, ymin, xmax, ymax))
                            crop_img = im0[ymin:ymax,
                                       xmin:xmax]  # HWC (602, 233, 3)  这个im0是读取的帧，获取该帧中框的位置 im0= <class 'numpy.ndarray'>

                            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                            crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                            gallery_img.append(crop_img)
                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img)  # torch.Size([7, 2048])
                    print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                    distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                    distmat = distmat.cpu().numpy()
                    distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                    index = distmat.argmin()
                    if distmat[index] < dist_thres:
                        # print('距离：%s' % distmat[index])
                        plot_one_box(gallery_loc[index], im0, label='find!', color=colors_[int(cls)])

            print('Done. (%.3fs)' % (time.time() - t))
            if webcam:
                cv2.imshow('person search', im0)
                cv2.waitKey(25)
            if save_res:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='person search')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
    parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')
    opt = parser.parse_args()
    logger.info(opt)
    weights, source, data, imgsz, conf_thres, iou_thres, max_det, classes, agnostic_nms, half, dnn, vid_stride, dist_thres, save_res= opt.weights, opt.source, opt.data, \
                                                                                                                                     opt.imgsz, opt.conf_thres, opt.iou_thres, opt.max_det, opt.classes, opt.agnostic_nms, opt.half, opt.dnn, opt.vid_stride, opt.dist_thres, opt.save_res

    with torch.no_grad():
        detect(source, imgsz, weights, dnn, data, half, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes,
               agnostic_nms=agnostic_nms, vid_stride=vid_stride, max_det=max_det, dist_thres=dist_thres, save_res=save_res)
