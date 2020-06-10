

import torch
import torchvision

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import (
    DefaultPredictor,
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch
)
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
#
import os
import json
import numpy as np
import cv2
import random


import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm



def get_dataset_dict(annotations_path, imgs_dir=None, debug=False):
    '''
    Maps annotations and images to a form usable by detectron2.

    Args:
        annotations_path: Path to the CrowdHuman annotations file.
        imgs_dir: Optional path to corresponding images if different
                  than 'annotations_path' directory.
        debug: Optional to print basic output from data.

    Returns:
        dataset_dicts: Dictionary mapping annotations and images
                        in a form for Detectron2.
    '''
    if imgs_dir is None:
        imgs_dir = os.path.dirname(annotations_path)

    print("Annotations from: ", annotations_path)
    print("Loading images from: ", imgs_dir)

    with open(annotations_path, 'r+') as f:
        datalist = f.readlines()

    inputfile = []
    inner = {}
    dataset_dicts = []

    # Only using a subset of the dataset for testing training.
    # Therefore find the names of those images and only load
    # those in if found.
    # NOTE:
    # - This behaviour will change when using the full dataset
    #   as all images will be present for loading.
    #
    img_names = [filename for filename in os.listdir(imgs_dir)]

    idx = 0
    for i in np.arange(len(datalist)):
        adata = json.loads(datalist[i])
        gtboxes = adata['gtboxes']

        # record["file_name"] = filename
        # record["image_id"] = idx
        # record["height"] = height
        # record["width"] = width

        img_name = adata['ID'] + ".jpg"
        if img_name in img_names:
            if debug:
                print(img_name)

            idx += 1
            img_name = adata['ID'] + ".jpg"
            record = {}
            filename = os.path.join(imgs_dir, img_name)
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            for gtbox in gtboxes:
                if gtbox['tag'] == 'person':
                    if debug:
                        print(gtbox)

                    # NOTE on bounding box types for ground truth
                    # and coordinates.
                    #
                    # 'hbox' --> head box
                    # 'vbox' --> visible body box
                    # 'fbox': Lenovo expands the body box
                    # Coordinates: x y w h
                    #
                    obj = {
                        "bbox": gtbox['hbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 0,
                        "iscrowd": 0 #1 if (len(gtboxes) > 0) else 0
                    }
                    objs.append(obj)

            if len(objs) > 0:
                record["annotations"] = objs
                dataset_dicts.append(record)

    return dataset_dicts




def get_training_config(arch_definition, dataset_train, dataset_test=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(arch_definition))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_train,)
    if dataset_test is not None:
        cfg.DATASETS.TEST = (dataset_test,)
    else:
        cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(arch_definition)  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (faces)
    cfg.MODEL.MASK_ON = False # Not doing any segmentation for this model.


def get_parser():
    '''
    Arguments for training a custom model.
    '''
    parser = default_argument_parser()
    parser.add_argument(
        "--train-data",
        type=str,
        default='',
        help="Path to training data"
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        default='',
        help="Path to training annotation data"
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default='',
        help="Path to validation data"
    )
    parser.add_argument(
        "--validation-annotations",
        type=str,
        default='',
        help="Path to validation annotation data"
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default='',
        help="Path to validation data"
    )
    parser.add_argument(
        "--train-registration",
        type=str,
        default='faces_train',
        help="Unique name for registering training dataset"
    )
    parser.add_argument(
        "--validation-registration",
        type=str,
        default='faces_validation',
        help="Unique name for registering validation dataset"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        # nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def setup(args, dataset_train='faces_train', dataset_validation='faces_validation'):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights_file
    cfg.merge_from_list(args.opts)
    return cfg


def main(args):


    dataset_train = "faces_train"
    dataset_validation = "faces_validation"
    cfg = setup(args)
    cfg.DATASETS.TRAIN = (dataset_train,)
    ### TODO:
    ### - Test with validation set.
    ###
    # cfg.DATASETS.TEST = (dataset_validation,)
    cfg.DATASETS.TEST = ()

    train_data = args.train_data
    train_annots = args.train_annotations
    validation_data = args.validation_data
    validation_annots = args.validation_annotations

    DatasetCatalog.register(dataset_train, lambda: get_dataset_dict(train_annots, train_data))
    MetadataCatalog.get(dataset_train).set(thing_classes=["face"])
    DatasetCatalog.register(dataset_validation, lambda: get_dataset_dict(validation_annots, validation_data))
    MetadataCatalog.get(dataset_validation).set(thing_classes=["face"])
    faces_metadata = MetadataCatalog.get(dataset_train)


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



if __name__ == "__main__":

    args = get_parser()
    print(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


    # Example usage:
    #
    # !python detectron2_repo/train_faces.py \
    # --config-file 'detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml' \
    # --train-data "$imgs_training_path" \
    # --train-annotations "$annotations_train" \
    # --validation-data "$imgs_validation_path" \
    # --validation-annotations "$annotations_validation" \
    # --weights-file 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl' \
    # SOLVER.BASE_LR 0.0025 \
    # SOLVER.IMS_PER_BATCH 2 \
    # MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
    # SOLVER.MAX_ITER 1000
