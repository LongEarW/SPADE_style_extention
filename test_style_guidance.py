"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import torch
from PIL import Image
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform
import numpy as np


opt = TestOptions().parse()

# Load style image
style_image_path_list = [
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00003079.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00003080.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00003082.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00003086.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00005840.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00005849.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00005855.jpg",
        "/home/wenqingwei0304/ADEChallengeData2016_outdoors/images/training/ADE_train_00005858.jpg",
        "/home/wenqingwei0304/ADE20K_2021_17_01/images/ADE/training/nature_landscape/coast/ADE_train_00020509.jpg",
        "/home/wenqingwei0304/ADE20K_2021_17_01/images/ADE/training/nature_landscape/coast/ADE_train_00020519.jpg"
    ]

style_image_list = []
style_image_tensor_list = []
for style_image_path in style_image_path_list:
    style_image = Image.open(style_image_path)
    params = get_params(opt, style_image.size)
    style_image = style_image.convert('RGB')
    style_image_list.append(np.array(style_image))
    transform_style_image = get_transform(opt, params)
    style_image_tensor = transform_style_image(style_image)
    style_image_tensor_list.append(style_image_tensor)

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    # # early stop
    # if i * opt.batchSize >= opt.how_many:
    #     break
    # keep groundtruth
    ground_truths = []
    for b in range(data_i['image'].shape[0]):
        ground_truths.append(data_i['image'][b].detach().clone())

    # generate with each style guidance
    for style_index, (style_image, style_image_tensor) in enumerate(zip(style_image_list, style_image_tensor_list)):
        # set style image
        for b in range(data_i['image'].shape[0]):
            data_i['image'][b] = style_image_tensor
        # generate images
        generated = model(data_i, mode='inference')
        # 
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            # print('process image... %s' % img_path[b])
            visuals = OrderedDict(
                [
                    ('style_image', style_image),
                    ('input_label', data_i['label'][b]),
                    ('synthesized_image', generated[b]),
                    ('ground_truth_image', ground_truths[b]),
                ]
            )
            save_img_path = [_.replace("ADE_val_", "ADE_val_styleidx{}_".format(style_index)) for _ in img_path[b:b + 1]]
            visualizer.save_images_compare(webpage, visuals, save_img_path, no_conversion=["style_image"])

webpage.save()
