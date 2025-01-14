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

opt = TestOptions().parse()
# Load your custom style image
style_image_path = './datasets/style_input_test.png'  # Update this path with your style image filename
style_image = Image.open(style_image_path)
params = get_params(opt, style_image.size)
style_image = style_image.convert('RGB')
transform_style_image = get_transform(opt, params)
style_image_tensor = transform_style_image(style_image)
# style_image = style_image.resize((opt.load_size, opt.load_size))  # Resize to match the model input size
# style_tensor = data.transform(style_image).unsqueeze(0)  # Apply necessary transformations and add batch dimension

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
    if i * (opt.batchSize - 1) >= opt.how_many:
        break
    # take first image for each batch as style input
    #  for b in range(data_i['image'].shape[0]):
        #  data_i['image'][b] = data_i['image'][0]
        
    for b in range(data_i['image'].shape[0]):
        data_i['image'][b] = style_image_tensor

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()