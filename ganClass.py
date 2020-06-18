import glob
import numpy as np
import cv2
import os
import functools
import time
from GanModules.options.test_options import TestOptions
from GanModules.data import create_dataset
from shutil import copyfile
from GanModules.models import create_model
from GanModules.data.crane_dataset import CraneDataset
from GanModules.util.util import tensor2im
import torch

#######################################
# Data Loader
#######################################


class CraneDatasetDataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CraneDataset(opt)
        #print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


class Gan:
    def __init__(self, weights):
        self.opt = TestOptions().parse()
        self.opt.dataset_mode = "crane"
        self.opt.single_image_path = None
        self.opt.model = "test"
        self.opt.name = "test"
        self.opt.no_dropout = True
        self.opt.weight_location = weights
        self.opt.num_threads = 1  # test code only supports num_threads = 1
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        self.model = create_model(self.opt)  # create a model given opt.model and other options
        self.model.setup(self.opt)

    def fake(self, img):
        img_file = cv2.imwrite("/tmp/tmp.jpg", img)
        self.opt.single_image_path = "/tmp/tmp.jpg"
        data_loader = CraneDatasetDataLoader(self.opt)
        data = data_loader.load_data()
        for i in data:
            self.model.set_input(i)
            self.model.test()
            result = self.model.get_current_visuals()
            for label, img_tensor in result.items():
                if label == "fake":
                    img_final = tensor2im(img_tensor)
        img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        return img_final
