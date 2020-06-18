import numpy as np
import cv2
import os
import functools
import time
from numpy import unravel_index
from ganClass import Gan
from maskRCNN import MaskRcnn
import torch


class JointDetector:
    def __int__(self, weights):
        self.detector = torch.load(weights)

    @staticmethod
    def resize(img):
        img = cv2.resize(img, (256, 256))
        return img

    def detection(self, img):
        hmSize = 64
        imgF = img / 255.0
        rgbDimension = 256
        rgb_batch = np.zeros([1, 3, rgbDimension, rgbDimension])
        imgFb = imgF[:, :, 0]
        imgFg = imgF[:, :, 1]
        imgFr = imgF[:, :, 2]
        rgb_batch[0, 0, :, :] = imgFb - 0.5
        rgb_batch[0, 1, :, :] = imgFg - 0.5
        rgb_batch[0, 2, :, :] = imgFr - 0.5
        rgb_tensor = torch.FloatTensor(rgb_batch)
        rgb_tensor_cuda = rgb_tensor.cuda()
        with torch.no_grad():
            out_cuda = self.detector(rgb_tensor_cuda)
        out = out_cuda.cpu()
        maxuvs = []
        maxHeat = []
        stackedHM = out[0, 0, :, :]
        for j in range(5):
            hm = out[0, j, :, :]
            if j != 0:
                stackedHM += hm
            maxuv = unravel_index(hm.argmax(), hm.shape)
            maxHeat.append(hm[maxuv[0], maxuv[1]])
            if hmSize == 256:
                maxuvs.append((maxuv[1], maxuv[0]))
            elif hmSize == 64:
                maxuvs.append((maxuv[1] * 4, maxuv[0] * 4))
        return maxuvs, maxHeat, stackedHM

    @staticmethod
    def draw_on_image(img, maxuvs, maxHeat, stackedHM):
        color = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        thickness = 2
        radius = 4
        img = cv2.circle(img, maxuvs[0], radius, color[0], thickness)
        img = cv2.circle(img, maxuvs[1], radius, color[1], thickness)
        img = cv2.circle(img, maxuvs[2], radius, color[2], thickness)
        img = cv2.circle(img, maxuvs[3], radius, color[3], thickness)
        img = cv2.circle(img, maxuvs[4], radius, color[4], thickness)

        img = cv2.line(img, maxuvs[0], maxuvs[1], color[0], thickness)
        img = cv2.line(img, maxuvs[1], maxuvs[2], color[1], thickness)
        img = cv2.line(img, maxuvs[2], maxuvs[3], color[2], thickness)
        img = cv2.line(img, maxuvs[3], maxuvs[4], color[3], thickness)

        start_point = [(0, 0), (0, 6), (0, 12), (0, 18), (0, 24)]
        end_point = [(8, 3), (8, 10), (8, 16), (8, 22), (8, 28)]

        img = cv2.rectangle(img, start_point[0], end_point[0], color[0], thickness)
        img = cv2.rectangle(img, start_point[1], end_point[1], color[1], thickness)
        img = cv2.rectangle(img, start_point[2], end_point[2], color[2], thickness)
        img = cv2.rectangle(img, start_point[3], end_point[3], color[3], thickness)
        img = cv2.rectangle(img, start_point[4], end_point[4], color[4], thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (15, 200)
        fontScale = 0.3
        colortt = (255, 0, 0)
        thicknesstt = 1
        img = cv2.putText(img, 'body, ' + str(maxHeat[0])[7:-1], org, font, fontScale, colortt, thicknesstt,
                          cv2.LINE_AA)
        org = (215, 10)
        img = cv2.putText(img, 'shld, ' + str(maxHeat[1])[7:-1], org, font, fontScale, colortt, thicknesstt,
                          cv2.LINE_AA)
        org = (215, 20)
        img = cv2.putText(img, 'elbw, ' + str(maxHeat[2])[7:-1], org, font, fontScale, colortt, thicknesstt,
                          cv2.LINE_AA)
        org = (215, 30)
        img = cv2.putText(img, 'wrst, ' + str(maxHeat[3])[7:-1], org, font, fontScale, colortt, thicknesstt,
                          cv2.LINE_AA)
        org = (215, 40)
        img = cv2.putText(img, 'hand, ' + str(maxHeat[4])[7:-1], org, font, fontScale, colortt, thicknesstt,
                          cv2.LINE_AA)
        org = (215, 5)
        img = cv2.putText(img, 'confidence', org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)

        stackHMmaxuv = unravel_index(stackedHM.argmax(), stackedHM.shape)
        max_prob_value = stackedHM[stackHMmaxuv[0], stackHMmaxuv[1]]
        normalized_stack_HM = np.array((stackedHM / max_prob_value) * 255, dtype=np.uint8)
        hm_cv = cv2.applyColorMap(normalized_stack_HM, cv2.COLORMAP_BONE)  # COLORMAP_AUTUMN   COLORMAP_BONE
        hm_cv = cv2.resize(hm_cv, (256, 256))
        stackimg = np.vstack((img, hm_cv))
        return stackimg


if __name__ == "__main__":
    # crane = CranePipeline("./Mask_to_coco/output","./crane_cycleGAN/checkpoints/crane", "AtoB")
    crane_masking_model = MaskRcnn("./Mask_to_coco/output")
    crane_joint_model = JointDetector("./cam1_epoch_110.pt")
    # crane_cam2_joint = JointDetector("./cam1_epoch_110.pt")
    crane_gan = Gan('Weights/latest_net_G_A.pth')

    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(10)

    counter = 0
    time_interval = 1.0

    while True:
        print('\nindex: {}'.format(counter))
        print('time : {} s   FPS: {}'.format(str(time_interval / 1000000.0)[0:5], 1.0 / (time_interval / 1000000.0)))
        if counter < 30:
            counter += 1
            time.sleep(0.01)
            continue
        # Capture frame-by-frame
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        img1 = img1[:, 0:480]
        img2 = img2[:, 0:480]

        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        img1 = crane_joint_model.resize(img1)
        inference1, masked_1 = crane_masking_model.inference(img1)
        if type(masked_1) == bool:
            continue

        img2 = crane_joint_model.resize(img2)
        inference2, masked_2 = crane_masking_model.inference(img2)
        if type(masked_2) == bool:
            continue

        fake_img_1 = crane_gan.fake(masked_1)
        cv2.imshow('GAN Camera {}'.format(1), fake_img_1)

        cv2.waitKey(1)

        resized_masked_1 = crane_joint_model.resize(masked_1[50:256, 0:206])
        maxuvs1, maxHeat1, stackedHM = crane_joint_model.detection(resized_masked_1)
        crane_joint_model.draw_on_image(resized_masked_1, maxuvs1, maxHeat1, stackedHM)

        resized_masked_2 = crane_joint_model.resize(masked_2[50:256, 0:206])
        maxuvs2, maxHeat2, stackedHM = crane_joint_model.detection(resized_masked_2)
        crane_joint_model.draw_on_image(resized_masked_2, maxuvs2, maxHeat2, stackedHM)

        counter += 1

