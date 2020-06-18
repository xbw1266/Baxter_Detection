#from crane_cycleGAN.crane_generate import *
from Mask_to_coco.testDet import *
import glob
import numpy as np
import cv2
import os
import functools
import time
from tqdm import tqdm
from numpy import unravel_index
from ganClass import Gan

class maskrcnn():
    def __init__(self, maskrcnn_model_path):
        self.maskrcnn_detector = self.setUpDetectron(maskrcnn_model_path)
        
    def setUpDetectron(self, model_path):
        detector = DetectronInf(model_path)
        print("Detectron Model Loaded!")
        return detector

    def getNewestImage(self, img ):
        self.newestImage = img

    def getInstanceImg(self,cam_index, save_to_file=False, vis=False):
        # if len(self.dataset) == 0:
        #     raise Exception("No dataset loaded")

        self.img_instance, mask = self.maskrcnn_detector.test(self.newestImage)
        self.masked_img = self.newestImage.copy()
        self.img_instance = cv2.resize(self.img_instance , (256,256))
        print(mask.shape)
        if mask.shape == (0,256,256):
            print('invalid image')
            return False
        self.masked_img[mask==0] = (255, 255, 255)
        
        stacked = np.vstack((self.img_instance , self.masked_img))
        # cv2.imshow("MaskRCNN_cam_{}".format(cam_index), self.img_instance)
        # cv2.imshow("Masked_cam_{}".format(cam_index), self.masked_img)
        cv2.imshow("Masked Camera {}".format(cam_index), stacked )
        cv2.waitKey(1) 
        return self.masked_img

    def output_masked_image():
        return self.masked_img , self.img_instance

        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break




class joint_2Ddetector():
    def __init__(self, pose_resnet_model_path, cam_index):
        self.pose_detector = torch.load(pose_resnet_model_path)
        self.detection2D = []
        self.max_uv = []
        self.max_heat = []
        self.stackedHM = []
        self.image = []
        self.cam_index = cam_index

    def getNewestImage(self, img ):
        img = cv2.resize(img, (256,256))
        self.image = img

    def detection(self):
        posenet = self.pose_detector
        img = self.image
        hmSize = 64
        imgF = img/255.0
        rgbDimension = 256
        rgb_batch = np.zeros([1,3,rgbDimension,rgbDimension])
        
        imgFb = imgF[:,:,0]
        imgFg = imgF[:,:,1]
        imgFr = imgF[:,:,2]
        rgb_batch[0,0,:,:] = imgFb - 0.5
        rgb_batch[0,1,:,:] = imgFg - 0.5
        rgb_batch[0,2,:,:] = imgFr - 0.5
        rgb_tensor = torch.FloatTensor(rgb_batch)
        rgb_tensor_cuda = rgb_tensor.cuda()
        with torch.no_grad():
            out_cuda = posenet(rgb_tensor_cuda)
        out = out_cuda.cpu()

        maxuvs = []
        maxHeat = []
        stackedHM = out[0,0,:,:]

        for j in range(5):
            hm = out[0,j,:,:]
            if j != 0:
                stackedHM += hm
            maxuv = unravel_index(hm.argmax(), hm.shape)
            maxHeat.append(hm[maxuv[0],maxuv[1]])
            #print(maxuv)
            if hmSize==256:
                maxuvs.append((maxuv[1],maxuv[0]))
            elif hmSize==64:
                maxuvs.append((maxuv[1]*4,maxuv[0]*4))

        self.max_uv = maxuvs
        self.max_heat = maxHeat
        self.stackedHM = stackedHM

        return maxuvs, maxHeat

    def draw_on_image(self):
        img = self.image
        maxuvs, maxHeat = self.max_uv , self.max_heat
        stackedHM = self.stackedHM 
        color = [(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255)]
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

        #img = cv2.line(img, start_point, end_point, color, thickness) 
        start_point = [(0,0),(0,6),(0,12),(0,18),(0,24)]
        end_point = [(8,3),(8,10),(8,16),(8,22),(8,28)]

        img = cv2.rectangle(img,start_point[0], end_point[0], color[0], thickness) 
        img = cv2.rectangle(img,start_point[1], end_point[1], color[1], thickness) 
        img = cv2.rectangle(img,start_point[2], end_point[2], color[2], thickness) 
        img = cv2.rectangle(img,start_point[3], end_point[3], color[3], thickness) 
        img = cv2.rectangle(img,start_point[4], end_point[4], color[4], thickness) 

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (15, 200) 
        fontScale = 0.3
        colortt = (255, 0, 0) 
        thicknesstt = 1
        img = cv2.putText(img, 'body, '+str(maxHeat[0])[7:-1], org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA) 
        org = (215, 10) 
        img = cv2.putText(img, 'shld, '+str(maxHeat[1])[7:-1], org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)
        org = (215, 20) 
        img = cv2.putText(img, 'elbw, '+str(maxHeat[2])[7:-1], org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)
        org = (215, 30) 
        img = cv2.putText(img, 'wrst, '+str(maxHeat[3])[7:-1], org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)
        org = (215, 40) 
        img = cv2.putText(img, 'hand, '+str(maxHeat[4])[7:-1], org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)
        org = (215, 5) 
        img = cv2.putText(img, 'confidence', org, font, fontScale, colortt, thicknesstt, cv2.LINE_AA)

        #savepath = 'testing_datasets/marked_image_by_model/image_{}.jpg'.format( i)

        stackHMmaxuv = unravel_index(stackedHM.argmax(), stackedHM.shape)
        max_prob_value = stackedHM[stackHMmaxuv[0] , stackHMmaxuv[1]]
        normalized_stack_HM = np.array( (stackedHM/max_prob_value )*255 , dtype = np.uint8)
        hm_cv =  cv2.applyColorMap(normalized_stack_HM, cv2.COLORMAP_BONE)  # COLORMAP_AUTUMN   COLORMAP_BONE
        hm_cv = cv2.resize(hm_cv , (256,256))
        #print('hm cv shape  ',hm_cv.shape)

        stackimg = np.vstack((img, hm_cv))
        cv2.imshow('2D joint Camera {}'.format(self.cam_index), stackimg)
        
        cv2.waitKey(10)


def get_time_us():
    # return the current ubuntu OS time, in micro-second, as int type 
    return int(time.time()*1000000.0)



if __name__ == "__main__":
    #crane = CranePipeline("./Mask_to_coco/output","./crane_cycleGAN/checkpoints/crane", "AtoB")
    crane_cam1_masking = maskrcnn("./Mask_to_coco/output")
    crane_cam1_joint   = joint_2Ddetector("./cam1_epoch_110.pt",1)
    crane_cam2_joint   = joint_2Ddetector("./cam1_epoch_110.pt",2)
    crane_gan  = Gan('Weights/latest_net_G_A.pth')

    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(10)

    counter = 0

    timestamp_us = get_time_us()
    preTime = timestamp_us
    time_interval = 1.0

    while(True):
        print('\nindex: {}'.format(counter))
        print('time : {} s   FPS: {}'.format(str(time_interval/1000000.0)[0:5] , 1.0/(time_interval/1000000.0) ))
        if counter <30:
            counter += 1
            time.sleep(0.01)
            continue
        # Capture frame-by-frame
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        img1 = img1[:,0:480]
        img2 = img2[:,0:480]

        img1 = cv2.resize(img1 , (256,256))
        img2 = cv2.resize(img2 , (256,256))

        
        crane_cam1_masking.getNewestImage(img1)
        masked_1 = crane_cam1_masking.getInstanceImg(1, save_to_file=True, vis=True)
        if type(masked_1) == bool:
        
            continue

        crane_cam1_masking.getNewestImage(img2)
        masked_2 = crane_cam1_masking.getInstanceImg(2 , save_to_file=True, vis=True)
        if type(masked_2) == bool:
            continue

        fake_img_1 = crane_gan.fake(masked_1)
        cv2.imshow('GAN Camera {}'.format(1), fake_img_1)
        
        cv2.waitKey(1)

        crane_cam1_joint.getNewestImage(masked_1[ 50:256 , 0:206 ])
        maxuvs1, maxHeat1 = crane_cam1_joint.detection()
        crane_cam1_joint.draw_on_image()

        crane_cam2_joint.getNewestImage(masked_2[ 50:256 , 0:206 ])
        maxuvs2, maxHeat12 = crane_cam2_joint.detection()
        crane_cam2_joint.draw_on_image()

        #crane.getCycleGanImage(save_to_file=True, vis=True)

        counter += 1

        newTime = get_time_us()
        time_interval = newTime - preTime 
        preTime = newTime



























