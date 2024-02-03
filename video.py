import cv2
import numpy as np
import os

class recorder:
    def __init__(self,video_path):
        self.video_path = video_path
        if self.video_path is not None:
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)

    def init(self,name,size = 256,fps = 30,enabled = True):
        if enabled:
            self.video_name = name
            self.video_size = size
            self.video_fps = fps
            if self.video_path is not None:
                self.video = cv2.VideoWriter(str(self.video_path)+"/"+self.video_name,cv2.VideoWriter_fourcc(*'mp4v'),self.video_fps,(self.video_size,self.video_size))
                    
    def record(self,image):
        if self.video_path is not None:
            self.video.write(image.transpose(1,2,0))
            
    def release(self):
        if self.video_path is not None:
            self.video.release()

