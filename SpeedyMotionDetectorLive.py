# AI495 Final Project
# Author: Charles Cheng and Sushma Chandra

# Import libraries
import numpy as np # pip install numpy 
import cv2 as cv # pip install opencv-python

######   Instructions of Use   ######

# To run, execute 
# >> python SpeedyMotionDetectorLive.py

# You may need to change the first line in the detect_falling function
# Replace 0 with the right camera index in cap = cv.VideoCapture(0)
# Also specify the correct fps and aspect ratio of your camera when you
# create a SpeedyDetectorLive object  

class SpeedyDetectorLive:

    def __init__(self, fps, aspect_ratio, channel=2, threshold=0.15, open_size=3, close_size=7, min_area=0.0, wH=0.33, wS=0.33, wV=0.33, min_speed=200.0, similarity=0.03, post_morph=False):
        self.fps = 30
        self.resolution = (aspect_ratio[0]*64, aspect_ratio[1]*64)
        self.img_size = self.resolution[0]*self.resolution[1]
        self.motion_mask = np.zeros(self.resolution, dtype=np.uint8)
        self.frame0 = None
        self.id_counter = 0
        self.obj_hist = {}
        self.motion_objs_stamped = []
        self.fast_objs = []
        self.show_all=False
        self.show_mask=False
        self.show_fast=True
        #self.get_camera()
        self.detect_falling(channel, threshold, open_size, close_size, min_area, wH, wS, wV, min_speed, similarity, post_morph, captured_fps=fps)

    def get_camera(self):
        all_camera_idx_available = []
        for camera_idx in range(20):
            cap = cv.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f'Camera index available: {camera_idx}')
                all_camera_idx_available.append(camera_idx)
                cap.release()
        print(all_camera_idx_available)

    def detect_falling(self, channel, threshold, open_size, close_size, min_area, wH, wS, wV, min_speed, similarity, post_morph, captured_fps=30, frame_limit=10000):
        cap = cv.VideoCapture(0)
        stride = captured_fps//self.fps
        ret, frame = cap.read()
        setup_time = 0 
        while not ret or setup_time < 30:
            ret, frame = cap.read()
            setup_time += 1
        print("Frame: ", 0, end='\r')
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        self.frame0 = (np.array(cv.resize(cv.flip(frame_hsv, -1), \
                                                    (self.resolution[1], self.resolution[0])), dtype=np.float32))
        prev_frame = self.frame0
        curr_frame_count = frame_count = 1
        while cap.isOpened() and curr_frame_count < frame_limit:
            if (curr_frame_count % stride) == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                # print("Frame: ", frame_count, end='\r')
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                curr_frame = (np.array(cv.resize(frame_hsv, \
                                                           (self.resolution[1], self.resolution[0])), dtype=np.float32))
                time = frame_count/self.fps
                # print("Time (s):", time, end='\r')
                static_motion_mask = self.detect_motion(self.frame0[:,:,channel], curr_frame[:,:,channel], threshold, open_size, close_size)
                dyn_motion_mask = self.detect_motion(prev_frame[:,:,channel], curr_frame[:,:,channel], threshold, open_size, close_size)
                self.motion_mask = np.logical_and(static_motion_mask, dyn_motion_mask).astype(np.uint8)
                if post_morph:
                    self.motion_mask = self.closing(self.opening(self.motion_mask, kernel_size=(open_size, open_size)), kernel_size=(close_size, close_size))
                self.motion_objs = cv.connectedComponentsWithStats(self.motion_mask, 4, cv.CV_32S)[2][1:]
                self.motion_filter(min_area)
                self.fast_objs = []
                self.motion_objs_stamped = []
                self.update_obj_hist(curr_frame, time, wH, wS, wV, min_speed, similarity)
                self.show_motion(curr_frame) 
                frame_count += 1
                prev_frame = curr_frame

                if abs(time % 10) < 1e-6:
                    self.obj_hist = {}
            else:
                ret = cap.grab()
                if not ret:
                    break
            curr_frame_count += 1
        cv.destroyAllWindows()
        cap.release()
        print(frame_count, "frames read")

    def detect_motion(self, prev_frame, curr_frame, threshold, open_size, close_size):
        mask = (np.abs(curr_frame-prev_frame)/255.0 > threshold).astype(np.uint8)
        return self.closing(self.opening(mask, kernel_size=(open_size, open_size)), kernel_size=(close_size, close_size))

    def closing(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        return cv.erode(cv.dilate(img, kernel, iterations=1), kernel, iterations=1)

    def opening(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        return cv.dilate(cv.erode(img, kernel, iterations=1), kernel, iterations=1)

    def motion_filter(self, min_area):
        self.motion_objs = [obj for obj in self.motion_objs if obj[2]*obj[3] > min_area*self.img_size]

    def update_obj_hist(self, frame, time, wH, wS, wV, min_speed, similarity):
        unique = {}
        for obj in self.motion_objs:
            raw_score = self.similiarity_score(frame, obj, wH, wS, wV)
            score = round(self.round_nearest(raw_score, similarity), 2)
            if score not in unique:
                if score in self.obj_hist:
                    new_obj = (time, self.obj_hist[score][-1][1], obj, score)
                    self.obj_hist[score].append(new_obj)
                    self.speed_filter(self.obj_hist[score][-2], new_obj, min_speed)
                else:
                    self.obj_hist[score] = [(time, self.id_counter, obj, score)]
                    self.id_counter += 1
                unique[score] = True
            self.motion_objs_stamped.append((time, 0, obj, score))

    def calculate_average(self, frame, tracked_obj, channel):
        sum = 0
        area = 0
        x = np.linspace(0, tracked_obj[3], tracked_obj[3], endpoint=False)
        wx = self.normal_dist(x, np.mean(x), np.std(x)*1.5)
        y = np.linspace(0, tracked_obj[2], tracked_obj[2], endpoint=False)
        wy = self.normal_dist(y, np.mean(y), np.std(y)*1.5)
        for row_pixel in range(tracked_obj[3]):
            for col_pixel in range(tracked_obj[2]):
                wxy = wx[row_pixel]*wy[col_pixel]
                area += wxy
                sum += wxy*frame[tracked_obj[1]+row_pixel, tracked_obj[0]+col_pixel, channel]
        return sum/area
    
    def similiarity_score(self, frame, tracked_obj, wH, wS, wV):
        return wH*self.calculate_average(frame, tracked_obj, 0)/179.0+wS*self.calculate_average(frame, tracked_obj, 1)/255.0+wV*self.calculate_average(frame, tracked_obj, 2)/255.0

    def speed_filter(self, prev_obj, curr_obj, min_speed):
        prev_centroid = np.array([prev_obj[2][1]+prev_obj[2][3]/2, prev_obj[2][0]+prev_obj[2][2]/2])
        curr_centroids = np.array([curr_obj[2][1]+curr_obj[2][3]/2, curr_obj[2][0]+curr_obj[2][2]/2])
        if np.linalg.norm(curr_centroids-prev_centroid)/(curr_obj[0]-prev_obj[0]) > min_speed:
            self.fast_objs.append(curr_obj)

    def show_motion(self, frame):
        img = frame.astype(np.uint8)
        if self.show_all:
            for obj in self.motion_objs_stamped:
                cv.putText(img, 'Score ' + str(obj[3]), (obj[2][0], obj[2][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, 2)
                cv.rectangle(img, (obj[2][0], obj[2][1]), (obj[2][0]+obj[2][2]-1, obj[2][1]+obj[2][3]-1), (0, 255, 255), 1) # red
        if self.show_fast:
            for obj in self.fast_objs:
                cv.putText(img, 'Speedy!', (obj[2][0], obj[2][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (63,255,255), 1, 2)
                cv.rectangle(img, (obj[2][0], obj[2][1]), (obj[2][0]+obj[2][2]-1, obj[2][1]+obj[2][3]-1), (63, 255, 255), 1) # green
        if self.show_mask:
            fg = cv.bitwise_or(img, img, mask=self.motion_mask)
            cv.imshow('motion mask', cv.cvtColor(fg, cv.COLOR_HSV2BGR))
        cv.namedWindow("object detection", cv.WINDOW_NORMAL)
        cv.resizeWindow("object detection", 5*self.resolution[1], 5*self.resolution[0])
        cv.imshow('object detection', cv.cvtColor(img, cv.COLOR_HSV2BGR))
        if cv.waitKey(1) & 0xFF == ord('q'):
            exit()
     
    def round_nearest(self, num, a):
        return round(num/a)*a

    def normal_dist(self, x, avg, std):
        ndist = np.pi*std*np.exp(-0.5*((x-avg)/std)**2)
        return ndist/max(ndist)

if __name__ == "__main__":
    falling_objs = SpeedyDetectorLive(30, (3, 4), channel=2, threshold=0.08, open_size=5, close_size=9, min_area=0.0005, wH=0.15, wS=0.25, wV=0.6, min_speed=400.0, similarity=0.05, post_morph=True)