'''
A Module which binds Yolov7 repo with Deepsort with modifications
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

# def ccw(A, B, C):
#     return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

# def intersect(A, B, C, D):
#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# def check_cross(checkline, traj_line):
#     previous_x, previous_y = traj_line[1][0], traj_line[1][1]
#     next_x, next_y = traj_line[0][0], traj_line[0][1]
#     x1, y1, x2, y2 = bbox
#     return intersect(Point(previous_x, previous_y), Point(next_x, next_y),
#                      Point(checkline[0][0], checkline[0][1]), Point(checkline[1][0], checkline[1][1]))

class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()
        self.vehicles_count = 0  # Counter variable for total vehicles seen
        self.counted_ids = set() 

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker
        
        #--------------------new--------
        # Initialize CSV writer
        self.csv_filename = "bbox_coordinates.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Frame', 'Track ID', 'Class', 'X_min', 'Y_min', 'X_max', 'Y_max', 'X_mid', 'Y_mid'])
        self.object_trajectories = {}
        #--------------------new--------

    # def bbox_crosses_area(self, bbox, area_points):
    #     x1, y1, x2, y2 = bbox
    #     mid_x = x1+((x2-x1)/2)
    #     mid_y = y1+((y2-y1)/2)
    #     area_points = area_points.astype(np.int32) 
    #     # Ensure area_points are integers
    #     print(x1,y1,x2,y2)

    #     # Check if any part of the bounding box enters the counting area
    #     return (mid_x < area_points[2, 0] and mid_x > area_points[0, 0] and
    #             mid_y > area_points[1, 1] and y2 < area_points[3, 1])
    
    def intersect_bbox_line(self, bbox, line):
        #return check_cross(line, bbox)
        # x1, y1, x2, y2 = bbox
        # line_x1, line_y1 = line[0]
        # line_x2, line_y2 = line[1]
    
        # # Calculate the midpoint of the bounding box
        # bbox_midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    
        # # Check if the midpoint intersects with the counting line
        # return (
        #     ((bbox_midpoint[0] >= min(line_x1, line_x2) and bbox_midpoint[0] <= max(line_x1, line_x2)) and
        #      (bbox_midpoint[1] >= min(line_y1, line_y2) and bbox_midpoint[1] <= max(line_y1, line_y2)))
        # )
        x1, y1, x2, y2 = bbox
        line_x1, line_y1 = line[0]
        line_x2, line_y2 = line[1]
        #Check if the bounding box intersects with the counting line
        return (
            ((x1 <= line_x1 and x2 >= line_x1) or (x2 >= line_x2 and x1 <= line_x2)) and
            ((y1 <= line_y1 and y2 >= line_y1) or (y2 >= line_y2 and y1 <= line_y2))
        )
    
    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        vehicles_count=0
        #counting_line= [(510, 35), (656, 78)]
        #counting_line = [(3, 1057), (1138, 118)]
        # counting_line = [(3, 1057), (1138, 118)]# Adjust as needed
        # HERE
        counting_lines = [
            [(800, 320) , (1534, 312)],
            [(1720, 320), (2560, 312)],
            [(2690, 320), (3400, 312) ]
        ]
        #  HERE
        #[(410, 39), (610, 89)]
        #[(510, 35), (656, 78)]
        #counting_line =[(117, 31), (678, 182)]
        #counting_line = [(450, 40), (656, 78)]
        #counting_line = [(510, 35), (656, 78)]
        #counting_line = [(450, 30), (656, 78)]
        #(117, 31), (678, 182)
        #print(counting_area[2, 0])
        #print(counting_area[0, 0])
        #print(counting_area[3, 1])
        #print(counting_area[2, 1])
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            
            #print("Counting area coordinates:", counting_area)
            # cv2.line(frame, (counting_line[0][0], counting_line[0][1]), (counting_line[1][0], counting_line[1][1]), (255, 0, 255), 5)
            # cv2.line(frame, (counting_line[0][0], counting_line[0][1]), (counting_line[1][0], counting_line[1][1]), (255, 0, 255), 5)
            # cv2.line(frame, (counting_line[0][0], counting_line[0][1]), (counting_line[1][0], counting_line[1][1]), (255, 0, 255), 5)
            #cv2.polylines(frame, [counting_area], isClosed=True, color=(255, 0, 0), thickness=2)
            # HERE
            for line in counting_lines:
                cv2.line(frame, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 255), 5)
            # HERE

            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)
                
                # #--------------------new--------
                # # Calculate the midpoint
                # x_mid = int((bboxes[i, 0] + bboxes[i, 2]) / 2)
                # y_mid = int((bboxes[i, 1] + bboxes[i, 3]) / 2)

                # # Save detection information to CSV
                # frame_data = [frame_num, None, class_name, int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3]), x_mid, y_mid]
                # self.csv_writer.writerow(frame_data)
                # #----------------new-------------
            
            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
  
         
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                
                # if self.intersect_bbox_line(bbox, counting_line):
                #   # Increment the count or perform any other action
                #   self.vehicles_count += 1
                #   print(f"Vehicle Counted! Total Vehicles: {self.vehicles_count}")

                # if self.intersect_bbox_line(bbox, counting_line):
                #     vehicles_count += 1
                #     cv2.putText(frame, "Vehicle Detected", (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if count_objects:
                  for i in range(count):
                      track_id = self.tracker.tracks[i].track_id
                      bbox = self.tracker.tracks[i].to_tlbr()
                      #--------------------new--------
                      # Calculate the midpoint
                      x_mid = int((bbox[0] + bbox[2]) / 2)
                      y_mid = int((bbox[1] + bbox[3]) / 2)
    
                      # Save detection information to CSV
                      frame_data = [frame_num, track_id, class_name, int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3]), x_mid, y_mid]
                      self.csv_writer.writerow(frame_data)
                      #----------------new-------------
                         # Draw the tracking path
                      if track_id in self.object_trajectories:
                          self.object_trajectories[track_id].append((x_mid, y_mid))
                      else:
                          self.object_trajectories[track_id] = [(x_mid, y_mid)]

                      # Display the tracking path on the frame
                      if len(self.object_trajectories[track_id]) > 1:
                          for j in range(1, len(self.object_trajectories[track_id])):
                              cv2.line(frame, self.object_trajectories[track_id][j - 1], self.object_trajectories[track_id][j], (0, 255, 0), 2)
                        #----------------new-------------
                    #   if track_id not in self.counted_ids:
                    #       #bbox = self.tracker.tracks[i].to_tlbr()                         
                    #       if self.intersect_bbox_line(bbox, counting_line):
                    #           self.counted_ids.add(track_id)
                    #           self.vehicles_count += 1  # Increment the total vehicles count
                      for line in counting_lines:
                        if self.intersect_bbox_line(bbox, line) and track_id not in self.counted_ids:
                            self.counted_ids.add(track_id)
                            self.vehicles_count += 1  # Increment the total vehicles count
                            break  # Stop checking once counted for one line


            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count} ||  {vehicles_count} vehicles")
            
            cv2.putText(frame, "Total Vehicles: {}".format(self.vehicles_count), (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()
        self.csv_file.close()