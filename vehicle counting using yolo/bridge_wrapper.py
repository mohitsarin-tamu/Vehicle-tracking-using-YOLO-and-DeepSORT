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



# Define the Point class and helper functions
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def check_cross(checkline, traj_line):
    previous_x, previous_y = traj_line[1].x, traj_line[1].y
    next_x, next_y = traj_line[0].x, traj_line[0].y
    return intersect(Point(previous_x, previous_y), Point(next_x, next_y),
                     Point(checkline[0][0], checkline[0][1]), Point(checkline[1][0], checkline[1][1]))



class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        
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
        self.crossed_track_ids = {i: set() for i in range(3)}  #new
        self.track_data = {}


    def check_cross(self, counting_line, traj_line):
    # Extract points for the counting line and trajectory line
        x1, y1 = counting_line[0]
        x2, y2 = counting_line[1]
        x3, y3 = traj_line[0]
        x4, y4 = traj_line[1]

        # Helper function to calculate the orientation
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        # Function to check if two line segments intersect
        def intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases (checking collinearity)
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True

            return False

        # Helper to check if point r lies on segment pq
        def on_segment(p, q, r):
            return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

        # Check intersection
        return intersect((x1, y1), (x2, y2), (x3, y3), (x4, y4))

    
    def intersect_bbox_line(self, bbox, line):
        x1, y1, x2, y2 = bbox
        line_x1, line_y1 = line[0]
        line_x2, line_y2 = line[1]
        #Check if the bounding box intersects with the counting line
        return (
            ((x1 <= line_x1 and x2 >= line_x1) or (x2 >= line_x2 and x1 <= line_x2)) and
            ((y1 <= line_y1 and y2 >= line_y1) or (y2 >= line_y2 and y1 <= line_y2))
        )
    
    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
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
        counting_lines = [
                   [(727,182), (970,156)],
                   [(1584,191), (1841,238)],
                   [(158,373), (261,659)]
        ]
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process

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
            
            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked in the frame: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

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

                    # for line_idx, counting_line in enumerate(counting_lines):
                    #     # Check if the trajectory crosses the counting line
                    #     if len(self.object_trajectories[track_id]) > 1:
                    #         traj_line = [self.object_trajectories[track_id][-2], self.object_trajectories[track_id][-1]]
                    #         if self.check_cross(counting_line, traj_line) and track_id not in self.crossed_track_ids[line_idx]:
                    #             self.crossed_track_ids[line_idx].add(track_id)
                    #             self.vehicles_count += 1  # Increment the total vehicles count
                    #             break 
                    # Check for crossings for each line
                    for line_idx, counting_line in enumerate(counting_lines):
                        if len(self.object_trajectories[track_id]) > 1:
                            traj_line = [self.object_trajectories[track_id][-2], self.object_trajectories[track_id][-1]]
                            if self.check_cross(counting_line, traj_line) and track_id not in self.crossed_track_ids[line_idx]:
                                self.crossed_track_ids[line_idx].add(track_id)
                                self.vehicles_count += 1  # Increment the crossing count for the vehicle
                                break  # Stop checking once a crossing is detected


            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count} ||  {vehicles_count} vehicles")
            
            # cv2.putText(frame, "Total Vehicles: {}".format(self.vehicles_count), (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        # 1. Prepare the data for export
            crossings_count = {}
            for line_idx in range(3):  # Assuming 3 counting lines
                crossings_count[line_idx] = df.groupby(pd.Grouper(freq='5T')).apply(
                    lambda x: len(set(x['Track ID']).intersection(crossed_track_ids[line_idx]))
                )

            # 2. Convert the crossing count into a DataFrame
            crossings_count_df = pd.DataFrame({
                'Timestamp': crossings_count[0].index,
                'Crossing Count Line 1': crossings_count[0].values,
                'Crossing Count Line 2': crossings_count[1].values,
                'Crossing Count Line 3': crossings_count[2].values
            })

            # 3. Export the data to Excel
            crossings_count_df.to_excel('crossings_count.xlsx', index=False)
            print("Crossing count data for all lines has been exported to crossings_count.xlsx")

        if show_live:
            cv2.destroyAllWindows()
        if self.csv_file:
            self.csv_file.close()