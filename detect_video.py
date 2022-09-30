######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 9/28/19
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images. It draws boxes and scores 
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import cv2
import numpy as np
import importlib.util
from imutils.video import VideoStream , FileVideoStream
import matplotlib.pyplot as plt
from meter_reader import meter_GorR

def detect(PATH_TO_CKPT, PATH_TO_LABELS, min_conf_threshold, use_TPU=False, save_result_img=False, keyboard_input=False):
    def _detect(frame, img_counter = [0]):
        # copy of original array for cv2... (god & gary bradski knows why...)
        frame = frame.copy()
        # Load image and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape 
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        print('scores:', scores)
        detected = False
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                detected = True
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                img_crop = frame[ymin:ymax,xmin:xmax]
                cv2.imwrite('./test.png', img_crop)
                img_crop = img_crop[:,:,::-1]

                # try:
                print('read meter ...')
                GorR , measurement_val = meter_GorR(img_crop, draw_flag=keyboard_input)
                if measurement_val >= 100:
                    measurement_val = 100.0000
                elif measurement_val <= 0:
                    measurement_val = 0.0000
                GorR = str(GorR)
                print('read meter done')
                # except:
                #     continue

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 2, 255), 4)
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%% ,%s ,%f' % (object_name, int(scores[i]*100) , GorR ,measurement_val) 
                #label = '%s: %d%% ' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 5) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        if detected and save_result_img == True:
            img_name = f"./opencv_frame_{img_counter[0]}.png"
            cv2.imwrite(img_name, frame)
            img_counter[0]+=1
        
        return frame

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)
    int8_model = (input_details[0]['dtype'] == np.int8)
    input_mean = 127.5
    input_std = 127.5

    # open camera
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)
    if keyboard_input:
        cv2.namedWindow("test")

    #cap.isOpened()
    #cap.running()
    while(cap.isOpened()):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        frame = frame.astype(np.uint8)
        # frame = frame[::-1, ::-1, :]
        # frame = cap.read()
        if frame is None :
            break


        if keyboard_input:
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                frame = _detect(frame)
            
            # 顯示圖片
            cv2.imshow('test', frame)

        else:
            frame = _detect(frame)
            plt.imshow(frame[:,:,::-1])
            plt.show()
            break
        
    # 釋放攝影機
    cap.release()

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

if __name__=='__main__':
    model_path = './tflite/mobilenetV2_model.tflite'
    label_path = './tflite/labelmap.txt'
    conf_th = 0.5
    detect(model_path, label_path, conf_th, keyboard_input=True, save_result_img = True)