# python3 ball_tracking.py -v tennis_aviGH3Ca-7c7IM_04093_04250.avi -s 5 -u 1.5 -l 1.5 -d True
# 
from collections import deque, defaultdict, OrderedDict
from buildmodel import calculate_distance,calculate_mask_probability,build_color_model, build_color_probabilitymap,initilize_motion_model,build_motion_model,resample_motion_model,calculate_particle_color_prob
import numpy as np
import argparse
# import imutils
import cv2
import time
import math
import sys
# import skvideo.io



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-d","--debug", type=bool,default=False, help="debug model")
ap.add_argument("-u","--upper", type=float,default=1.2, help="upper hsv color ratio")
ap.add_argument("-l","--lower", type=float,default=1.2, help="lower hsv color ratio")
ap.add_argument("-s","--search", type=float, default=3, help="max_serach_distance")
args = vars(ap.parse_args())
max_serach_distance=0;
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points

pts = deque(maxlen=args["buffer"])
debug=args["debug"]
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
    # camera = skvideo.io.vreader(args["video"])
    print("Loading video:\t",args["video"])


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
cropping = False
my_green_range = (0, 0, 0);
center_dict = defaultdict(OrderedDict)
center_list_not_dict = []
radius_dict = defaultdict(OrderedDict)
x_co=0
y_co=0
initial_radius =0

number_of_particles=100
# Normalize likelihood
likelihood_vector=np.full(number_of_particles,1.0/number_of_particles)


def range_check(rftPt,x_length,y_length):    
    for i in range(0,len(rftPt)):
        if refPt[i][0]<0:
            refPt[i]=(0 , refPt[i][1])
        elif refPt[i][0]>x_length:
            refPt[i]=(x_length-1, refPt[i][1])
    
    for i in range(0,len(rftPt)):
        if refPt[i][1]<0:
            refPt[i]=(refPt[i][0],0)
        elif refPt[i][1]>y_length:
            refPt[i]=(refPt[i][0],y_length-1)

def cal_distance(tuple1,tuple2):
    return math.sqrt(abs(tuple1[0]-tuple2[0])**2+abs(tuple1[1]-tuple2[1])**2)

def compare_radius(new_radius,radius):
    print('compare_radius',new_radius,type(new_radius),radius,type(radius))
    if radius<=5 and new_radius<=2*radius and new_radius>=radius/2.0:
        return True
    elif new_radius<=1.6*radius and new_radius>=radius/1.6:
        return True
    else:
        return False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, my_green_range
    global x_co, y_co, max_serach_distance, initial_radius

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)

        y_length=frame.shape[0]
        x_length=frame.shape[1]
        max_serach_distance=min(y_length,x_length)/args["search"]
        
        range_check(refPt,x_length,y_length)

        x_co=(int)((refPt[0][0]+refPt[1][0])/2)
        y_co=(int)((refPt[0][1]+refPt[1][1])/2)
        circle_radius = min(abs(refPt[1][1]-y_co),abs(refPt[0][0]-x_co))
        initial_radius = circle_radius
        print("xco_yco",x_co,y_co)

        s=frame[y_co,x_co]
        my_green_range=s

        for i in range(0,len(s)):
            my_green_range[i]=s[i].item()

        cv2.circle(frame,(x_co,y_co), circle_radius, [s[0].item(),s[1].item(),s[2].item()], -1)

        temp_hsv_color=cv2.cvtColor(np.uint8([[[s[0],s[1],s[2]]]]),cv2.COLOR_BGR2HSV)
        hsv_color=temp_hsv_color[0][0]
        back_bgr_color=cv2.cvtColor(temp_hsv_color,cv2.COLOR_HSV2BGR)
        
        # cv2.circle(frame,(x_co,y_co), 63, [bgr_my_green[0].item(),bgr_my_green[1].item(),bgr_my_green[2].item()], -1)
        cv2.imshow("Frame", frame)
        my_green_range=(hsv_color[0],hsv_color[1],hsv_color[2])
        # print("My_ball_color in side: ",my_green_range)
        


def get_greenUpper(my_green_range,ratio = 1.2):
    H=(float)(my_green_range[0]*ratio)
    S=(float)(my_green_range[1]*ratio)
    V=(float)(my_green_range[2]*ratio)
    if(H>179):
        H=float(179)
    if(S>255):
        S=float(255)
    if(V>255):
        V=float(255)
    return (H,S,V)

def get_greenLower(my_green_range, ratio=1.2):
    H=(float)(my_green_range[0]/ratio)
    S=(float)(my_green_range[1]/ratio)
    V=(float)(my_green_range[2]/ratio)
    return (H,S,V)

def remove_max_value(my_list):
    if(len(my_list)>0):
        max_value=cv2.contourArea(max(my_list,key=cv2.contourArea))
        # print('Max Value:',max_value)
        for i in range(0,len(my_list)):
            # print(my_list[i])
            if cv2.contourArea(my_list[i])== max_value:
                my_list.pop(i)
                break



# --------------------
# Here starts the main function

(grabbed, frame) = camera.read()
if not grabbed:
    print('Failed to catch video')


# Here is where wo go choose the frame to cut the ball
while grabbed:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, we want this
    if key == ord("r"):
        break
    # if the 'n' key is pressed, next frame
    elif key == ord("n"):
        (grabbed, frame) = camera.read()


cv2.destroyAllWindows()

backup_frame=frame
x_length=backup_frame.shape[1]
y_length=backup_frame.shape[0]
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_and_crop)



# Here is where we cut the ball from the frame
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        frame = backup_frame.copy()
    # if the 'c' key is pressed, break from the loop cut this one
    elif key == ord("c"):
        break


# To initialize motion model
particle_motion_model=initilize_motion_model(x_co,y_co)
# Calculate the target patch color historgram
color_patch_target_his=build_color_model(frame[refPt[0][1]:(refPt[1][1]+1),refPt[0][0]:refPt[1][0]+1])

print("My_ball_color: ",my_green_range)
greenLower = get_greenLower(my_green_range,args["lower"])
greenUpper = get_greenUpper(my_green_range,args["upper"])
# print('Lower: ',greenLower,type(greenLower[0]))
# print('Upper: ',greenUpper,type(greenUpper[0]))

cv2.destroyAllWindows()

# Use to store center in each fram, only increment when the ball is detected.
count_flag=0
center_list_not_dict.append((x_co,y_co))
center_dict[count_flag]=(x_co,y_co)
radius_dict[count_flag]=initial_radius
pts.appendleft(center_dict[count_flag])
print('Initial point:\t',center_dict[count_flag],"radius:\t",radius_dict[count_flag])
# keep looping
while True:
    
    # grab the current frame
    (grabbed, frame) = camera.read()
    # Build the color probability map
    rk=build_color_probabilitymap(color_patch_target_his,frame)
    


    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, (greenUpper[0],greenUpper[1],greenUpper[2]))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # print(type(mask),mask.shape)
    # print(mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    rk=build_color_probabilitymap(color_patch_target_his,frame)
    if debug:
        while debug:
    # display the image and wait for a keypress
            cv2.drawContours(mask, cnts, -1, (0,255,0), 3)
            cv2.imshow("Mask", mask)
            cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
            cv2.imshow("name",rk)
            key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                break



        # for i in range(0,len(cnts)):
        #     print(cv2.contourArea(cnts[i]))
    # print('cnts:',len(cnts),type(cnts))
    center = None
    variance_this_turn=20

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        while(len(cnts)>0):
            # print(count_flag,len(cnts))
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            print('Here radius',radius)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # print(center)
            # print(int(x), int(y))
            # break


            # print('max_serach_distance:', max_serach_distance)
            # print('radius dict:\t',radius_dict)
            # print(radius)
            if compare_radius(radius,radius_dict[0]):
            # if True:
                print('Actual distance:\t',cal_distance(center_dict[count_flag],center))

                if cal_distance(center_dict[count_flag],center)<= max_serach_distance:
                    count_flag+=1
                    center_dict[count_flag]=center
                    center_list_not_dict.append(center)
                    radius_dict[count_flag]=radius
                    pts.appendleft(center_dict[count_flag])

                    # my_green_range=hsv[center[1]][center[0]]
                    # greenLower = get_greenLower(my_green_range,args["lower"])
                    # greenUpper = get_greenUpper(my_green_range,args["upper"])
                    # print('New color:',my_green_range)
                    break
                elif cal_distance(center_dict[count_flag],(int(x),int(y)))<=max_serach_distance:
                    count_flag+=1
                    center_dict[count_flag]=(int(x),int(y))
                    center_list_not_dict.append((int(x),int(y)))
                    radius_dict[count_flag]=radius
                    pts.appendleft(center_dict[count_flag])
                    # my_green_range=hsv[int(y)][int(x)]
                    # greenLower = get_greenLower(my_green_range,args["lower"])
                    # greenUpper = get_greenUpper(my_green_range,args["upper"])
                    # print('New color:',my_green_range)
                    break
                # else:
                #     remove_max_value(cnts)

            remove_max_value(cnts)
            try: 
                variance_this_turn = calculate_distance(center,center_list_not_dict[count_flag-1])
            except:
                print(count_flag,len(center_list_not_dict))
                variance_this_turn = 50

            # break

        # only proceed if the radius meets a minimum size
        if len(cnts)>0 and radius > 5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)



    # particle_motion_model=np.append(particle_motion_model,[[center[0],center[1]]],axis=0)
    # Resample it
    particle_motion_model=resample_motion_model(particle_motion_model,likelihood_vector,residual=True)
    # Update new particle with variance
    if variance_this_turn<20:
        variance_this_turn=variance_this_turn*1.5

    if center is not None:
        for i in range(0,10):
            particle_motion_model[i]=(center[1],center[0])

    particle_motion_model=build_motion_model(particle_motion_model,x_length,y_length,variance=35)
    # Enforce the motion_model

    if center is not None:
        for i in range(10,20):
            particle_motion_model[i]=(center[1],center[0])
    # Calculate the likelihodd vector
    likelihood_vector=calculate_particle_color_prob(particle_motion_model,initial_radius,rk,x_length,y_length)
    # 
    likelihodd_vector_mask=calculate_mask_probability(particle_motion_model,initial_radius,center,variance_this_turn)
    # 
    likelihood_vector=np.multiply(likelihood_vector,likelihodd_vector_mask)

    likelihood_vector=np.divide(likelihood_vector,sum(likelihood_vector))

    max_index = np.argmax(likelihood_vector)
    print ('max_index',max_index)
    # update the points queue    

    # loop over the set of tracked points
    # print(pts)
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.0)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    count_flag_pair_temp=0
    for pair in particle_motion_model:
        # Show all particles 
        cv2.circle(frame, (pair[1],pair[0]), initial_radius, (255, 0, 255), 1)
        count_flag_pair_temp+=1
        # if count_flag_pair_temp >=50:
        #     break
    # cv2.circle(frame, (x_length-1,y_length-1), initial_radius, (255, 255, 0), 1)
    # cv2.circle(frame, (int(np.mean(particle_motion_model[:,1])),int(np.mean(particle_motion_model[:,1]))), initial_radius, (255, 0, 255), 5)
    cv2.circle(frame, (particle_motion_model[max_index,1],particle_motion_model[max_index,0]), initial_radius, (255, 0, 255), 5)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()