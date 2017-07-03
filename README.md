# Tennis ball tracking

## Report link
https://github.com/Kevinkeya/BallTrackingPublic/blob/master/tracking-tennis-ball.pdf

## Envrioment requirments
Python 3 and OpenCV 3.2 required.


## Introduction
Tennis ball tracking using particle filtering and color mask.

## Command Line
### Video when mask is working
```
python3 ball_tracking.py -v tennis_aviGH3Ca-7c7IM_04093_04250.avi -u 1.5 -l 1.5 -s 5 -d True
```
### Video when mask is not working
```
python3 ball_tracking.py -v tennis_avi9WGK1fSdBJs_00112_00250.avi -u 1.5 -l 1.5 -s 5 -d True
```
### Picking up the frame
In the first window that pops up, pressing `n` to choose next frame, and pressing `r` for choosing current frame to label the tennis ball.
### Lableing the tennis ball.
In the second window, pressing left button of mouse from top left to bottom right to draw a rectangular for the tennis ball. Make the rectangular as small as possible but do not left out the tennis because the radius and color distribution are both important for the algorithms.
Press `c` when you are statisfied and `r` to reset cutting.
### Tracking Windows.
Keep pressing `r` for next frame in the orignal frame. It may be a little slow if the video size is a bit large. There are two other windows showing mask and the likelihood.
