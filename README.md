# Image Alignment and Warping

Image alignment refers to the technique of alignment of a pair of images of same scene taken by rotating the camera.

The steps for image alignment are:
1.	Local features extraction from the images. For this, we can use SIFT or SURF features.
2.	Similarity (distance) computation between every descriptors of one image with the descriptors of the other image.
3.	Selection of best matches of the features.
4.	Run RANSAC to perform Homography estimation.
5.	Warping of the images


## Pre-requisites:
Python and OpenCV. The code has been implemented on Python 3.6 and OpenCV3.0

## Features Extraction
SIFT (Scale Invariant Features Transform) are extracted from the images.

## Distance Computation (Features Matching)
FLANN (Fast Library for Approximate Nearest Neighbors) is used. It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features

## Estimate Homograph
FLANN (Fast Library for Approximate Nearest Neighbors) is used. It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features

## Image Warping
Alignment of the image pair so that the features line up perfectly.
Running the program

> python Image_Align.py

