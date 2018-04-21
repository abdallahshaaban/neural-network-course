# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:38:15 2018

@author: Lenovo-PC
"""
from PIL import Image
import cv2 

img = cv2.imread("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training\\Model1 - Cat.jpg",0)
GrayImage = cv2.resize(img, (50, 50)) 