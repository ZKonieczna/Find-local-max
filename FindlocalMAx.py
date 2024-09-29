# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:06:17 2021

@author: zukon
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import napari


mat_fname=r"D:\20230123_Oligos_trialBeads\Lifetime_Data\LifetimeImageData.mat"
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
lifetimes=mat_contents['lifetimeImageData']


mat_fname=r"D:\20230123_Oligos_trialBeads\Lifetime_Data\LifetimeAlphaData.mat"
mat_contents2 = sio.loadmat(mat_fname,squeeze_me=True)
intensities=mat_contents2['lifetimeAlphaData']

def see(number):
    plane=lifetimes[number]
    plt.imshow(plane)
   
    return plane

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(intensities[100])
    viewer.add_image(lifetimes[100])
    
labels = viewer.layers["Labels"].data    


intensity_values=[]

for i in range(len(intensities)):
    label_number=1
    intensity_label=intensities[i][labels==label_number]
    intensity_av=intensity_label.mean()
    intensity_values.append(intensity_av)
plt.plot(intensity_values)

def FindMaxima(intensity_values):
  maxima = []
  length = len(intensity_values)
  if length >= 2:
    if intensity_values[0] > intensity_values[1]:
      maxima.append(intensity_values[0])
       
    if length > 3:
      for i in range(1, length-1):     
        if intensity_values[i] > intensity_values[i-1] and intensity_values[i] > intensity_values[i+1]:
          maxima.append(intensity_values[i])

    if intensity_values[length-1] > intensity_values[length-2]:    
      maxima.append(intensity_values[length-1])   
      
     
  return maxima
   
print(FindMaxima(intensity_values))

LocalMin1 = 4620.3130312449375
LocalMin1_index = intensity_values.index(LocalMin1)
print (LocalMin1_index)

LocalMin2 = 1618.5805015036588
LocalMin2_index = intensity_values.index(LocalMin2)
print(LocalMin2_index)

LocalMin3 = 798.8865449966922
LocalMin3_index = intensity_values.index(LocalMin3)
print(LocalMin3_index)

