#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:22:57 2021
@author: Mathew
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
from skimage.io import imread
from skimage import filters,measure
import matplotlib.pyplot as plt
import napari
from scipy.interpolate import make_interp_spline, BSpline
from scipy.spatial import distance





# Convert to wavelength- thede are from the fits to the TS Bead Data. 
m=0.5288
c=430


# Read the files 

mat_fname=r"D:\20240417_ZK-25-Dis1Well6_ROI2_hist2_no_laser\Lifetime_Data\LifetimeImageData.mat"
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
lifetimes=mat_contents['lifetimeImageData']


mat_fname=r"D:\20240417_ZK-25-Dis1Well6_ROI2_hist2_no_laser\Lifetime_Data\LifetimeAlphaData.mat"
mat_contents2 = sio.loadmat(mat_fname,squeeze_me=True)
intensities=mat_contents2['lifetimeAlphaData']


# This is to make an summed intensity image over all wavelengths to perfom the thresholding on
sum_int=np.sum(intensities[0:512],axis=0)
plt.imshow(sum_int)
plt.colorbar()
plt.savefig('BigBeads.jpg', dpi=1200)
plt.show()


# The below just thresholds the image based on intensity value - could also use Otsu method
thresh=2000000
binary_im=sum_int>thresh
plt.imshow(binary_im)
plt.show()

# Now get the stack only with the thresholded intensities or lifetimes present:
thresholded_intensities=binary_im*intensities
thresholded_lifetimes=binary_im*lifetimes

thresholded_intenisties_sum=np.sum(thresholded_intensities, axis=0)
twod_intensities=thresholded_intenisties_sum

plt.imshow(twod_intensities)
plt.colorbar()
plt.show()


#get an intensity spectrum for selected pixels

intensities_only_thresh=intensities*binary_im
intensity_wl=[]
int_sdev=[]
wl=[]
for i in range(0,512):
    wavelength_val=i*m+c
    plane=intensities_only_thresh[i]
    plane_list=plane.flatten()
    values_only=plane_list[plane_list>0]
    intensity_mean=values_only.mean()
    intensity_sdev=values_only.std()
    
    intensity_wl.append(intensity_mean)
    int_sdev.append(intensity_sdev)
    wl.append(wavelength_val)
    
    
plt.plot(wl,intensity_wl)
#plt.savefig('BigBeads_int.jpg', dpi=1200)

#smooth out the curve to get rid of noise

#ave_int=np.asarray(intensity_wl) #change list to np array
#int_sdev_new=np.asarray(int_sdev)
#wl_int=np.asarray(wl)
#std_intensity=np.std(mask, axis=(1,2))

#wlnew=np.linspace(wl_int.min(), wl_int.max(), 100) #generate new x axis for smooth curve

#spl=make_interp_spline(wl_int, ave_int, k=3) #interpolate 
#ave_int_smooth=spl(wlnew) 

#spl_sdev=make_interp_spline(wl_int, int_sdev_new, k=3)
#sdev_smooth=spl_sdev(wlnew)

#plt.plot(wlnew, ave_int_smooth) #use new parameters
#plt.fill_between(wlnew,ave_int_smooth-sdev_smooth, ave_int_smooth+sdev_smooth,alpha=0.1, edgecolor='#4b4b4b', facecolor='#4b4b4b', antialiased=True)
#plt.xlim([470,730])
#plt.savefig('Spectrum_smooth.pdf', dpi=1200)
#plt.show()

x_axis=np.linspace(0,511,512)
x_axis_correct=c+x_axis*m

plt.plot(x_axis_correct,intensity_mean)
plt.fill_between(x_axis_correct,intensity_mean-intensity_sdev, intensity_mean+intensity_sdev,alpha=0.1, edgecolor='#ff0000', facecolor='#ff0000', antialiased=True)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
##plt.ylim([0,1000])
#plt.xlim([500,700])
#plt.legend()
plt.show()   



twod_intensities=twod_intensities/twod_intensities.max()
plt.imshow(twod_intensities)
plt.colorbar()
plt.show()

#trying to get rid of 0 values
twod_intensities=twod_intensities>0
twod_intensities=twod_intensities/twod_intensities.max()
plt.imshow(twod_intensities)
plt.colorbar()
plt.show()

range_lifetimes=thresholded_lifetimes[65:145]
twod_lifetime=np.mean(range_lifetimes,axis=0)
#bright_pixel_lifetime=binary*twod_lifetime
plt.imshow(twod_lifetime, cmap='jet_r', vmin=0, vmax=1)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(1)
ax.set_facecolor('black')

m=ax.imshow(twod_lifetime,cmap='plasma',vmin=1.8,vmax=2.4,alpha=twod_intensities)
fig.colorbar(m)
fig.savefig('Big beads peak 1.jpg',dpi=1200)
plt.show()

# This is to make a summed intensity image over defined rage/all wavelengths - axis=0
intensities=intensities[0:512]
sum_int=np.sum(intensities,axis=0)
plt.imshow(sum_int)
plt.colorbar()
plt.savefig('GiantBeads.jpg', dpi=1200)
plt.show()


#mask=sum_int>25000
#inclusions=mask*intensities

#plt.imshow(inclusions)
#plt.show()
# Sum intensity over all pixels across wavelengths
int_sum=np.sum(intensities,axis=(1,2))

#Correct x-axis to represent wavelengths and not sensor pixels
x_axis=np.linspace(0,511,512)
x_axis_correct=c+m*x_axis
plt.plot(x_axis_correct,int_sum)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.savefig('GiantBeads_int.jpg', dpi=1200)
plt.show()

#Show intensity across a couple pixels

textstr='%s nm to %s nm - intensity'%(510,550)

range_intensities=intensities[65:145]
twod_lifetime=np.mean(range_intensities,axis=0)
#binary=twod_lifetime>3000
#binary_dim=twod_lifetime<3000
plt.imshow(twod_lifetime)

intensity_all=twod_lifetime.flatten()

plt.colorbar()
plt.title(textstr)
plt.show()

#Show lifetimes across a couple of pixels

textstr='%s nm to %s nm - lifetime'%(510,550)

range_lifetimes=lifetimes[65:145]
twod_lifetime=np.mean(range_lifetimes,axis=0)
plt.imshow(twod_lifetime)
plt.colorbar()
plt.show()

#bright_pixel_lifetime=binary*twod_lifetime
all=twod_lifetime.flatten()
plt.hist(all, bins=10, range=[0.5,1.5])
plt.title(textstr)
plt.show()

#dim_pixel_lifetime=binary_dim*twod_lifetime
#dim_all=dim_pixel_lifetime.flatten()
#plt.hist(dim_all, bins=50, range=[0.2,2.0])
#plt.show()

#lifetime_all=twod_lifetime.flatten()
#plt.plot(lifetime_all, intensity_all)
#plt.show()


plt.imshow(twod_lifetime, cmap='jet_r', vmin=0.5, vmax=3)
plt.colorbar()
plt.title(textstr)
plt.show()
list_lifetimes=twod_lifetime.flatten()
real_lifteimes=list_lifetimes[(list_lifetimes>0)]
plt.hist(real_lifteimes, bins = 25,range=[0,1.5], rwidth=0.9,ec='black',color='darkmagenta',alpha=0.8)
plt.ylabel('Number of pixels')
plt.xlabel('Lifetime (ns)')
#plt.title(textstr)
#plt.savefig('ADOTA3_LF_hist_magenta_SI.pdf', dpi=1200)
plt.show()


#alpha masked image below

intensities=thresholded_intensities[100:490]
sum_int=np.sum(intensities,axis=0)
plt.imshow(sum_int)
plt.colorbar()
#plt.savefig('Brain3Intensity3.pdf',dpi=1200)
plt.show()

range_lf=thresholded_lifetimes[100:490]
twod_lf=np.mean(range_lf,axis=0)
plt.imshow(twod_lf, cmap='jet_r',vmin=0, vmax=1.5)
plt.colorbar()
#plt.savefig('Brain_colourbar3.jpg',dpi=1200)
plt.show()


fig, ax = plt.subplots(1)
ax.set_facecolor('black')

m=ax.imshow(twod_lf,cmap='jet_r',vmin=0,vmax=1.5,alpha=sum_int)
#fig.savefig('Lifetimes_Brain3.jpg',dpi=1200)
plt.show()


all=twod_lf.flatten()
plt.hist(all, bins=80, range=[0.2,2])
plt.xlim([1,2])
plt.xlabel('Lifetime (ns)')
plt.ylabel('Frequency')
#plt.savefig('histogramIV_ROI4.jpg',dpi=1200)
plt.show()


















# Now to analyse some of the features.

labelled_image=measure.label(binary_im)
number_of_clusters=labelled_image.max()

# Make the arrays for the periphery vs. centre. There looks to be a difference between the outer and inner
# part of the droplets, hence need to separate.

periphery_image=np.zeros(labelled_image.shape)
centre_image=np.zeros(labelled_image.shape)

# Perform stats on the image:
measure_image=measure.regionprops_table(labelled_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length'))
xcoord=measure_image["centroid-0"]
ycoord=measure_image["centroid-1"]
lengths=measure_image["major_axis_length"]  

# Go through each of clusters.
for num in range(1,number_of_clusters):
    
    # Ideally want to make a plot that shows distance from the centre point.
    distance_image=np.zeros(labelled_image.shape)
    
    # Select only the one droplet
    image_to_show=labelled_image==num
    
    # Make an image with just the coordinates
    wid=image_to_show.shape
    x = np.linspace(0, wid[0],wid[0])
    y = np.linspace(0, wid[1],wid[1])
    
    xv, yv = np.meshgrid(x, y)
    
    # Calculate the distances from the centre to each point in the droplet using the coordinate system. 
    image_dist=((yv-xcoord[num-1])**2+(xv-ycoord[num-1])**2)**(0.5)
    image_dist_clust=image_dist*image_to_show
 
    # This is the threshold that determines whether the pixel is in the periphery or the centre. 
    length=0.6*(lengths[num-1]/2)
    
    # Now make the image
    image_periphery=image_dist_clust>length
    image_centre=(image_dist_clust<=length)*image_to_show
    
    # Add to overall images that contain all of the clusters. 
    periphery_image=periphery_image+image_periphery
    centre_image=centre_image+image_centre

# Generate wavelength images
periphery_image_wl=periphery_image*max_int
centre_image_wl=centre_image*max_int 


# Show the plots. 
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(periphery_image)
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")
axes[1].imshow(centre_image)


fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(periphery_image_wl,cmap='rainbow',vmin=460,vmax=560)
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")
axes[1].imshow(centre_image_wl,cmap='rainbow',vmin=460,vmax=560)    

# Make histograms for periphery and centres. 
periph=periphery_image_wl.flatten()
cents=centre_image_wl.flatten()


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.7)
axes[0].hist(periph, bins = 20,range=[450,550], rwidth=0.9,color='#0000ff')
axes[1].hist(cents, bins = 20,range=[450,550], rwidth=0.9,color='#ff0000')
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")
axes[0].set_xlabel('Wavelength (nm)')
axes[1].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Number of Features')
axes[1].set_ylabel('Number of Features')


intensities_only_thresh=intensities*binary_im
intensity_wl=[]
wl=[]
for i in range(0,512):
    wavelength_val=i*m+c
    plane=intensities_only_thresh[i]
    plane_list=plane.flatten()
    values_only=plane_list[plane_list>0]
    intensity_mean=values_only.mean()
    
    intensity_wl.append(intensity_mean)
    wl.append(wavelength_val)
    
    
plt.plot(wl,intensity_wl)
    
    



lifetimes_thresh=binary_im*lifetimes


def lifetime_hists(wavelength1,wavelength2):
    
    wavelength1pos=int((wavelength1-c)/m)
    wavelength2pos=int((wavelength2-c)/m)
    
    print(wavelength1pos)
    print(wavelength2pos)
    
    test_lifetime=lifetimes_thresh[wavelength1pos:wavelength2pos]
    twod_lifetime=np.mean(test_lifetime,axis=0)

    plt.imshow(twod_lifetime,vmin=1,vmax=2)
    plt.colorbar()
    plt.show()
    
    
    textstr='%s nm to %s nm'%(wavelength1,wavelength2)
    list_lifetimes=twod_lifetime.flatten()
    real_lifteimes=list_lifetimes[(list_lifetimes>0)]
    
    plt.hist(real_lifteimes, bins = 100,range=[1,2], rwidth=0.9,ec='black',color='#ff0000',alpha=0.8)
    plt.xlabel('Number of pixels')
    plt.ylabel('Lifetime (ns)')
    plt.title(textstr)