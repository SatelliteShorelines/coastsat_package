# How to use this script
# 1. Set the inputs to match the session you want to test
# 2. Modify the thresholds to match the thresholds used to download the data
# - Change these lines to match the thresholds used to download the data
# max_cloud_no_data_cover = 0.9
# max_cloud_cover = 0.8
# 3. Run the script
# - If the script prints "The cloud cover and no data mask is working properly" then the cloud mask is working properly
# - If the script prints "The cloud cover and no data mask is not working properly OR you used the incorrect thresholds that weren't the ones used to download the data" then the cloud mask is not working properly



# load basic modules
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
from typing import List, Dict, Union, Tuple
import time
from functools import wraps
import traceback
from datetime import timezone

# earth engine module
import ee

# modules to download, unzip and stack the images
import requests
from urllib.request import urlretrieve
import zipfile
import shutil
from osgeo import gdal

# additional modules
from datetime import datetime, timedelta
import pytz
from skimage import morphology, transform
from scipy import ndimage

# from tqdm import tqdm
from tqdm.auto import tqdm
import logging
import matplotlib.patches as mpatches

# CoastSat modules
from coastsat import SDS_preprocess, SDS_tools, gdal_merge,SDS_download
from coastsat.SDS_preprocess import preprocess_single

# region of interest (longitude, latitude)
polygon = [[[151.2957545, -33.7012561],
            [151.297557, -33.7388075],
            [151.312234, -33.7390216],
            [151.311204, -33.701399],
            [151.2957545, -33.7012561]]] 
# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = SDS_tools.smallest_rectangle(polygon)
# date range
dates = ['2010-01-01', '2011-01-01']
dates = ['2023-12-01', '2024-01-01']
# satellite missions ['L5','L7','L8','L9','S2']
# sat_list = ['S2']
sat_list = ['L5','L7','L8','L9','S2']
# choose Landsat collection 'C01' or 'C02'
collection = 'C02'
# name of the site
sitename = 'test_cloud_filter_combined_90_cloud_80'
# sitename = 'test_cloud_filter_combined_100_cloud_100'
# sitename = 'test_cloud_filter_combined_100_cloud_100_trial2'
# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'data')
# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath,
         'landsat_collection': collection}

import matplotlib.pyplot as plt


# load a dataset of downloaded images
metadata = SDS_download.get_metadata(inputs) 

# Testing the cloud coverage and no data & cloud mask coverage 
# these should be the thresholds used to download the data if not the program will claim the cloud mask is not working properly
max_cloud_no_data_cover = 0.9
max_cloud_cover = 0.8

# constants for testing purposes
apply_cloud_mask = True
do_cloud_mask= True
cloud_mask_issue = False
s2cloudless_prob = 60 # this MUST BE A WHOLE NUMBER

# flag to check if the cloud mask is working properly
filtered_correct_flag = True

# loop through the satellites in the downloaded metadata
for satname in metadata.keys():
    filepath = SDS_tools.get_filepath(inputs, satname)
    filenames = metadata[satname]["filenames"]
    # loop through the images
    for i in tqdm(
        range(len(filenames)), desc=f"{satname}: Analyzing imagery", leave=True, position=0
    ):
        # get image filename
        fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
        shoreline_date = os.path.basename(fn[0])[:19]
        # load the combined cloud mask and no data mask saved as cloud_mask
        (
            im_ms,
            georef,
            cloud_mask,
            im_extra,
            im_QA,
            im_nodata,
        ) = preprocess_single(fn, satname, cloud_mask_issue, False, 'C02', do_cloud_mask, s2cloudless_prob)   

        # check if the cloud and no data mask exceed the threshold
        cloud_cover_combined = np.sum(cloud_mask) / cloud_mask.size
        if cloud_cover_combined > max_cloud_no_data_cover:
            filtered_correct_flag = False
            # optional plotting of the cloud mask combined with the no data mask
            #     plt.title(f"S2 combined {os.path.basename(fn[0])} cloud mask")
            #     plt.imshow(cloud_mask, cmap='gray')

            #     # Create a legend
            #     labels = ["No Cloud (False)", "Cloud (True)"]
            #     colors = [plt.cm.gray(0), plt.cm.gray(255)]  # Adjust these colors to match your colormap
            #     patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            #     plt.legend(handles=patches)
            #     plt.show()
            print(f"{os.path.basename(fn[0])}: cloud cover combined: {cloud_cover_combined} exceeded max_cloud_no_data_cover: {max_cloud_no_data_cover}")
        
        # isolate the cloud mask from the combined cloud and no data mask called cloud_mask ( yes its confusing I didn't name it)
        cloud_mask_alone = np.logical_xor(cloud_mask, im_nodata)
        
        # compute updated cloud cover percentage (without no data pixels)
        valid_pixels = np.sum(~im_nodata)
        cloud_cover = np.sum(cloud_mask_alone.astype(int)) / valid_pixels.astype(int)
        if cloud_cover > max_cloud_cover:
            filtered_correct_flag = False
        # # optional plotting of the cloud mask alone
        #     print(f"fn: {fn}")
        #     plt.title(f"S2 alone {os.path.basename(fn[0])} cloud_mask_alone")
        #     plt.imshow(cloud_mask_alone, cmap='gray')

        #     # Create a legend
        #     labels = ["No Cloud (False)", "Cloud (True)"]
        #     colors = [plt.cm.gray(0), plt.cm.gray(255)]  # Adjust these colors to match your colormap
        #     patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        #     plt.legend(handles=patches)
        #     plt.show()
            print(f"{os.path.basename(fn[0])}: cloud cover: {cloud_cover} exceeded max_cloud_cover: {max_cloud_cover}")   



       
# if not then we know the script isn't working properly
if filtered_correct_flag:
    print("The cloud cover and no data mask is working properly")
else:
    print("The cloud cover and no data mask is not working properly OR you used the incorrect thresholds that weren't the ones used to download the data")

    
