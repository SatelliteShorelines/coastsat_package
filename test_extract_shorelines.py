# ==========================================================#
# Shoreline extraction from satellite images
# ==========================================================#

# Kilian Vos WRL 2018

# %% 1. Initial settings

# load modules
import os
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.ion()
import pandas as pd
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
from coastsat import (
    SDS_download,
    SDS_preprocess,
    SDS_shoreline,
    SDS_tools,
    SDS_transects,
)

# region of interest (longitude, latitude in WGS84)
polygon = [
    [
        [151.301454, -33.700754],
        [151.311453, -33.702075],
        [151.307237, -33.739761],
        [151.294220, -33.736329],
        [151.301454, -33.700754],
    ]
]
# can also be loaded from a .kml polygon
# kml_polygon = os.path.join(os.getcwd(), 'examples', 'NARRA_polygon.kml')
# polygon = SDS_tools.polygon_from_kml(kml_polygon)
# convert polygon to a smallest rectangle (sides parallel to coordinate axes)
polygon = SDS_tools.smallest_rectangle(polygon)

# date range
# dates = ["1984-01-01", "2022-01-01"]
# dates = ["1984-01-01", "1989-05-01"]
dates = ["2022-01-01", "2022-12-01"]

# satellite missions
# sat_list = ["L5", "L7", "L8"]
# sat_list = ["L5", "L7", "L8", "L9", "S2"]
sat_list = ["L8", "L9", "S2"]
collection = "C02"  # choose Landsat collection 'C01' or 'C02'
# name of the site
sitename = "extract_shorelines_test"

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), "data")

# put all the inputs into a dictionnary
inputs = {
    "polygon": polygon,
    "dates": dates,
    "sat_list": sat_list,
    "sitename": sitename,
    "filepath": filepath_data,
    "landsat_collection": collection,
}

# before downloading the images, check how many images are available for your inputs
# SDS_download.check_images_available(inputs)

# %% 2. Retrieve images

# only uncomment this line if you want Landsat Tier 2 images (not suitable for time-series analysis)
inputs["include_T2"] = True

# retrieve satellite images from GEE
# metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs)

# %% 3. Batch shoreline detection

# settings for the shoreline extraction
settings = {
    # general parameters:
    "cloud_thresh": 0.1,  # threshold on maximum cloud cover
    "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
    "output_epsg": 28356,  # epsg code of spatial reference system desired for the output
    # quality control:
    "check_detection": False,  # if True, shows each shoreline detection to the user for validation
    "adjust_detection": False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    "min_beach_area": 1000,  # minimum area (in metres^2) for an object to be labelled as a beach
    "min_length_sl": 500,  # minimum length (in metres) of shoreline perimeter to be valid
    "cloud_mask_issue": False,  # switch this parameter to True if sand pixels are masked (in black) on many images
    "sand_color": "default",  # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "pan_off": False,  # True to switch pansharpening off for Landsat 7/8/9 imagery
    # add the inputs defined previously
    "inputs": inputs,
    "create_plot": True,  # True create a matplotlib plot of the image with the datetime as the title. False save as a standard JPG
    "remove_cloud_mask": False,  # True, Remove cloud mask from image processing
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
# SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings["reference_shoreline"] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings["max_dist_ref"] = 300

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# remove duplicates (images taken on the same date by the same satellite)
output = SDS_tools.remove_duplicates(output)
# remove inaccurate georeferencing (set threshold to 10 m)
output = SDS_tools.remove_inaccurate_georef(output, 10)

# for GIS applications, save output into a GEOJSON layer
geomtype = "points"  # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
gdf = gdf.set_crs(settings["output_epsg"])
gdf.to_crs("epsg:4326", inplace=True)

if gdf is None:
    raise Exception("output does not contain any mapped shorelines")
# gdf.crs = {"init": "epsg:" + str(settings["output_epsg"])}  # set layer projection
# save GEOJSON layer to file
gdf.to_file(
    os.path.join(
        inputs["filepath"],
        inputs["sitename"],
        "%s_output_%s.geojson" % (sitename, geomtype),
    ),
    driver="GeoJSON",
    encoding="utf-8",
)

geomtype = "lines"  # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
gdf = gdf.set_crs(settings["output_epsg"])
gdf.to_crs("epsg:4326", inplace=True)

if gdf is None:
    raise Exception("output does not contain any mapped shorelines")
# gdf.crs = {"init": "epsg:" + str(settings["output_epsg"])}  # set layer projection
# save GEOJSON layer to file
gdf.to_file(
    os.path.join(
        inputs["filepath"],
        inputs["sitename"],
        "%s_output_%s.geojson" % (sitename, geomtype),
    ),
    driver="GeoJSON",
    encoding="utf-8",
)
