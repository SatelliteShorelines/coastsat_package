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
# earth engine module
import ee
from coastsat import SDS_download, SDS_tools

# region of interest (longitude, latitude in WGS84)
polygon = [
    [
        [-117.51525453851755, 33.29929946257779],
        [-117.51549193684221, 33.33963611101142],
        [-117.46743525920786, 33.33982627488089],
        [-117.46721999025168, 33.2994893366537],
        [-117.51525453851755, 33.29929946257779],
    ]
]
# can also be loaded from a .kml polygon
# kml_polygon = os.path.join(os.getcwd(), 'examples', 'NARRA_polygon.kml')
# polygon = SDS_tools.polygon_from_kml(kml_polygon)
# convert polygon to a smallest rectangle (sides parallel to coordinate axes)
# polygon = SDS_tools.smallest_rectangle(polygon)

# date range
# dates = ["1984-01-01", "2022-01-01"]
# dates = ["2022-01-01", "2022-12-01"]
# dates = ["2014-06-01", "2015-03-01"]
dates = ["2014-04-29", "2014-06-01"]


# satellite missions
# sat_list = ["L5", "L7", "L8"]
# sat_list = ["L5", "L7", "L8", "L9", "S2"]
sat_list = [
    "L8",
]
collection = "C02"  # choose Landsat collection 'C01' or 'C02'
# name of the site
sitename = "l8_failure2"
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
    "include_T2": True,
}

ee.Initialize()

# before downloading the images, check how many images are available for your inputs
# SDS_download.check_images_available(inputs)
qa_band_Landsat = "QA_PIXEL"
qa_band_S2 = "QA60"
bands_dict = {
    "L5": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
    "L7": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
    "L8": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
    "L9": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
    "S2": ["B2", "B3", "B4", "B8", "s2cloudless", "B11", qa_band_S2],
}

# test data
satname = "L8"
image_id = "LANDSAT/LC08/C02/T2_TOA/LT08_137207_20140501"  # corrupt image
# image_id = "LANDSAT/LC08/C02/T1_TOA/LC08_040037_20141219" # normal image
image_ee = ee.Image(image_id)
im_folder = (
    r"C:\development\doodleverse\coastsat_package\coastsat_package\data\l8_failure2"
)
filepaths = SDS_tools.create_folder_structure(im_folder, satname)

bands_id = bands_dict[satname]
im_bands = image_ee.getInfo()["bands"]
# first delete dimensions key from dictionary
# otherwise the entire image is extracted (don't know why)
for j in range(len(im_bands)):
    del im_bands[j]["dimensions"]
proj_ms = image_ee.select("B1").projection()
ee_region_ms = SDS_download.adjust_polygon(inputs["polygon"], proj_ms)
bands = {}
bands["ms"] = [
    im_bands[_] for _ in range(len(im_bands)) if im_bands[_]["id"] in bands_id
]
fp_ms = filepaths[1]

try:
    SDS_download.download_tif(
        image_ee, ee_region_ms, bands["ms"], fp_ms, satname, image_id=image_id
    )
    raise AssertionError("RequestSizeExceededError was not raised")
except SDS_download.RequestSizeExceededError:
    pass  # The exception was raised, so the test should pass
