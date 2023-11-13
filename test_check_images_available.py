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

SDS_download.check_images_available(inputs)
