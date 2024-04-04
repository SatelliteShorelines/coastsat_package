"""
This module contains all the functions needed to download the satellite images
from the Google Earth Engine server

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""


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


# CoastSat modules
from coastsat import SDS_preprocess, SDS_tools, gdal_merge

np.seterr(all="ignore")  # raise/ignore divisions by 0 and nans
gdal.PushErrorHandler("CPLQuietErrorHandler")


def release_logger(logger):
    """
    Release the logger and its associated file handlers.

    :param logger: The logger object to be released.
    """
    # Remove all handlers associated with the logger
    for handler in logger.handlers[:]:
        # Close the handler if it's a FileHandler
        if isinstance(handler, logging.FileHandler):
            handler.close()
        # Remove the handler from the logger
        logger.removeHandler(handler)


def setup_logger(
    folder,
    base_filename="download_report",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    # Determine the next available log file number
    i = 1
    while True:
        log_filename = f"{base_filename}{i}.txt" if i > 1 else f"{base_filename}.txt"
        log_filepath = os.path.join(folder, log_filename)
        if not os.path.exists(log_filepath):
            break
        i += 1

    # Create a custom logger
    logger = logging.getLogger("satellite_download_logger")
    logger.setLevel(logging.INFO)  # Log all levels of messages

    # Create handlers
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    log_format = logging.Formatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger


# Custom exception classes
class RequestSizeExceededError(Exception):
    pass


class TooManyRequests(Exception):
    pass


# def retry_deprecated(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         max_tries = 3
#         # Get image_id from kwargs or use 'Unknown'
#         image_id = kwargs.get("image_id", "Unknown image id")
#         logger = kwargs.get("logger", None)
#         delay = 1
#         # attempt to download the image up to max_tries times
#         for tries in range(max_tries):
#             try:
#                 print(f"calling {func.__name__}")
#                 return func(*args, **kwargs)
#             except Exception as e:
#                 print(print("CLASS", type(e)))
#                 if logger:
#                     logger.warning(
#                         f"Retry {tries + 1}/{max_tries} for function {func.__name__} with image_id {kwargs.get('image_id', 'N/A')} due to {e}"
#                     )
#                 if tries + 1 < max_tries:
#                     # wait with exponential backoff
#                     print(
#                         f"Retry {tries + 1}/{max_tries} for function {func.__name__} with image_id {kwargs.get('image_id', 'N/A')} due to {type(e).__name__} error."
#                     )
#                     time.sleep(delay)
#                     # delay_multiplier *= backoff
#                 else:
#                     # Re-raise the last exception if max retries have been exceeded )(i.e. no more retries)
#                     print(
#                         f"Max retries {tries + 1}/{max_tries}  exceeded for {func.__name__} due to {type(e).__name__}"
#                     )
#                     raise TooManyRequests(
#                         f"Failed to process {image_id} after {max_tries} attempts due to: {e}"
#                     )

#     return wrapper


import functools  # retry v2


def retry(func):
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        # Get image_id from kwargs or use 'Unknown'
        image_id = kwargs.get("image_id", "Unknown image id")
        logger = kwargs.get("logger", None)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # first attempt does not wait
                if attempt > 0:
                    time.sleep(1)
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f"Attempt {attempt+1}/{max_attempts} failed with error: {type(e).__name__}"
                )
                if logger:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_attempts} for function {func.__name__} with image_id {kwargs.get('image_id', 'N/A')} due to {e}"
                    )
                if attempt == max_attempts - 1:
                    print(
                        f"Max retries {attempt+1}/{max_attempts}  exceeded for {func.__name__} due to {type(e).__name__}"
                    )
                    raise

    return wrapper_retry


def debug_kill_wifi():
    import os

    os.system("netsh wlan disconnect")


@retry
def remove_dimensions_from_bands(image_ee, **kwargs):
    # first delete dimensions key from dictionary
    # otherwise the entire image is extracted (don't know why)
    im_bands = image_ee.getInfo()["bands"]
    
    # remove some additional masks provided with S2
    im_bands = [band for band in im_bands if 'MSK_CLASSI' not in band['id']]
    for j in range(len(im_bands)):
        if "dimensions" in im_bands[j]:
            del im_bands[j]["dimensions"]
    return im_bands


def get_image_quality(satname: str, im_meta: dict) -> str:
    """
    Get the image quality for a given satellite name and image metadata.
    If the image quality field is missing 'NA' is returned

    Args:
        satname (str): Satellite name identifier (e.g. 'L5', 'L7', 'L8', 'L9', 'S2')
        im_meta (dict): Image metadata containing the properties field

    Returns:
        im_quality (str): Image quality value from the metadata based on satellite name

    Satellite information:
        L5, L7: Landsat 5 and Landsat 7
        L8, L9: Landsat 8 and Landsat 9
        S2: Sentinel-2
    """
    # Mapping of satellite names to their respective quality keys
    quality_keys = {
        "L5": "IMAGE_QUALITY",
        "L7": "IMAGE_QUALITY",
        "L8": "IMAGE_QUALITY_OLI",
        "L9": "IMAGE_QUALITY_OLI",
    }
    if satname in quality_keys:
        return im_meta["properties"].get(quality_keys[satname], "NA")
    elif satname == "S2":
        # List of possible quality flag names for Sentinel-2
        flag_names = ["RADIOMETRIC_QUALITY", "RADIOMETRIC_QUALITY_FLAG"]
        # Try to find the first existing quality flag in the metadata properties
        for flag in flag_names:
            if flag in im_meta["properties"]:
                return im_meta["properties"][flag]
        # If no quality flag is found, log a warning and default to "PASSED"
        print(
            "WARNING: could not find Sentinel-2 geometric quality flag,"
            + " raise an issue at https://github.com/kvos/CoastSat/issues"
            + " and add your inputs in text (not a screenshot, please)."
        )
        return "PASSED"


def get_georeference_accuracy(satname: str, im_meta: dict) -> str:
    """
    Get the accuracy of geometric reference based on the satellite name and
    the image metadata.

    Landsat default value of accuracy is RMSE = 12m

    Sentinel-2 don't provide a georeferencing accuracy (RMSE as in Landsat), instead the images include a quality control flag in the metadata that
    indicates georeferencing accuracy: a value of 1 signifies a passed geometric check, while -1 denotes failure(i.e., the georeferencing is not accurate).
    This flag is stored in the image's metadata, which is additional information about the image stored with it. However, the specific property or field in the metadata where this flag is stored can vary across the Sentinel-2 archive, meaning it's not always in the same place or under the same name.

    Parameters:
    satname (str): Satellite name, e.g., 'L5', 'L7', 'L8', 'L9', or 'S2'.
    im_meta (dict): Image metadata containing the properties for geometric
                    reference accuracy.

    Returns:
    str: The accuracy of the geometric reference.
    """
    if satname in ["L5", "L7", "L8", "L9"]:
        # average georefencing error across Landsat collection (RMSE = 12m)
        acc_georef = im_meta["properties"].get("GEOMETRIC_RMSE_MODEL", 12)
    elif satname == "S2":
        flag_names = [
            "GEOMETRIC_QUALITY_FLAG",
            "GEOMETRIC_QUALITY",
            "quality_check",
            "GENERAL_QUALITY_FLAG",
        ]
        for flag in flag_names:
            if flag in im_meta["properties"]:
                return im_meta["properties"][flag]
        # if none fo the flags appeared then set the acc_georef to passed
        acc_georef = "PASSED"
        # acc_georef = -1 # old value returned before update
    return acc_georef


def handle_duplicate_image_names(
    all_names: List[str],
    bands: Dict[str, str],
    im_fn: Dict[str, str],
    im_date: str,
    satname: str,
    sitename: str,
    suffix: str,
) -> Dict[str, str]:
    """
    This function handles duplicate names for image files. If an image file name is already in use,
    it adds a '_dupX' suffix to the name (where 'X' is a counter for the number of duplicates).

    Parameters:
    all_names (list): A list containing all image file names that have been handled so far.
    bands (dict): A dictionary where the keys are the band names.
    im_fn (dict): A dictionary where the keys are the band names and the values are the current file names for each band.
    im_date (str): A string representing the date when the image was taken.
    satname (str): A string representing the name of the satellite that took the image.
    sitename: A string representing name of directory containing all downloaded images
    suffix (str): A string representing the file extension or other suffix to be added to the file name.

    Returns:
    im_fn (dict): The updated dictionary where the keys are the band names and the values are the modified file names for each band.
    """
    # if multiple images taken at the same date add 'dupX' to the name (duplicate)
    duplicate_counter = 0
    while im_fn["ms"] in all_names:
        duplicate_counter += 1
        for key in bands.keys():
            im_fn[key] = (
                f"{im_date}_{satname}_{sitename}"
                f"_{key}_dup{duplicate_counter}{suffix}"
            )
    return im_fn


def filter_bands(bands_source, band_ids):
    """Filters the bands from bands_source based on the given band_ids."""
    return [band for band in bands_source if band["id"] in band_ids]


def merge_image_tiers(inputs, im_dict_T1, im_dict_T2):
    """
    Merges im_dict_T2 into im_dict_T1 based on the keys provided in inputs["sat_list"].
    Parameters:
    - inputs: Dictionary with "sat_list" as one of its keys containing a list of keys to check.
    - im_dict_T1: First dictionary to merge into.
    - im_dict_T2: Second dictionary containing values to be added to im_dict_T1.
    Returns:
    - Updated im_dict_T1 after merging.
    """

    # Merge tier 2 imagery into dictionary if include_T2 is True
    if inputs.get("include_T2", False):
        for key in inputs["sat_list"]:
            if key in ["S2", "L9"]:
                continue
            else:
                # Check if key exists in both dictionaries
                if key in im_dict_T1 and key in im_dict_T2:
                    im_dict_T1[key] += im_dict_T2[key]
                elif key in im_dict_T2:  # If key only exists in im_dict_T2
                    im_dict_T1[key] = im_dict_T2[key]  # Add it to im_dict_T1

    return im_dict_T1


def retrieve_images(
    inputs,
    cloud_threshold: float = 99.9,
    cloud_mask_issue: bool = False,
    save_jpg: bool = True,
    apply_cloud_mask: bool = True,
):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2
    covering the area of interest and acquired between the specified dates.
    The downloaded images are in .TIF format and organised in subfolders, divided
    by satellite mission. The bands are also subdivided by pixel resolution.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded

    save_jpg: bool:
        save jpgs for each image downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # create a new directory for this site with the name of the site
    im_folder = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(im_folder):
        os.makedirs(im_folder)
    # Initialize the logger
    logger = setup_logger(im_folder)
    # initialise connection with GEE server
    ee.Initialize()

    # check image availabiliy and retrieve list of images
    im_dict_T1, im_dict_T2 = check_images_available(inputs)

    # merge the two image collections tiers into a single dictionary
    im_dict_T1 = merge_image_tiers(inputs, im_dict_T1, im_dict_T2)

    # remove UTM duplicates in S2 collections (they provide several projections for same images)
    if "S2" in inputs["sat_list"] and len(im_dict_T1["S2"]) > 0:
        im_dict_T1["S2"] = filter_S2_collection(im_dict_T1["S2"])
        # get s2cloudless collection
        im_dict_s2cloudless = get_s2cloudless(im_dict_T1["S2"], inputs)

    # bands for each mission
    if inputs["landsat_collection"] == "C01":
        qa_band_Landsat = "BQA"
    elif inputs["landsat_collection"] == "C02":
        qa_band_Landsat = "QA_PIXEL"
    else:
        raise Exception(
            "Landsat collection %s does not exist, " % inputs["landsat_collection"]
            + "choose C01 or C02."
        )
    bands_dict = {
        "L5": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
        "L7": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
        "L8": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
        "L9": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
        "S2": ["B2", "B3", "B4", "B8", "s2cloudless", "B11", "QA60"],
    }

    # main loop to download the images for each satellite mission
    # print('\nDownloading images:')
    suffix = ".tif"
    count = 1
    num_satellites = len(im_dict_T1.keys())
    for satname in tqdm(
        im_dict_T1.keys(), desc=f"Downloading Imagery for {num_satellites} satellites"
    ):
        count += 1
        # create subfolder structure to store the different bands
        filepaths = SDS_tools.create_folder_structure(im_folder, satname)
        # initialise variables and loop through images
        bands_id = bands_dict[satname]
        all_names = []  # list for detecting duplicates
        # loop through each image
        pbar = tqdm(
            range(len(im_dict_T1[satname])),
            desc=f"Downloading Imagery for {satname}",
            leave=True,
        )
        for i in pbar:
            try:
                # initalize the variables
                # filepath (fp) for the multispectural file
                fp_ms = ""
                # store the bands availble
                bands = dict([])
                # dictionary containing the filepaths for each type of file downloaded
                im_fn = dict([])

                # get image metadata
                im_meta = im_dict_T1[satname][i]

                # get time of acquisition (UNIX time) and convert to datetime
                acquisition_time = im_meta["properties"]["system:time_start"]
                im_timestamp = datetime.fromtimestamp(
                    acquisition_time / 1000, tz=pytz.utc
                )
                im_date = im_timestamp.strftime("%Y-%m-%d-%H-%M-%S")

                # get epsg code
                im_epsg = int(im_meta["bands"][0]["crs"][5:])

                # get quality flags (geometric and radiometric quality)
                accuracy_georef = get_georeference_accuracy(satname, im_meta)
                image_quality = get_image_quality(satname, im_meta)

                # select image by id
                image_ee = ee.Image(im_meta["id"])
                # for S2 add s2cloudless probability band
                if satname == "S2":
                    if len(im_dict_s2cloudless[i]) == 0:
                        raise Exception(
                            "could not find matching s2cloudless image, raise issue on Github at"
                            + "https://github.com/kvos/CoastSat/issues and provide your inputs."
                        )
                    im_cloud = ee.Image(im_dict_s2cloudless[i]["id"])
                    cloud_prob = im_cloud.select("probability").rename("s2cloudless")
                    image_ee = image_ee.addBands(cloud_prob)

                # update the loading bar with the status
                pbar.set_description_str(
                    desc=f"{satname}: Loading bands for {i}th image ", refresh=True
                )
                # first delete dimensions key from dictionary
                # otherwise the entire image is extracted (don't know why)
                im_bands = remove_dimensions_from_bands(
                    image_ee, image_id=im_meta["id"], logger=logger
                )

                # =============================================================================================#
                # Landsat 5 download
                # =============================================================================================#
                if satname == "L5":
                    fp_ms = filepaths[1]
                    fp_mask = filepaths[2]
                    # select multispectral bands
                    bands["ms"] = [
                        im_bands[_]
                        for _ in range(len(im_bands))
                        if im_bands[_]["id"] in bands_id
                    ]
                    # adjust polygon to match image coordinates so that there is no resampling
                    proj = image_ee.select("B1").projection()
                    pbar.set_description_str(
                        desc=f"{satname}: adjusting polygon {i}th image ", refresh=True
                    )
                    ee_region = adjust_polygon(
                        inputs["polygon"], proj, image_id=im_meta["id"], logger=logger
                    )
                    # download .tif from EE (one file with ms bands and one file with QA band)
                    pbar.set_description_str(
                        desc=f"{satname}: Downloading tif for {i}th image ",
                        refresh=True,
                    )
                    fn_ms, fn_QA = download_tif(
                        image_ee,
                        ee_region,
                        bands["ms"],
                        fp_ms,
                        satname,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    # create filename for image
                    for key in bands.keys():
                        im_fn[key] = (
                            im_date
                            + "_"
                            + satname
                            + "_"
                            + inputs["sitename"]
                            + "_"
                            + key
                            + suffix
                        )
                    # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
                    im_fn = handle_duplicate_image_names(
                        all_names,
                        bands,
                        im_fn,
                        im_date,
                        satname,
                        inputs["sitename"],
                        suffix,
                    )
                    im_fn["mask"] = im_fn["ms"].replace("_ms", "_mask")
                    filename_ms = im_fn["ms"]
                    all_names.append(im_fn["ms"])

                    # resample ms bands to 15m with bilinear interpolation
                    fn_in = fn_ms
                    fn_target = fn_ms
                    fn_out = os.path.join(fp_ms, im_fn["ms"])
                    pbar.set_description_str(
                        desc=f"{satname}: Transforming {i}th image ", refresh=True
                    )
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=True,
                        resampling_method="bilinear",
                    )

                    # resample QA band to 15m with nearest-neighbour interpolation
                    fn_in = fn_QA
                    fn_target = fn_QA
                    fn_out = os.path.join(fp_mask, im_fn["mask"])
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=True,
                        resampling_method="near",
                    )

                    # delete original downloads
                    for original_file in [fn_ms, fn_QA]:
                        os.remove(original_file)
                    if save_jpg:
                        # location of the tif folder for that satellite
                        # For ex. S2 contains 2 tif folders /ms /swir /mask
                        tif_paths = SDS_tools.get_filepath(inputs, satname)
                        SDS_preprocess.save_single_jpg(
                            filename=im_fn["ms"],
                            tif_paths=tif_paths,
                            satname=satname,
                            sitename=inputs["sitename"],
                            cloud_thresh=cloud_threshold,
                            cloud_mask_issue=cloud_mask_issue,
                            filepath_data=inputs["filepath"],
                            collection=inputs["landsat_collection"],
                            apply_cloud_mask=apply_cloud_mask,
                        )

                # =============================================================================================#
                # Landsat 7, 8 and 9 download
                # =============================================================================================#
                elif satname in ["L7", "L8", "L9"]:
                    fp_ms = filepaths[1]
                    fp_pan = filepaths[2]
                    fp_mask = filepaths[3]
                    # if C01 is selected, for images after 2022 adjust the name of the QA band
                    # as the name has changed for Collection 2 images (from BQA to QA_PIXEL)
                    if inputs["landsat_collection"] == "C01":
                        if not "BQA" in [_["id"] for _ in im_bands]:
                            bands_id[-1] = "QA_PIXEL"
                    # select bands (multispectral and panchromatic)
                    bands["ms"] = [
                        im_bands[_]
                        for _ in range(len(im_bands))
                        if im_bands[_]["id"] in bands_id
                    ]
                    bands["pan"] = [
                        im_bands[_]
                        for _ in range(len(im_bands))
                        if im_bands[_]["id"] in ["B8"]
                    ]
                    # adjust polygon for both ms and pan bands
                    proj_ms = image_ee.select("B1").projection()
                    proj_pan = image_ee.select("B8").projection()
                    pbar.set_description_str(
                        desc=f"{satname}: adjusting polygon {i}th image ", refresh=True
                    )
                    ee_region_ms = adjust_polygon(
                        inputs["polygon"],
                        proj_ms,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    ee_region_pan = adjust_polygon(
                        inputs["polygon"],
                        proj_pan,
                        image_id=im_meta["id"],
                        logger=logger,
                    )

                    # download both ms and pan bands from EE
                    pbar.set_description_str(
                        desc=f"{satname}: Downloading tif for {i}th image ",
                        refresh=True,
                    )
                    fn_ms, fn_QA = download_tif(
                        image_ee,
                        ee_region_ms,
                        bands["ms"],
                        fp_ms,
                        satname,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    fn_pan = download_tif(
                        image_ee,
                        ee_region_pan,
                        bands["pan"],
                        fp_pan,
                        satname,
                        image_id=im_meta["id"],
                    )
                    # create filename for both images (ms and pan)
                    for key in bands.keys():
                        im_fn[key] = (
                            im_date
                            + "_"
                            + satname
                            + "_"
                            + inputs["sitename"]
                            + "_"
                            + key
                            + suffix
                        )
                    # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
                    pbar.set_description_str(
                        desc=f"{satname}: remove duplicates for {i}th image ",
                        refresh=True,
                    )
                    im_fn = handle_duplicate_image_names(
                        all_names,
                        bands,
                        im_fn,
                        im_date,
                        satname,
                        inputs["sitename"],
                        suffix,
                    )
                    im_fn["mask"] = im_fn["ms"].replace("_ms", "_mask")
                    filename_ms = im_fn["ms"]
                    all_names.append(im_fn["ms"])

                    # resample the ms bands to the pan band with bilinear interpolation (for pan-sharpening later)
                    fn_in = fn_ms
                    fn_target = fn_pan
                    fn_out = os.path.join(fp_ms, im_fn["ms"])
                    pbar.set_description_str(
                        desc=f"{satname}: Transforming {i}th image ", refresh=True
                    )
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=False,
                        resampling_method="bilinear",
                    )

                    # resample QA band to the pan band with nearest-neighbour interpolation
                    fn_in = fn_QA
                    fn_target = fn_pan
                    fn_out = os.path.join(fp_mask, im_fn["mask"])
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=False,
                        resampling_method="near",
                    )

                    # rename pan band
                    try:
                        os.rename(fn_pan, os.path.join(fp_pan, im_fn["pan"]))
                    except:
                        os.remove(os.path.join(fp_pan, im_fn["pan"]))
                        os.rename(fn_pan, os.path.join(fp_pan, im_fn["pan"]))
                    # delete original downloads
                    for _ in [fn_ms, fn_QA]:
                        os.remove(_)

                    if save_jpg:
                        tif_paths = SDS_tools.get_filepath(inputs, satname)
                        SDS_preprocess.save_single_jpg(
                            filename=im_fn["ms"],
                            tif_paths=tif_paths,
                            satname=satname,
                            sitename=inputs["sitename"],
                            cloud_thresh=cloud_threshold,
                            cloud_mask_issue=cloud_mask_issue,
                            filepath_data=inputs["filepath"],
                            collection=inputs["landsat_collection"],
                            apply_cloud_mask=apply_cloud_mask,
                        )

                # =============================================================================================#
                # Sentinel-2 download
                # =============================================================================================#
                elif satname in ["S2"]:
                    fp_ms = filepaths[1]
                    fp_swir = filepaths[2]
                    fp_mask = filepaths[3]
                    # select bands (10m ms RGB+NIR+s2cloudless, 20m SWIR1, 60m QA band)
                    # Assuming bands_id is a predefined list, and im_bands is a predefined source list
                    bands = {
                        "ms": filter_bands(im_bands, bands_id[:5]),
                        "swir": filter_bands(im_bands, bands_id[5:6]),
                        "mask": filter_bands(im_bands, bands_id[-1:]),
                    }
                    # adjust polygon for both ms and pan bands
                    # RGB and NIR bands 10m resolution and same footprint
                    proj_ms = image_ee.select("B1").projection()
                    # SWIR band 20m resolution and different footprint
                    proj_swir = image_ee.select("B11").projection()
                    proj_mask = image_ee.select("QA60").projection()
                    pbar.set_description_str(
                        desc=f"{satname}: adjusting polygon {i}th image ", refresh=True
                    )
                    ee_region_ms = adjust_polygon(
                        inputs["polygon"],
                        proj_ms,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    ee_region_swir = adjust_polygon(
                        inputs["polygon"],
                        proj_swir,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    ee_region_mask = adjust_polygon(
                        inputs["polygon"],
                        proj_mask,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    # download the ms, swir and QA bands from EE
                    pbar.set_description_str(
                        desc=f"{satname}: Downloading tif for {i}th image ",
                        refresh=True,
                    )
                    fn_ms = download_tif(
                        image_ee,
                        ee_region_ms,
                        bands["ms"],
                        fp_ms,
                        satname,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    fn_swir = download_tif(
                        image_ee,
                        ee_region_swir,
                        bands["swir"],
                        fp_swir,
                        satname,
                        image_id=im_meta["id"],
                        logger=logger,
                    )
                    fn_QA = download_tif(
                        image_ee,
                        ee_region_mask,
                        bands["mask"],
                        fp_mask,
                        satname,
                        image_id=im_meta["id"],
                        logger=logger,
                    )

                    # create filename for the three images (ms, swir and mask)
                    for key in bands.keys():
                        im_fn[key] = (
                            im_date
                            + "_"
                            + satname
                            + "_"
                            + inputs["sitename"]
                            + "_"
                            + key
                            + suffix
                        )
                    # if multiple images taken at the same date add 'dupX' to the name (duplicate)
                    pbar.set_description_str(
                        desc=f"{satname}: remove duplicates for {i}th image ",
                        refresh=True,
                    )
                    im_fn = handle_duplicate_image_names(
                        all_names,
                        bands,
                        im_fn,
                        im_date,
                        satname,
                        inputs["sitename"],
                        suffix,
                    )
                    filename_ms = im_fn["ms"]
                    all_names.append(im_fn["ms"])

                    # resample the 20m swir band to the 10m ms band with bilinear interpolation
                    fn_in = fn_swir
                    fn_target = fn_ms
                    fn_out = os.path.join(fp_swir, im_fn["swir"])
                    pbar.set_description_str(
                        desc=f"{satname}: Transforming {i}th image ", refresh=True
                    )
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=False,
                        resampling_method="bilinear",
                    )

                    # resample 60m QA band to the 10m ms band with nearest-neighbour interpolation
                    fn_in = fn_QA
                    fn_target = fn_ms
                    fn_out = os.path.join(fp_mask, im_fn["mask"])
                    warp_image_to_target(
                        fn_in,
                        fn_out,
                        fn_target,
                        double_res=False,
                        resampling_method="near",
                    )

                    # delete original downloads
                    for _ in [fn_swir, fn_QA]:
                        os.remove(_)
                    # rename the multispectral band file
                    dst = os.path.join(fp_ms, im_fn["ms"])
                    if not os.path.exists(dst):
                        os.rename(fn_ms, dst)
                    if save_jpg:
                        tif_paths = SDS_tools.get_filepath(inputs, satname)
                        SDS_preprocess.save_single_jpg(
                            filename=im_fn["ms"],
                            tif_paths=tif_paths,
                            satname=satname,
                            sitename=inputs["sitename"],
                            cloud_thresh=cloud_threshold,
                            cloud_mask_issue=cloud_mask_issue,
                            filepath_data=inputs["filepath"],
                            collection=inputs["landsat_collection"],
                            apply_cloud_mask=apply_cloud_mask,
                        )
            except Exception as error:
                print(
                    f"\nThe download for satellite {satname} image '{im_meta.get('id','unknown')}' failed due to {type(error).__name__ }"
                )
                logger.error(
                    f"The download for satellite {satname} {im_meta.get('id','unknown')} failed due to \n {error} \n Traceback {traceback.format_exc()}"
                )
                continue
            finally:
                try:
                    # get image dimensions (width and height)
                    if fp_ms:
                        if im_fn.get("ms", "unknown") == "unknown":
                            raise Exception(
                                f"Could not find ms band filename {im_meta.get('id','unknown')}"
                            )
                        image_path = os.path.join(fp_ms, im_fn.get("ms", "unknown"))

                        width, height = SDS_tools.get_image_dimensions(image_path)
                        # write metadata in a text file for easy access
                        filename_txt = (
                            im_fn["ms"].replace("_ms", "").replace(".tif", "")
                        )
                        metadict = {
                            "filename": filename_ms,
                            "epsg": im_epsg,
                            "acc_georef": accuracy_georef,
                            "image_quality": image_quality,
                            "im_width": width,
                            "im_height": height,
                        }
                        # no matter what attempt to write metadata
                        with open(
                            os.path.join(filepaths[0], filename_txt + ".txt"), "w"
                        ) as f:
                            for key in metadict.keys():
                                f.write("%s\t%s\n" % (key, metadict[key]))

                        if im_fn.get("ms", "unknown") != "unknown":
                            logger.info(
                                f"Successfully downloaded image id {im_meta.get('id','unknown')} as {im_fn.get('ms')}"
                            )
                except Exception as e:
                    # print(traceback.format_exc())
                    logger.error(
                        f"Could not save metasdata for {im_meta.get('id','unknown')} that failed.\n{e}"
                    )
                    continue
    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs)
    # save metadata dict
    metadata_json = os.path.join(im_folder, inputs["sitename"] + "_metadata" + ".json")
    SDS_preprocess.write_to_json(metadata_json, metadata)
    print("Satellite images downloaded from GEE and save in %s" % im_folder)
    return metadata


def parse_date_from_filename(filename: str) -> datetime:
    date_str = filename[0:19]
    return pytz.utc.localize(
        datetime(
            int(date_str[:4]),
            int(date_str[5:7]),
            int(date_str[8:10]),
            int(date_str[11:13]),
            int(date_str[14:16]),
            int(date_str[17:19]),
        )
    )


def read_metadata_file(filepath: str) -> Dict[str, Union[str, int, float]]:
    metadata_keys = [
        "filename",
        "epsg",
        "acc_georef",
        "im_quality",
        "im_width",
        "im_height",
    ]

    # Mapping of actual file keys to metadata keys
    key_mapping = {"image_quality": "im_quality"}

    # Initialize the metadata dictionary with default values.
    metadata = {
        "filename": "",
        "epsg": "",
        "acc_georef": -1,
        "im_quality": -1,
        "im_width": -1,
        "im_height": -1,
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split("\t")
            if len(parts) < 2:
                continue  # Skip lines without a tab character

            key = parts[0].strip()
            value = parts[1].strip()

            # Map the actual key in the file to the metadata key
            key = key_mapping.get(key, key)

            # If the mapped key is not in metadata_keys, then skip it.
            if key not in metadata_keys:
                continue

            # Convert value to the appropriate type based on the key
            if key in ["epsg", "im_width", "im_height"]:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        print(
                            f"Error: Unable to convert {key} {value} to a numeric value."
                        )
            elif key in ["acc_georef", "im_quality"]:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep the value as a string if conversion to float fails

            # Update the metadata dictionary with the extracted key-value pair.
            metadata[key] = value

    return metadata



def get_metadata(inputs):
    """
    Gets the metadata from the downloaded images by parsing .txt files located
    in the \meta subfolder.

    KV WRL 2018
    modified by Sharon Fitzpatrick 2023

    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # Construct the directory path containing the images
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The directory {filepath} does not exist.")
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions that were specified in the inputs
    satellite_list = inputs.get("sat_list", ["L5", "L7", "L8", "L9", "S2"])
    for satname in satellite_list:
        sat_path = os.path.join(filepath, satname)
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {
                "filenames": [],
                "dates": [],
                "epsg": [],
                "acc_georef": [],
                "im_quality": [],
                "im_dimensions": [],
            }
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(sat_path, "meta")
            # get the list of filenames and sort it chronologically
            if not os.path.exists(filepath_meta):
                continue
            # Get the list of filenames and sort it chronologically
            filenames_meta = sorted(os.listdir(filepath_meta))
            # loop through the .txt files
            for im_meta in filenames_meta:
                if inputs.get("dates", None) is None:
                    raise ValueError("The 'dates' key is missing from the inputs.")
                    
                start_date = datetime.strptime(inputs['dates'][0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end_date = datetime.strptime(inputs['dates'][1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                input_date = parse_date_from_filename(im_meta)
                # if the image date is outside the specified date range, skip it
                if input_date < start_date or input_date > end_date:
                    continue
                meta_filepath = os.path.join(filepath_meta, im_meta)
                meta_info = read_metadata_file(meta_filepath)

                # Append meta info to the appropriate lists in the metadata dictionary
                metadata[satname]["filenames"].append(meta_info["filename"])
                metadata[satname]["acc_georef"].append(meta_info["acc_georef"])
                metadata[satname]["epsg"].append(meta_info["epsg"])
                metadata[satname]["dates"].append(
                    parse_date_from_filename(meta_info["filename"])
                )
                metadata[satname]["im_quality"].append(meta_info["im_quality"])
                # if the metadata file didn't contain im_height or im_width set this as an empty list
                if meta_info["im_height"] == -1 or meta_info["im_height"] == -1:
                    metadata[satname]["im_dimensions"].append([])
                else:
                    metadata[satname]["im_dimensions"].append(
                        [meta_info["im_height"], meta_info["im_width"]]
                    )

    # save a json file containing the metadata dict
    metadata_json = os.path.join(filepath, f"{inputs['sitename']}_metadata.json")
    SDS_preprocess.write_to_json(metadata_json, metadata)

    return metadata


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################

def remove_existing_imagery(image_dict:dict, metadata:dict,sat_list:list[str])->dict:
    """
    Removes existing imagery from the image dictionary based on the provided metadata.

    Args:
        image_dict (dict): A dictionary containing satellite imagery data.
            Each image_dict contains a list of images for each satellite.
            Each image in the list must contain the keys ['properties']['system:time_start']
            Each image is a dictionary containing the image properties, dates and metadata obtained from the server it was downloaded from.
            Example:
            {
                'L5': [{'type': 'Image',
                        'bands': [{'id': 'B1',
                        'data_type': {'type': 'PixelType',
                        'precision': 'int',
                        'min': 0,
                        'max': 65535},
                        'dimensions': [1830, 1830],
                        'crs': 'EPSG:32618',
                        'crs_transform': [60, 0, 499980, 0, -60, 4500000]},...},  
                {...}, ...],   
            }
        metadata (dict): A dictionary containing metadata information.
            Contains metadata for each satellite in the following format:
            {
                'L5':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L7':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L8':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L9':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'S2':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]}
            }
        sat_list (list[str]): A list of satellite names.

    Returns:
        dict: The updated image dictionary after removing existing imagery.
    """
    for satname in sat_list:
        if satname in metadata and metadata[satname]['dates']:
            avail_date_list = [datetime.fromtimestamp(image['properties']['system:time_start'] / 1000, tz=pytz.utc).replace( microsecond=0) for image in image_dict[satname]]
            if len(avail_date_list) == 0:
                print(f'{satname}:There are {len(avail_date_list)} images available, {len(metadata[satname]["dates"])} images already exist, {len(avail_date_list)} to download')
                continue
            downloaded_dates = metadata[satname]['dates']
            if len(downloaded_dates) == 0:
                print(f'{satname}:There are {len(avail_date_list)} images available, {len(downloaded_dates)} images already exist, {len(avail_date_list)} to download')
                continue
            # get the indices of the images that are not already downloaded
            idx_new = np.where([ not avail_date in downloaded_dates for avail_date in avail_date_list])[0]
            image_dict[satname] = [image_dict[satname][index] for index in idx_new]
            print(f'{satname}:There are {len(avail_date_list)} images available, {len(downloaded_dates)} images already exist, {len(idx_new)} to download')
    return image_dict

def check_images_available(inputs):
    """
    Scan the GEE collections to see how many images are available for each
     satellite mission (L5,L7,L8,L9,S2), collection (C01,C02) and tier (T1,T2).

     KV WRL 2018

     Arguments:
     -----------
     inputs: dict
         inputs dictionary

     Returns:
     -----------
     im_dict_T1: list of dict
         list of images in Tier 1 and Level-1C
     im_dict_T2: list of dict
         list of images in Tier 2 (Landsat only)
    """

    dates = [datetime.strptime(_, "%Y-%m-%d") for _ in inputs["dates"]]
    dates_str = inputs["dates"]
    polygon = inputs["polygon"]

    # check if dates are in chronological order
    if dates[1] <= dates[0]:
        raise Exception("Verify that your dates are in the correct chronological order")

    # check if EE was initialised or not
    try:
        ee.ImageCollection("LANDSAT/LT05/C01/T1_TOA")
    except:
        ee.Initialize()

    print(
        "Number of images available between %s and %s:" % (dates_str[0], dates_str[1]),
        end="\n",
    )

    # get images in Landsat Tier 1 as well as Sentinel Level-1C
    print("- In Landsat Tier 1 & Sentinel-2 Level-1C:")
    col_names_T1 = {
        "L5": "LANDSAT/LT05/%s/T1_TOA" % inputs["landsat_collection"],
        "L7": "LANDSAT/LE07/%s/T1_TOA" % inputs["landsat_collection"],
        "L8": "LANDSAT/LC08/%s/T1_TOA" % inputs["landsat_collection"],
        "L9": "LANDSAT/LC09/C02/T1_TOA",  # only C02 for Landsat 9
        "S2": "COPERNICUS/S2_HARMONIZED",
    }
    im_dict_T1 = dict([])
    sum_img = 0
    # gets the list of images for each satellite mission
    for satname in inputs["sat_list"]:
        im_list = get_image_info(
            col_names_T1[satname],
            satname,
            polygon,
            dates_str,
            S2tile=inputs.get("S2tile", ""),
        )
        # S2 contains many duplicates images so filter collection to only keep images with same UTM Zone projection
        if satname == "S2":
            im_list = filter_S2_collection(im_list)
        sum_img = sum_img + len(im_list)
        print("     %s: %d images" % (satname, len(im_list)))
        im_dict_T1[satname] = im_list

    # CREATES A DICTIONARY OF SATELLITES CONTAINING THE IMAGES IN TIER 1 IN COLLECTION C01 AND C02
    # if using C01 (only goes to the end of 2021), complete with C02 for L7 and L8
    if dates[1] > datetime(2022, 1, 1) and inputs["landsat_collection"] == "C01":
        print("  -> completing Tier 1 with C02 after %s..." % "2022-01-01")
        col_names_C02 = {
            "L7": "LANDSAT/LE07/C02/T1_TOA",
            "L8": "LANDSAT/LC08/C02/T1_TOA",
        }
        dates_C02 = ["2022-01-01", dates_str[1]]
        for satname in inputs["sat_list"]:
            # L7 and L8 have images in both C01 and C02, so complete each list with the other collection
            if satname not in ["L7", "L8"]:
                continue
            im_list = get_image_info(
                col_names_C02[satname], satname, polygon, dates_C02
            )
            sum_img = sum_img + len(im_list)
            print("     %s: %d images" % (satname, len(im_list)))
            im_dict_T1[satname] += im_list

    print("  Total to download: %d images" % sum_img)
    
    
    # if the directory already exists, remove the images that already exist
    filepath = os.path.join(inputs['filepath'],inputs['sitename'])
    if os.path.exists(filepath):
        # get the metadata and satellites that need to be filtered
        sat_list = inputs["sat_list"]
        metadata = get_metadata(inputs)
        # remove any images that already exist from im_dict_T1 because they've already been downloaded
        im_dict_T1 = remove_existing_imagery(im_dict_T1, metadata,sat_list)

    # if only S2 is in sat_list, stop here as no Tier 2 for Sentinel
    if len(inputs["sat_list"]) == 1 and inputs["sat_list"][0] == "S2":
        return im_dict_T1, []

    # CREATES A DICTIONARY OF SATELLITES CONTAINING THE IMAGES IN TIER 2
    # if user also requires Tier 2 images, check the T2 collections as well
    col_names_T2 = {
        "L5": "LANDSAT/LT05/%s/T2_TOA" % inputs["landsat_collection"],
        "L7": "LANDSAT/LE07/%s/T2_TOA" % inputs["landsat_collection"],
        "L8": "LANDSAT/LC08/%s/T2_TOA" % inputs["landsat_collection"],
    }
    print("- In Landsat Tier 2 (not suitable for time-series analysis):", end="\n")
    im_dict_T2 = dict([])
    sum_img = 0
    for satname in inputs["sat_list"]:
        if satname in ["L9", "S2"]:
            continue  # no Tier 2 for Sentinel-2 and Landsat 9
        im_list = get_image_info(col_names_T2[satname], satname, polygon, dates_str)
        sum_img = sum_img + len(im_list)
        print("     %s: %d images" % (satname, len(im_list)))
        im_dict_T2[satname] = im_list

    # also complete with C02 for L7 and L8 after 2022
    if dates[1] > datetime(2022, 1, 1) and inputs["landsat_collection"] == "C01":
        print("  -> completing Tier 2 with C02 after %s..." % "2022-01-01")
        col_names_C02 = {
            "L7": "LANDSAT/LE07/C02/T2_TOA",
            "L8": "LANDSAT/LC08/C02/T2_TOA",
        }
        dates_C02 = ["2022-01-01", dates_str[1]]
        for satname in inputs["sat_list"]:
            # L7 and L8 have images in both C01 and C02, so complete each list with the other collection
            if satname not in ["L7", "L8"]:
                continue  # only L7 and L8
            im_list = get_image_info(
                col_names_C02[satname], satname, polygon, dates_C02
            )
            sum_img = sum_img + len(im_list)
            print("     %s: %d images" % (satname, len(im_list)))
            im_dict_T2[satname] += im_list

    print("  Total Tier 2: %d images" % sum_img)

    return im_dict_T1, im_dict_T2


def get_s2cloudless(image_list: list, inputs: dict):
    """
    Match the list of Sentinel-2 (S2) images with the corresponding s2cloudless images.

    Parameters:
    image_list (List[Dict]): A list of dictionaries, each representing metadata for an S2 image.
    inputs (Dict[str, Union[str, List[str]]]): A dictionary containing:
        - 'dates': List of dates in 'YYYY-mm-dd' format as strings.
        - 'polygon': A list of coordinates defining the polygon of interest.

    Returns:
    List[Union[Dict, List]]: A list where each element is either a dictionary containing metadata of a matched
    s2cloudless image or an empty list if no match is found.
    """
    try:
        # Convert string dates to datetime objects
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in inputs["dates"]]
        polygon = inputs["polygon"]
        collection = "COPERNICUS/S2_CLOUD_PROBABILITY"
        # get s2cloudless collection
        s2cloudless_col = (
            ee.ImageCollection(collection)
            .filterBounds(ee.Geometry.Polygon(polygon))
            .filterDate(dates[0], dates[1])
        )
        cloud_images_list = s2cloudless_col.getInfo().get("features", [])
        # Extract image IDs from the s2cloudless collection
        cloud_indices = [
            image["properties"]["system:index"] for image in cloud_images_list
        ]
        # match with S2 images
        matched_cloud_images = []
        for image_meta in image_list:
            index = image_meta["properties"]["system:index"]

            if index in cloud_indices:
                matched_index = cloud_indices.index(index)
                matched_cloud_images.append(cloud_images_list[matched_index])
            else:  # append an empty list if no match is found
                matched_cloud_images.append([])

        return matched_cloud_images
    except Exception as e:
        raise e


@retry  # Apply the retry decorator to the function
def get_image_info(collection, satname, polygon, dates, **kwargs):
    """
    Reads info about EE images for the specified collection, satellite, and dates

    KV WRL 2022

    Arguments:
    -----------
    collection: str
        name of the collection (e.g. 'LANDSAT/LC08/C02/T1_TOA')
    satname: str
        name of the satellite mission
    polygon: list
        coordinates of the polygon in lat/lon
    dates: list of str
        start and end dates (e.g. '2022-01-01')

    Returns:
    -----------
    im_list: list of ee.Image objects
        list with the info for the images
    """
    # get info about images
    ee_col = ee.ImageCollection(collection)
    # Initialize the collection with filterBounds and filterDate
    col = ee_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(
        dates[0], dates[1]
    )
    # If "S2tile" key is in kwargs and its associated value is truthy (not an empty string, None, etc.),
    # then apply an additional filter to the collection.
    if kwargs.get("S2tile"):
        col = col.filterMetadata("MGRS_TILE", "equals", kwargs["S2tile"])  # 58GGP
        print(f"Only keeping user-defined S2tile: {kwargs['S2tile']}")
    im_list = col.getInfo().get("features")
    # remove very cloudy images (>95% cloud cover)
    im_list = remove_cloudy_images(im_list, satname)
    return im_list


def remove_cloudy_images(im_list, satname, prc_cloud_cover=95):
    """
    Removes from the EE collection very cloudy images (>95% cloud cover)

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection
    satname:
        name of the satellite mission
    prc_cloud_cover: int
        percentage of cloud cover acceptable on the images

    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """

    # remove very cloudy images from the collection (>95% cloud)
    if satname in ["L5", "L7", "L8", "L9"]:
        cloud_property = "CLOUD_COVER"
    elif satname in ["S2"]:
        cloud_property = "CLOUDY_PIXEL_PERCENTAGE"
    cloud_cover = [_["properties"][cloud_property] for _ in im_list]
    if np.any([_ > prc_cloud_cover for _ in cloud_cover]):
        idx_delete = np.where([_ > prc_cloud_cover for _ in cloud_cover])[0]
        im_list_upt = [x for k, x in enumerate(im_list) if k not in idx_delete]
    else:
        im_list_upt = im_list

    return im_list_upt


@retry
def adjust_polygon(polygon, proj, **kwargs):
    """
    Adjust polygon of ROI to fit exactly with the pixels of the underlying tile

    KV WRL 2022

    Arguments:
    -----------
    polygon: list
        polygon containing the lon/lat coordinates to be extracted,
        longitudes in the first column and latitudes in the second column,
        there are 5 pairs of lat/lon with the fifth point equal to the first point:
        ```
        polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
        [151.3, -33.7]]]
        ```
    proj: ee.Proj
        projection of the underlying tile

    Returns:
    -----------
    ee_region: ee
        updated list of images
    """
    # adjust polygon to match image coordinates so that there is no resampling
    polygon_ee = ee.Geometry.Polygon(polygon)
    # convert polygon to image coordinates
    polygon_coords = np.array(
        ee.List(polygon_ee.transform(proj, 1).coordinates().get(0)).getInfo()
    )
    # make it a rectangle
    xmin = np.min(polygon_coords[:, 0])
    ymin = np.min(polygon_coords[:, 1])
    xmax = np.max(polygon_coords[:, 0])
    ymax = np.max(polygon_coords[:, 1])
    # round to the closest pixels
    rect = [np.floor(xmin), np.floor(ymin), np.ceil(xmax), np.ceil(ymax)]
    # convert back to epsg 4326
    ee_region = ee.Geometry.Rectangle(rect, proj, True, False).transform("EPSG:4326")
    return ee_region


# decorator to try the download up to 3 times
@retry
def download_tif(
    image: ee.Image,
    polygon: list,
    bands: list[dict],
    filepath: str,
    satname: str,
    **kwargs,
) -> Union[str, Tuple[str, str]]:
    """
    Downloads a .TIF image from the ee server. The image is downloaded as a
    zip file then moved to the working directory, unzipped and stacked into a
    single .TIF file. Any QA band is saved separately.

    KV WRL 2018

    Arguments:
    -----------
    image: ee.Image
        Image object to be downloaded
    polygon: list
        polygon containing the lon/lat coordinates to be extracted
        longitudes in the first column and latitudes in the second column
    bands: list of dict
        list of bands to be downloaded
    filepath: str
        location where the temporary file should be saved
    satname: str
        name of the satellite missions ['L5','L7','L8','S2']
    Returns:
    -----------
    Downloads an image in a file named data.tif

    """

    # for the old version of ee raise an exception
    if int(ee.__version__[-3:]) <= 201:
        raise Exception(
            "CoastSat2.0 and above is not compatible with earthengine-api version below 0.1.201."
            + "Try downloading a previous CoastSat version (1.x)."
        )
    # for the newer versions of ee
    else:
        # crop and download
        download_id = ee.data.getDownloadId(
            {
                "image": image,
                "region": polygon,
                "bands": bands,
                "filePerBand": True,
                "name": "image",
            }
        )
        response = requests.get(
            ee.data.makeDownloadUrl(download_id),
            timeout=(30, 30),  # 30 seconds to connect, 30 seconds to read
        )
        fp_zip = os.path.join(filepath, "temp.zip")
        with open(fp_zip, "wb") as fd:
            fd.write(response.content)
        # unzip the individual bands
        with zipfile.ZipFile(fp_zip) as local_zipfile:
            for fn in local_zipfile.namelist():
                local_zipfile.extract(fn, filepath)
            fn_all = [os.path.join(filepath, _) for _ in local_zipfile.namelist()]
        os.remove(fp_zip)
        # now process the individual bands:
        # - for Landsat
        if satname in ["L5", "L7", "L8", "L9"]:
            # if there is only one band, it's the panchromatic
            if len(fn_all) == 1:
                # return the filename of the .tif
                return fn_all[0]
            # otherwise there are multiple multispectral bands so we have to merge them into one .tif
            else:
                # select all ms bands except the QA band (which is processed separately)
                fn_tifs = [_ for _ in fn_all if not "QA" in _]
                filename = "ms_bands.tif"
                # build a VRT and merge the bands (works the same with pan band)
                outds = gdal.BuildVRT(
                    os.path.join(filepath, "temp.vrt"), fn_tifs, separate=True
                )
                outds = gdal.Translate(os.path.join(filepath, filename), outds)
                # remove temporary files
                os.remove(os.path.join(filepath, "temp.vrt"))
                for _ in fn_tifs:
                    os.remove(_)
                if os.path.exists(os.path.join(filepath, filename + ".aux.xml")):
                    os.remove(os.path.join(filepath, filename + ".aux.xml"))
                # return file names (ms and QA bands separately)
                fn_image = os.path.join(filepath, filename)
                fn_QA = [_ for _ in fn_all if "QA" in _][0]
                return fn_image, fn_QA
        # - for Sentinel-2
        if satname in ["S2"]:
            # if there is only one band, it's either the SWIR1 or QA60
            if len(fn_all) == 1:
                # return the filename of the .tif
                return fn_all[0]
            # otherwise there are multiple multispectral bands so we have to merge them into one .tif
            else:
                # select all ms bands except the QA band (which is processed separately)
                fn_tifs = fn_all
                filename = "ms_bands.tif"
                # build a VRT and merge the bands (works the same with pan band)
                outds = gdal.BuildVRT(
                    os.path.join(filepath, "temp.vrt"), fn_tifs, separate=True
                )
                outds = gdal.Translate(os.path.join(filepath, filename), outds)
                # remove temporary files
                os.remove(os.path.join(filepath, "temp.vrt"))
                for _ in fn_tifs:
                    os.remove(_)
                if os.path.exists(os.path.join(filepath, filename + ".aux.xml")):
                    os.remove(os.path.join(filepath, filename + ".aux.xml"))
                # return filename of the merge .tif file
                fn_image = os.path.join(filepath, filename)
                return fn_image


def warp_image_to_target(
    fn_in, fn_out, fn_target, double_res=True, resampling_method="bilinear"
):
    """
    Resample an image on a new pixel grid based on a target image using gdal_warp.
    This is used to align the multispectral and panchromatic bands, as well as just downsample certain bands.

    KV WRL 2022

    Arguments:
    -----------
    fn_in: str
        filepath of the input image (points to .tif file)
    fn_out: str
        filepath of the output image (will be created)
    fn_target: str
        filepath of the target image
    double_res: boolean
        this function can be used to downsample images by settings the input and target
        filepaths to the same imageif the input and target images are the same and settings
        double_res = True to downsample by a factor of 2
    resampling_method: str
        method using to resample the image on the new pixel grid. See gdal_warp documentation
        for options (https://gdal.org/programs/gdalwarp.html)

    Returns:
    -----------
    Creates a new .tif file (fn_out)

    """
    # get output extent from target image
    im_target = gdal.Open(fn_target, gdal.GA_ReadOnly)
    georef_target = np.array(im_target.GetGeoTransform())
    xres = georef_target[1]
    yres = georef_target[5]
    if double_res:
        xres = int(georef_target[1] / 2)
        yres = int(georef_target[5] / 2)
    extent_pan = SDS_tools.get_image_bounds(fn_target)
    extent_coords = np.array(extent_pan.exterior.coords)
    xmin = np.min(extent_coords[:, 0])
    ymin = np.min(extent_coords[:, 1])
    xmax = np.max(extent_coords[:, 0])
    ymax = np.max(extent_coords[:, 1])

    # use gdal_warp to resample the input onto the target image pixel grid
    options = gdal.WarpOptions(
        xRes=xres,
        yRes=yres,
        outputBounds=[xmin, ymin, xmax, ymax],
        resampleAlg=resampling_method,
        targetAlignedPixels=False,
    )
    gdal.Warp(fn_out, fn_in, options=options)

    # check that both files have the same georef and size (important!)
    im_target = gdal.Open(fn_target, gdal.GA_ReadOnly)
    im_out = gdal.Open(fn_out, gdal.GA_ReadOnly)
    georef_target = np.array(im_target.GetGeoTransform())
    georef_out = np.array(im_out.GetGeoTransform())
    size_target = np.array([im_target.RasterXSize, im_target.RasterYSize])
    size_out = np.array([im_out.RasterXSize, im_out.RasterYSize])
    if double_res:
        size_target = size_target * 2
    if np.any(np.nonzero(georef_target[[0, 3]] - georef_out[[0, 3]])):
        raise Exception("Georef of pan and ms bands do not match for image %s" % fn_out)
    if np.any(np.nonzero(size_target - size_out)):
        raise Exception("Size of pan and ms bands do not match for image %s" % fn_out)


###################################################################################################
# Sentinel-2 functions
###################################################################################################


def filter_S2_collection(im_list):
    """
    Removes duplicates from the EE collection of Sentinel-2 images (many duplicates)
    Finds the images that were acquired at the same time but have different utm zones.

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection

    Returns:
    -----------
    im_list_flt: list
        filtered list of images
    """

    # get datetimes
    timestamps = [
        datetime.fromtimestamp(_["properties"]["system:time_start"] / 1000, tz=pytz.utc)
        for _ in im_list
    ]
    # get utm zone projections
    utm_zones = np.array([int(_["bands"][0]["crs"][5:]) for _ in im_list])
    if len(np.unique(utm_zones)) == 1:
        return im_list
    else:
        idx_max = np.argmax([np.sum(utm_zones == _) for _ in np.unique(utm_zones)])
        utm_zone_selected = np.unique(utm_zones)[idx_max]
        # find the images that were acquired at the same time but have different utm zones
        idx_all = np.arange(0, len(im_list), 1)
        idx_covered = np.ones(len(im_list)).astype(bool)
        idx_delete = []
        i = 0
        while 1:
            same_time = (
                np.abs([(timestamps[i] - _).total_seconds() for _ in timestamps])
                < 60 * 60 * 24
            )
            idx_same_time = np.where(same_time)[0]
            same_utm = utm_zones == utm_zone_selected
            # get indices that have the same time (less than 24h apart) but not the same utm zone
            idx_temp = np.where(
                [same_time[j] == True and same_utm[j] == False for j in idx_all]
            )[0]
            idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time]]
            # if more than 2 images with same date and same utm, drop the last ones
            if len(idx_keep) > 2:
                idx_temp = np.append(idx_temp, idx_keep[-(len(idx_keep) - 2) :])
            for j in idx_temp:
                idx_delete.append(j)
            idx_covered[idx_same_time] = False
            if np.any(idx_covered):
                i = np.where(idx_covered)[0][0]
            else:
                break
        # update the collection by deleting all those images that have same timestamp
        # and different utm projection
        im_list_flt = [x for k, x in enumerate(im_list) if k not in idx_delete]

    return im_list_flt


def merge_overlapping_images(metadata, inputs):
    """
    Merge simultaneous overlapping images that cover the area of interest.
    When the area of interest is located at the boundary between 2 images, there
    will be overlap between the 2 images and both will be downloaded from Google
    Earth Engine. This function merges the 2 images, so that the area of interest
    is covered by only 1 image.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata_updated: dict
        updated metadata

    """

    # only for Sentinel-2 at this stage (not sure if this is needed for Landsat images)
    sat = "S2"
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    filenames = metadata[sat]["filenames"]
    total_images = len(filenames)

    # nested function
    def duplicates_dict(lst):
        "return duplicates and indices"

        def duplicates(lst, item):
            return [i for i, x in enumerate(lst) if x == item]

        return dict((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1)

    # first pass on images that have the exact same timestamp
    duplicates = duplicates_dict([_.split("_")[0] for _ in filenames])
    # {"S2-2029-2020": [0,1,2,3]}
    # {"duplicate_filename": [indices of duplicated files]"}

    total_removed_step1 = 0
    if len(duplicates) > 0:
        # loop through each pair of duplicates and merge them
        for key in duplicates.keys():
            idx_dup = duplicates[key]
            # get full filenames (3 images and .txtt) for each index and bounding polygons
            fn_im, polygons, im_epsg = [], [], []
            for index in range(len(idx_dup)):
                # image names
                fn_im.append(
                    [
                        os.path.join(filepath, "S2", "10m", filenames[idx_dup[index]]),
                        os.path.join(
                            filepath,
                            "S2",
                            "20m",
                            filenames[idx_dup[index]].replace("10m", "20m"),
                        ),
                        os.path.join(
                            filepath,
                            "S2",
                            "60m",
                            filenames[idx_dup[index]].replace("10m", "60m"),
                        ),
                        os.path.join(
                            filepath,
                            "S2",
                            "meta",
                            filenames[idx_dup[index]]
                            .replace("_10m", "")
                            .replace(".tif", ".txt"),
                        ),
                    ]
                )
                try:
                    # bounding polygons
                    polygons.append(SDS_tools.get_image_bounds(fn_im[index][0]))
                    im_epsg.append(metadata[sat]["epsg"][idx_dup[index]])
                except AttributeError:
                    print(
                        "\n Error getting the TIF. Skipping this iteration of the loop"
                    )
                    continue
                except FileNotFoundError:
                    print(f"\n The file {fn_im[index][0]} did not exist")
                    continue

            # check if epsg are the same, print a warning message
            if len(np.unique(im_epsg)) > 1:
                print(
                    "WARNING: there was an error as two S2 images do not have the same epsg,"
                    + " please open an issue on Github at https://github.com/kvos/CoastSat/issues"
                    + " and include your script so I can find out what happened."
                )
            # find which images contain other images
            contain_bools_list = []
            for i, poly1 in enumerate(polygons):
                contain_bools = []
                for k, poly2 in enumerate(polygons):
                    if k == i:
                        contain_bools.append(True)
                        # print('%d*: '%k+str(poly1.contains(poly2)))
                    else:
                        # print('%d: '%k+str(poly1.contains(poly2)))
                        contain_bools.append(poly1.contains(poly2))
                contain_bools_list.append(contain_bools)
            # look if one image contains all the others
            contain_all = [np.all(_) for _ in contain_bools_list]
            # if one image contains all the others, keep that one and delete the rest
            if np.any(contain_all):
                idx_keep = np.where(contain_all)[0][0]
                for i in [_ for _ in range(len(idx_dup)) if not _ == idx_keep]:
                    # print('removed %s'%(fn_im[i][-1]))
                    # remove the 3 .tif files + the .txt file
                    for k in range(4):
                        os.chmod(fn_im[i][k], 0o777)
                        os.remove(fn_im[i][k])
                    total_removed_step1 += 1
        # load metadata again and update filenames
        metadata = get_metadata(inputs)
        filenames = metadata[sat]["filenames"]

    # find the pairs of images that are within 5 minutes of each other and merge them
    time_delta = 5 * 60  # 5 minutes in seconds
    dates = metadata[sat]["dates"].copy()
    pairs = []
    for i, date in enumerate(metadata[sat]["dates"]):
        # dummy value so it does not match it again
        dates[i] = pytz.utc.localize(datetime(1, 1, 1) + timedelta(days=i + 1))
        # calculate time difference
        time_diff = np.array([np.abs((date - _).total_seconds()) for _ in dates])
        # find the matching times and add to pairs list
        boolvec = time_diff <= time_delta
        if np.sum(boolvec) == 0:
            continue
        else:
            idx_dup = np.where(boolvec)[0][0]
            pairs.append([i, idx_dup])
    total_merged_step2 = len(pairs)
    # because they could be triplicates in S2 images, adjust the pairs for consecutive merges
    for i in range(1, len(pairs)):
        if pairs[i - 1][1] == pairs[i][0]:
            pairs[i][0] = pairs[i - 1][0]

    # check also for quadruplicates and remove them
    pair_first = [_[0] for _ in pairs]
    idx_remove_pair = []
    for idx in np.unique(pair_first):
        # calculate the number of duplicates
        n_duplicates = sum(pair_first == idx)
        # if more than 3 duplicates, delete the other images so that a max of 3 duplicates are handled
        if n_duplicates > 2:
            for i in range(2, n_duplicates):
                # remove the last image: 3 .tif files + the .txt file
                idx_last = [pairs[_] for _ in np.where(pair_first == idx)[0]][i][-1]
                fn_im = [
                    os.path.join(filepath, "S2", "10m", filenames[idx_last]),
                    os.path.join(
                        filepath, "S2", "20m", filenames[idx_last].replace("10m", "20m")
                    ),
                    os.path.join(
                        filepath, "S2", "60m", filenames[idx_last].replace("10m", "60m")
                    ),
                    os.path.join(
                        filepath,
                        "S2",
                        "meta",
                        filenames[idx_last].replace("_10m", "").replace(".tif", ".txt"),
                    ),
                ]
                for k in range(4):
                    os.chmod(fn_im[k], 0o777)
                    os.remove(fn_im[k])
                # store the index of the pair to remove it outside the loop
                idx_remove_pair.append(np.where(pair_first == idx)[0][i])
    # remove quadruplicates from list of pairs
    pairs = [i for j, i in enumerate(pairs) if j not in idx_remove_pair]

    # for each pair of image, first check if one image completely contains the other
    # in that case keep the larger image. Otherwise merge the two images.
    for i, pair in enumerate(pairs):
        # get filenames of all the files corresponding to the each image in the pair
        fn_im = []
        for index in range(len(pair)):
            fn_im.append(
                [
                    os.path.join(filepath, "S2", "10m", filenames[pair[index]]),
                    os.path.join(
                        filepath,
                        "S2",
                        "20m",
                        filenames[pair[index]].replace("10m", "20m"),
                    ),
                    os.path.join(
                        filepath,
                        "S2",
                        "60m",
                        filenames[pair[index]].replace("10m", "60m"),
                    ),
                    os.path.join(
                        filepath,
                        "S2",
                        "meta",
                        filenames[pair[index]]
                        .replace("_10m", "")
                        .replace(".tif", ".txt"),
                    ),
                ]
            )
        # get polygon for first image
        try:
            polygon0 = SDS_tools.get_image_bounds(fn_im[0][0])
            im_epsg0 = metadata[sat]["epsg"][pair[0]]
        except AttributeError:
            print("\n Error getting the TIF. Skipping this iteration of the loop")
            continue
        except FileNotFoundError:
            print(f"\n The file {fn_im[index][0]} did not exist")
            continue
        # get polygon for second image
        try:
            polygon1 = SDS_tools.get_image_bounds(fn_im[1][0])
            im_epsg1 = metadata[sat]["epsg"][pair[1]]
        except AttributeError:
            print("\n Error getting the TIF. Skipping this iteration of the loop")
            continue
        except FileNotFoundError:
            print(f"\n The file {fn_im[index][0]} did not exist")
            continue
        # check if epsg are the same
        if not im_epsg0 == im_epsg1:
            print(
                "WARNING: there was an error as two S2 images do not have the same epsg,"
                + " please open an issue on Github at https://github.com/kvos/CoastSat/issues"
                + " and include your script so we can find out what happened."
            )
            break
        # check if one image contains the other one
        if polygon0.contains(polygon1):
            # if polygon0 contains polygon1, remove files for polygon1
            for k in range(4):  # remove the 3 .tif files + the .txt file
                os.chmod(fn_im[1][k], 0o777)
                os.remove(fn_im[1][k])
            # print('removed 1')
            continue
        elif polygon1.contains(polygon0):
            # if polygon1 contains polygon0, remove image0
            for k in range(4):  # remove the 3 .tif files + the .txt file
                os.chmod(fn_im[0][k], 0o777)
                os.remove(fn_im[0][k])
            # print('removed 0')
            # adjust the order in case of triplicates
            if i + 1 < len(pairs):
                if pairs[i + 1][0] == pair[0]:
                    pairs[i + 1][0] = pairs[i][1]
            continue
        # otherwise merge the two images after masking the nodata values
        else:
            for index in range(len(pair)):
                # read image
                (
                    im_ms,
                    georef,
                    cloud_mask,
                    im_extra,
                    im_QA,
                    im_nodata,
                ) = SDS_preprocess.preprocess_single(fn_im[index], sat, False, "C01")
                # in Sentinel2 images close to the edge of the image there are some artefacts,
                # that are squares with constant pixel intensities. They need to be masked in the
                # raster (GEOTIFF). It can be done using the image standard deviation, which
                # indicates values close to 0 for the artefacts.
                if len(im_ms) > 0:
                    # calculate image std for the first 10m band
                    im_std = SDS_tools.image_std(im_ms[:, :, 0], 1)
                    # convert to binary
                    im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                    # dilate to fill the edges (which have high std)
                    mask10 = morphology.dilation(im_binary, morphology.square(3))
                    # mask the 10m .tif file (add no_data where mask is True)
                    SDS_tools.mask_raster(fn_im[index][0], mask10)
                    # now calculate the mask for the 20m band (SWIR1)
                    # for the older version of the ee api calculate the image std again
                    if int(ee.__version__[-3:]) <= 201:
                        # calculate std to create another mask for the 20m band (SWIR1)
                        im_std = SDS_tools.image_std(im_extra, 1)
                        im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                        mask20 = morphology.dilation(im_binary, morphology.square(3))
                    # for the newer versions just resample the mask for the 10m bands
                    else:
                        # create mask for the 20m band (SWIR1) by resampling the 10m one
                        mask20 = ndimage.zoom(mask10, zoom=1 / 2, order=0)
                        mask20 = transform.resize(
                            mask20,
                            im_extra.shape,
                            mode="constant",
                            order=0,
                            preserve_range=True,
                        )
                        mask20 = mask20.astype(bool)
                    # mask the 20m .tif file (im_extra)
                    SDS_tools.mask_raster(fn_im[index][1], mask20)
                    # create a mask for the 60m QA band by resampling the 20m one
                    mask60 = ndimage.zoom(mask20, zoom=1 / 3, order=0)
                    mask60 = transform.resize(
                        mask60,
                        im_QA.shape,
                        mode="constant",
                        order=0,
                        preserve_range=True,
                    )
                    mask60 = mask60.astype(bool)
                    # mask the 60m .tif file (im_QA)
                    SDS_tools.mask_raster(fn_im[index][2], mask60)
                    # make a figure for quality control/debugging
                    # im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
                    # fig,ax= plt.subplots(2,3,tight_layout=True)
                    # ax[0,0].imshow(im_RGB)
                    # ax[0,0].set_title('RGB original')
                    # ax[1,0].imshow(mask10)
                    # ax[1,0].set_title('Mask 10m')
                    # ax[0,1].imshow(mask20)
                    # ax[0,1].set_title('Mask 20m')
                    # ax[1,1].imshow(mask60)
                    # ax[1,1].set_title('Mask 60 m')
                    # ax[0,2].imshow(im_QA)
                    # ax[0,2].set_title('Im QA')
                    # ax[1,2].imshow(im_nodata)
                    # ax[1,2].set_title('Im nodata')
                else:
                    continue

            # once all the pairs of .tif files have been masked with no_data, merge the using gdal_merge
            fn_merged = os.path.join(filepath, "merged.tif")
            for k in range(3):
                # merge masked bands
                gdal_merge.main(
                    ["", "-o", fn_merged, "-n", "0", fn_im[0][k], fn_im[1][k]]
                )
                # remove old files
                os.chmod(fn_im[0][k], 0o777)
                os.remove(fn_im[0][k])
                os.chmod(fn_im[1][k], 0o777)
                os.remove(fn_im[1][k])
                # rename new file
                fn_new = fn_im[0][k].split(".")[0] + "_merged.tif"
                os.chmod(fn_merged, 0o777)
                os.rename(fn_merged, fn_new)

            # open both metadata files
            metadict0 = dict([])
            with open(fn_im[0][3], "r") as f:
                metadict0["filename"] = f.readline().split("\t")[1].replace("\n", "")
                metadict0["acc_georef"] = float(
                    f.readline().split("\t")[1].replace("\n", "")
                )
                metadict0["epsg"] = int(f.readline().split("\t")[1].replace("\n", ""))
            metadict1 = dict([])
            with open(fn_im[1][3], "r") as f:
                metadict1["filename"] = f.readline().split("\t")[1].replace("\n", "")
                metadict1["acc_georef"] = float(
                    f.readline().split("\t")[1].replace("\n", "")
                )
                metadict1["epsg"] = int(f.readline().split("\t")[1].replace("\n", ""))
            # check if both images have the same georef accuracy
            if np.any(
                np.array([metadict0["acc_georef"], metadict1["acc_georef"]]) == -1
            ):
                metadict0["georef"] = -1
            # add new name
            metadict0["filename"] = metadict0["filename"].split(".")[0] + "_merged.tif"
            # remove the old metadata.txt files
            os.chmod(fn_im[0][3], 0o777)
            os.remove(fn_im[0][3])
            os.chmod(fn_im[1][3], 0o777)
            os.remove(fn_im[1][3])
            # rewrite the .txt file with a new metadata file
            fn_new = fn_im[0][3].split(".")[0] + "_merged.txt"
            with open(fn_new, "w") as f:
                for key in metadict0.keys():
                    f.write("%s\t%s\n" % (key, metadict0[key]))

            # update filenames list (in case there are triplicates)
            filenames[pair[0]] = metadict0["filename"]

    print(
        "%d out of %d Sentinel-2 images were merged (overlapping or duplicate)"
        % (total_removed_step1 + total_merged_step2, total_images)
    )

    # update the metadata dict
    metadata_updated = get_metadata(inputs)

    return metadata_updated
