"""Tests for the Sentinel-1 (SAR) workflow: download, reading, detection and outputs."""

from datetime import datetime
import inspect
import logging
from pathlib import Path
import re
import sys

import matplotlib
import numpy as np
import pytest
import pytz

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal, osr

from coastsat import (
    SDS_download,
    SDS_plotting,
    SDS_preprocess,
    SDS_sar_model,
    SDS_shoreline,
    SDS_tools,
)
from coastsat.SDS_download import (
    get_band_order_from_meta,
    get_downloaded_sar_polarizations,
    get_incomplete_sar_dates,
    get_metadata,
    process_sentinel1_image,
    read_metadata_file,
    remove_existing_imagery,
    write_metadata_file,
)
from coastsat.SDS_preprocess import build_sar_preview, preprocess_image, save_sar_image
from coastsat.SDS_sar_model import SarModelUnavailable
from coastsat.SDS_shoreline import (
    compute_sar_water_fraction,
    describe_sar_otsu_image,
    get_pixel_size_for_satellite,
    get_sar_band_index,
    get_sar_detection_polarization,
    get_sar_otsu_image,
    load_sar_segmenter,
    otsu_threshold,
    save_sar_probability,
    segment_sar_water,
    warn_unknown_sar_detection_polarization,
)
from coastsat.SDS_tools import (
    get_polarization_from_filename,
    read_sar_image,
    read_sar_valid_mask,
    swap_polarization_in_filename,
)


# =================================================================================== #
# Constants and helpers shared by more than one section
# =================================================================================== #


class FakeFilter:
    """Stand-in for ee.Filter: the real one needs an initialised algorithm registry."""

    @staticmethod
    def eq(prop, value):
        return ("eq", prop, value)

    @staticmethod
    def listContains(prop, value):
        return ("listContains", prop, value)

    @staticmethod
    def And(*filters):
        return ("And", filters)

    @staticmethod
    def Or(*filters):
        return ("Or", filters)


@pytest.fixture
def fake_ee_filter(monkeypatch):
    monkeypatch.setattr(SDS_download.ee, "Filter", FakeFilter)


DATE = "2023-12-03-19-15-49"
SITENAME = "site"
IMAGE_ID = "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20231203T191549"
GEOTRANSFORM = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0]
SIZE = 8


def s1_path(polar, suffix=""):
    name = f"{DATE}_S1_{SITENAME}_{polar}{suffix}.tif"
    return str(Path("data") / SITENAME / "S1" / polar / name)


class RecordingLogger:
    def __init__(self):
        self.info_messages = []
        self.warnings = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))


DUAL_POL = [s1_path("VV"), s1_path("VH")]


# =================================================================================== #
# Tests for the Sentinel-1 polarization list and the Earth Engine polarization filter
# =================================================================================== #


def contains(polarization):
    return ("listContains", "transmitterReceiverPolarisation", polarization)


# ------------------------------------------------------------------ polarization list


def test_defaults_to_vv_and_vh():
    # both are downloaded by default, so the [VV, VH, VV-VH] composite is available
    assert SDS_download.get_sentinel1_polarizations({}) == ["VV", "VH"]


def test_default_comes_from_one_constant():
    assert list(SDS_tools.DEFAULT_SAR_POLARIZATIONS) == ["VV", "VH"]
    assert (
        SDS_download.DEFAULT_SENTINEL_1_PROPERTIES["transmitterReceiverPolarisation"]
        == list(SDS_tools.DEFAULT_SAR_POLARIZATIONS)
    )


@pytest.mark.parametrize(
    "properties,expected",
    [
        # the channel order of im_ms must not depend on how the user listed them
        ({"transmitterReceiverPolarisation": ["VH", "VV"]}, ["VV", "VH"]),
        ({"transmitterReceiverPolarisation": [" vh ", "vv"]}, ["VV", "VH"]),
        ({"transmitterReceiverPolarisation": ["VH", "vh", "VH"]}, ["VH"]),
        ({"transmitterReceiverPolarisation": "VV"}, ["VV"]),
        ({"transmitterReceiverPolarisation": ["VH"]}, ["VH"]),
        # merging is key by key, so setting only instrumentMode keeps the default VV+VH
        ({"instrumentMode": "EW"}, ["VV", "VH"]),
    ],
    ids=[
        "request-order",
        "case-and-whitespace",
        "duplicates",
        "bare-string",
        "single-polarization",
        "other-properties-only",
    ],
)
def test_get_sentinel1_polarizations(properties, expected):
    inputs = {"sentinel_1_properties": properties}

    assert SDS_download.get_sentinel1_polarizations(inputs) == expected


@pytest.mark.parametrize(
    "polarizations,match",
    [(["VV", "XX"], "XX"), ([], "at least one polarization")],
    ids=["invalid", "empty"],
)
def test_rejects_unusable_polarizations(polarizations, match):
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": polarizations}}

    with pytest.raises(ValueError, match=match):
        SDS_download.get_sentinel1_polarizations(inputs)


# ------------------------------------------------------------------ ee filter


def test_multiple_polarizations_are_anded(fake_ee_filter):
    # AND, not OR: every scene must carry all of them so the full set can be downloaded
    assert SDS_download.create_polarization_filter(["VV", "VH"]) == (
        "And",
        (contains("VV"), contains("VH")),
    )


def test_single_polarization_is_not_wrapped(fake_ee_filter):
    assert SDS_download.create_polarization_filter(["VH"]) == contains("VH")


def test_polarizations_accepts_a_bare_string_polarization(fake_ee_filter):
    assert SDS_download.create_polarization_filter("VH") == contains("VH")


def test_or_mode_still_available(fake_ee_filter):
    assert SDS_download.create_polarization_filter(["VV", "VH"], mode="or") == (
        "Or",
        (contains("VV"), contains("VH")),
    )


@pytest.mark.parametrize(
    "polarizations,kwargs,match",
    [
        (["XX"], {}, "XX"),
        ([], {}, "(?i)at least one polarization"),
        (["VV", "VH"], {"mode": "xor"}, "mode"),
    ],
    ids=["invalid-polarization", "empty-list", "unknown-mode"],
)
def test_filter_rejects_unusable_arguments(fake_ee_filter, polarizations, kwargs, match):
    with pytest.raises(ValueError, match=match):
        SDS_download.create_polarization_filter(polarizations, **kwargs)


# =================================================================================== #
# Tests that Sentinel-1 metadata properties become Earth Engine filters
#
# instrumentMode and orbitProperties_pass used to be read into sentinel_1_properties and
# then silently ignored - setting them had no effect on the query. They now become
# ee.Filter.eq filters on the S1 collection, and get_tier2_images no longer raises a
# KeyError when S1 is in sat_list (S1 has no Tier 2 collection, like S2 and L9).
# =================================================================================== #


# --------------------------------------------------- create_sentinel1_metadata_filters


def test_no_properties_means_no_filters(fake_ee_filter):
    assert SDS_download.create_sentinel1_metadata_filters({}) == []


@pytest.mark.parametrize(
    "prop,value",
    [("instrumentMode", "IW"), ("orbitProperties_pass", "DESCENDING")],
)
def test_a_property_becomes_an_eq_filter(fake_ee_filter, prop, value):
    filters = SDS_download.create_sentinel1_metadata_filters({prop: value})

    assert filters == [("eq", prop, value)]


def test_values_are_normalised(fake_ee_filter):
    # GEE property values are upper case; a lower-case request must still match
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"instrumentMode": " iw ", "orbitProperties_pass": "ascending"}
    )

    assert filters == [
        ("eq", "instrumentMode", "IW"),
        ("eq", "orbitProperties_pass", "ASCENDING"),
    ]


def test_polarizations_are_not_this_functions_business(fake_ee_filter):
    # they have their own AND filter, see create_polarization_filter
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"transmitterReceiverPolarisation": ["VV", "VH"]}
    )

    assert filters == []


@pytest.mark.parametrize(
    "prop,value",
    [("instrumentMode", "XX"), ("orbitProperties_pass", "SIDEWAYS")],
)
def test_rejects_an_invalid_property_value(fake_ee_filter, prop, value):
    with pytest.raises(ValueError, match=prop):
        SDS_download.create_sentinel1_metadata_filters({prop: value})


def test_the_default_properties_are_valid(fake_ee_filter):
    # the defaults must never be the thing that raises
    filters = SDS_download.create_sentinel1_metadata_filters(
        SDS_download.DEFAULT_SENTINEL_1_PROPERTIES
    )

    assert filters == [("eq", "instrumentMode", "IW")]


# ----------------------------------------------------------- wiring into the S1 query


def tier1_inputs(sentinel_1_properties=None):
    inputs = {
        "sat_list": ["S1"],
        "sitename": "site",
    }
    if sentinel_1_properties is not None:
        inputs["sentinel_1_properties"] = sentinel_1_properties
    return inputs


@pytest.fixture
def recorded_queries(monkeypatch):
    calls = []

    def fake_get_image_info(collection, satname, polygon, dates, **kwargs):
        calls.append({"collection": collection, "satname": satname, **kwargs})
        return []

    monkeypatch.setattr(SDS_download, "get_image_info", fake_get_image_info)
    return calls


def test_tier1_passes_the_merged_properties_to_the_query(recorded_queries):
    SDS_download.get_tier1_images(
        tier1_inputs({"orbitProperties_pass": "DESCENDING"}),
        polygon=[[[151.3, -33.7]]],
        dates=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    (call,) = recorded_queries
    # user values are merged over the defaults key by key
    assert call["sentinel_1_properties"]["orbitProperties_pass"] == "DESCENDING"
    assert call["sentinel_1_properties"]["instrumentMode"] == "IW"


def test_tier1_defaults_still_filter_the_instrument_mode(recorded_queries):
    SDS_download.get_tier1_images(
        tier1_inputs(),
        polygon=[[[151.3, -33.7]]],
        dates=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    (call,) = recorded_queries
    assert call["sentinel_1_properties"]["instrumentMode"] == "IW"
    assert "orbitProperties_pass" not in call["sentinel_1_properties"]


# -------------------------------------------------------------------- get_tier2_images


def test_tier2_skips_s1_instead_of_raising(recorded_queries):
    # col_names_T2 has no "S1" key; S1 + tier2 used to raise KeyError
    im_dict = SDS_download.get_tier2_images(
        {"sat_list": ["L5", "S1"], "sitename": "site"},
        polygon=[[[151.3, -33.7]]],
        dates_str=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    assert "S1" not in im_dict
    assert list(im_dict.keys()) == ["L5"]


def test_tier2_with_only_s1_returns_an_empty_dict(recorded_queries):
    im_dict = SDS_download.get_tier2_images(
        {"sat_list": ["S1"], "sitename": "site"},
        polygon=[[[151.3, -33.7]]],
        dates_str=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    assert im_dict == {}
    assert recorded_queries == []


# =================================================================================== #
# Tests for locating and swapping the polarization token in a SAR filename
#
# This is what pairs the VV and VH files of a scene, which live in separate folders
# but otherwise share a name.
# =================================================================================== #


@pytest.mark.parametrize(
    "filename,expected",
    [
        (f"{DATE}_S1_site_VV.tif", "VV"),
        (f"{DATE}_S1_site_VH.tif", "VH"),
        (f"{DATE}_S1_site_HH.tif", "HH"),
        (f"{DATE}_S1_site_HV.tif", "HV"),
        # get_file_name() appends _dupN AFTER the polarization, so it is not last
        (f"{DATE}_S1_site_VV_dup0.tif", "VV"),
        (f"{DATE}_S1_site_VH_dup12.tif", "VH"),
        # optical products carry no polarization
        (f"{DATE}_S2_site_ms.tif", None),
        (f"{DATE}_L8_site_pan.tif", None),
    ],
)
def test_get_polarization_from_filename(filename, expected):
    assert get_polarization_from_filename(filename) == expected


def test_reads_the_token_from_a_full_path():
    path = str(Path("data") / "site" / "S1" / "VH" / f"{DATE}_S1_site_VH.tif")
    assert get_polarization_from_filename(path) == "VH"


def test_does_not_match_polarization_inside_sitename():
    # a site called 'myVHsite' must not be mistaken for a VH product
    filename = f"{DATE}_S1_myVHsite_VV.tif"
    assert get_polarization_from_filename(filename) == "VV"
    assert swap_polarization_in_filename(filename, "VH") == f"{DATE}_S1_myVHsite_VH.tif"


@pytest.mark.parametrize(
    "filename,expected",
    [
        (f"{DATE}_S1_site_VV.tif", f"{DATE}_S1_site_VH.tif"),
        (f"{DATE}_S1_site_VV_dup0.tif", f"{DATE}_S1_site_VH_dup0.tif"),
        (f"{DATE}_S1_site_VH.tif", f"{DATE}_S1_site_VH.tif"),
        (f"{DATE}_S2_site_ms.tif", f"{DATE}_S2_site_ms.tif"),
    ],
    ids=["replaces-the-token", "keeps-the-dup-suffix", "same-polarization", "no-token"],
)
def test_swap_polarization_in_filename(filename, expected):
    assert swap_polarization_in_filename(filename, "VH") == expected


# =================================================================================== #
# Tests for the Sentinel-1 folder layout: one folder per polarization
#
# Covers creating the folders, listing them back, and pairing up the files of a scene -
# including sessions downloaded before multi-polarization support, which have VH only.
# =================================================================================== #


def layout_write_tif(path, value=1.0, size=8, geotransform=None):
    """Write a small single-band Float64 .tif, like an Earth Engine S1 download."""
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geotransform or [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return path


def s1_filename(polar, suffix=""):
    return f"{DATE}_S1_{SITENAME}_{polar}{suffix}.tif"


def layout_make_session(root, polarizations, suffix=""):
    """Create <root>/<site>/S1/<POLAR>/ with one .tif each, and return the inputs dict."""
    for index, polar in enumerate(polarizations):
        layout_write_tif(root / SITENAME / "S1" / polar / s1_filename(polar, suffix), value=index)
    return {"filepath": str(root), "sitename": SITENAME}


# ------------------------------------------------------------------ create_folder_structure


def test_creates_one_folder_per_polarization(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VV", "VH"])

    # the composite is built in memory, so it gets no folder of its own
    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]
    assert all(Path(fp).is_dir() for fp in filepaths)


def test_folders_are_created_in_canonical_order(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VH", "VV"])

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]


def test_defaults_to_vv_and_vh_when_no_polarizations_given(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1")

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]


def test_layout_accepts_a_bare_string_polarization(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", "VV")

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV"]


def test_optical_layout_is_unchanged(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S2")

    assert [Path(fp).name for fp in filepaths] == ["meta", "ms", "swir", "mask"]


# ------------------------------------------------------------------ get_filepath


@pytest.mark.parametrize(
    "created,expected",
    [
        (["VV", "VH"], ["VV", "VH"]),
        # created VH first, but VV must still come first
        (["VH", "VV"], ["VV", "VH"]),
        # a legacy VH-only session: a folder that does not exist is not returned
        (["VH"], ["VH"]),
    ],
    ids=["both", "canonical-order", "legacy-vh-only"],
)
def test_get_filepath_returns_existing_polarization_folders(tmp_path, created, expected):
    inputs = layout_make_session(tmp_path, created)

    filepath = SDS_tools.get_filepath(inputs, "S1")

    assert [Path(fp).name for fp in filepath] == expected


# ------------------------------------------------------------------ get_filenames


def test_get_filenames_pairs_vv_and_vh(tmp_path):
    inputs = layout_make_session(tmp_path, ["VV", "VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VV"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VV"), s1_filename("VH")]
    assert all(Path(p).exists() for p in fn)


def test_get_filenames_pairs_from_either_polarization(tmp_path):
    # the metadata records the first polarization, but starting from VH must work too
    inputs = layout_make_session(tmp_path, ["VV", "VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VV"), s1_filename("VH")]


def test_get_filenames_pairs_duplicate_named_files(tmp_path):
    inputs = layout_make_session(tmp_path, ["VV", "VH"], suffix="_dup0")
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VV", "_dup0"), filepath, "S1")

    assert [Path(p).name for p in fn] == [
        s1_filename("VV", "_dup0"),
        s1_filename("VH", "_dup0"),
    ]


def test_get_filenames_skips_a_polarization_missing_from_disk(tmp_path):
    # a scene downloaded before multi-polarization support, sitting in a VV+VH session
    inputs = layout_make_session(tmp_path, ["VV", "VH"])
    (tmp_path / SITENAME / "S1" / "VV" / s1_filename("VV")).unlink()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]


def test_get_filenames_ignores_an_empty_polarization_folder(tmp_path):
    inputs = layout_make_session(tmp_path, ["VH"])
    (tmp_path / SITENAME / "S1" / "HH").mkdir()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]


def test_get_filenames_raises_when_nothing_exists(tmp_path):
    inputs = layout_make_session(tmp_path, ["VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    with pytest.raises(FileNotFoundError):
        SDS_tools.get_filenames(f"1999-01-01-00-00-00_S1_{SITENAME}_VH.tif", filepath, "S1")


def test_get_filenames_accepts_a_string_filepath(tmp_path):
    # get_filepath is monkeypatched to a plain string in test_extract_shorelines_all_zero_ms
    layout_make_session(tmp_path, ["VH"])
    folder = str(tmp_path / SITENAME / "S1" / "VH")

    fn = SDS_tools.get_filenames(s1_filename("VH"), folder, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]


# =================================================================================== #
# Tests that a Sentinel-1 scene downloads every requested polarization
#
# Earth Engine is never contacted: download_S1_image is replaced with a stub that
# writes a small .tif, so this exercises the folder layout, naming, metadata and the
# rollback that stops a half-downloaded scene from being left on disk.
# =================================================================================== #


def download_band(polarization):
    return {
        "id": polarization,
        "crs": "EPSG:32756",
        "crs_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0],
        "dimensions": [25000, 16000],
    }


def download_im_meta(polarizations=("VV", "VH")):
    return {
        "id": IMAGE_ID,
        "bands": [download_band(polar) for polar in polarizations],
        "properties": {
            "orbitProperties_pass": "DESCENDING",
            "transmitterReceiverPolarisation": list(polarizations),
            "resolution": "H",
            "resolution_meters": 10,
            "instrumentMode": "IW",
        },
    }


def download_inputs_for(polarizations=("VV", "VH")):
    return {
        "sitename": SITENAME,
        "polygon": [[[151.3, -33.7], [151.4, -33.7], [151.4, -33.8], [151.3, -33.8]]],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": list(polarizations),
            "instrumentMode": "IW",
        },
    }


# a distinct constant per polarization, so the composite bands are identifiable
DOWNLOAD_VALUES = {"VV": -12.0, "VH": -20.0, "HH": -30.0}


def download_write_tif(path, value=-20.0, size=8):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return str(path)


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the Earth Engine download with a stub that writes 'image.<POLAR>.tif'."""
    calls = []

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        calls.append(polarization)
        return download_write_tif(
            Path(save_path) / f"image.{polarization}.tif",
            DOWNLOAD_VALUES.get(polarization, -20.0),
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)
    return calls


@pytest.fixture
def failing_download(monkeypatch):
    """Stub that succeeds for VV then fails for VH, to exercise the rollback."""

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        if polarization == "VH":
            raise RuntimeError("Earth Engine request failed")
        return download_write_tif(
            Path(save_path) / f"image.{polarization}.tif",
            DOWNLOAD_VALUES.get(polarization, -20.0),
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)


@pytest.fixture
def logger():
    return logging.getLogger("test_s1_download")


def run(tmp_path, polarizations, meta, logger, all_names=None):
    filepaths = SDS_tools.create_folder_structure(
        str(tmp_path), "S1", list(polarizations)
    )
    result = process_sentinel1_image(
        image_ee=None,
        im_meta=meta,
        inputs=download_inputs_for(polarizations),
        filepaths=filepaths,
        im_date=DATE,
        suffix=".tif",
        all_names=[] if all_names is None else all_names,
        logger=logger,
    )
    return result, filepaths


def s1_dir(tmp_path, polar):
    return tmp_path / "S1" / polar


# ------------------------------------------------------------------ happy path


def test_downloads_every_requested_polarization(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    assert result["skip_image"] is False
    assert fake_download == ["VV", "VH"]
    assert (s1_dir(tmp_path, "VV") / f"{DATE}_S1_{SITENAME}_VV.tif").exists()
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH.tif").exists()


def test_downloads_in_canonical_order(tmp_path, fake_download, logger):
    run(tmp_path, ["VH", "VV"], download_im_meta(), logger)

    assert fake_download == ["VV", "VH"]


def test_reports_the_band_order(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    assert result["band_order"] == ["VV", "VH"]
    assert result["filename_ms"] == f"{DATE}_S1_{SITENAME}_VV.tif"


def test_writes_one_metadata_file_for_the_scene(tmp_path, fake_download, logger):
    _, filepaths = run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    meta_files = sorted(Path(filepaths[0]).glob("*.txt"))
    assert [f.name for f in meta_files] == [f"{DATE}_S1_{SITENAME}.txt"]


def test_metadata_records_the_band_order(tmp_path, fake_download, logger):
    _, filepaths = run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    meta_info = read_metadata_file(str(Path(filepaths[0]) / f"{DATE}_S1_{SITENAME}.txt"))

    assert meta_info["band_order"] == ["VV", "VH"]
    assert meta_info["filename"] == f"{DATE}_S1_{SITENAME}_VV.tif"
    assert meta_info["epsg"] == 32756
    assert meta_info["instrumentMode"] == "IW"


def test_single_polarization_still_works(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VH"], download_im_meta(), logger)

    assert result["band_order"] == ["VH"]
    assert fake_download == ["VH"]
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH.tif").exists()


def test_each_polarization_is_registered_once(tmp_path, fake_download, logger):
    # get_file_name() adds the name itself; adding it again would misnumber duplicates
    all_names = []
    run(tmp_path, ["VV", "VH"], download_im_meta(), logger, all_names=all_names)

    assert all_names == [
        f"{DATE}_S1_{SITENAME}_VV.tif",
        f"{DATE}_S1_{SITENAME}_VH.tif",
    ]


def test_duplicate_scene_gets_matching_dup_suffixes(tmp_path, fake_download, logger):
    all_names = []
    run(tmp_path, ["VV", "VH"], download_im_meta(), logger, all_names=all_names)
    result, _ = run(tmp_path, ["VV", "VH"], download_im_meta(), logger, all_names=all_names)

    # both polarizations of the second scene must carry the SAME _dup counter,
    # or the read path could not pair them back up
    assert result["filename_ms"] == f"{DATE}_S1_{SITENAME}_VV_dup0.tif"
    assert (s1_dir(tmp_path, "VV") / f"{DATE}_S1_{SITENAME}_VV_dup0.tif").exists()
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH_dup0.tif").exists()


def test_only_the_polarization_folders_are_written(tmp_path, fake_download, logger):
    # the composite is built in memory for the preview, never saved as a .tif
    run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    assert sorted(p.name for p in (tmp_path / "S1").iterdir()) == ["VH", "VV", "meta"]


# ------------------------------------------------------------------ skipping and rollback


def test_skips_a_scene_missing_a_requested_polarization(tmp_path, fake_download, logger):
    result, filepaths = run(tmp_path, ["VV", "VH"], download_im_meta(["VV"]), logger)

    assert result["skip_image"] is True
    assert fake_download == []  # nothing downloaded at all
    assert list(Path(filepaths[0]).glob("*.txt")) == []


def test_partial_download_is_rolled_back(tmp_path, failing_download, logger):
    result, filepaths = run(tmp_path, ["VV", "VH"], download_im_meta(), logger)

    assert result["skip_image"] is True
    # the VV file that did download must not be left behind
    assert list(s1_dir(tmp_path, "VV").glob("*.tif")) == []
    assert list(s1_dir(tmp_path, "VH").glob("*.tif")) == []
    assert list(Path(filepaths[0]).glob("*.txt")) == []


def test_invalid_polarization_skips_the_image(tmp_path, fake_download, logger):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VH"])
    inputs = download_inputs_for(["VH"])
    inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] = ["XX"]

    result = process_sentinel1_image(
        image_ee=None,
        im_meta=download_im_meta(),
        inputs=inputs,
        filepaths=filepaths,
        im_date=DATE,
        suffix=".tif",
        all_names=[],
        logger=logger,
    )

    assert result["skip_image"] is True


# =================================================================================== #
# Tests for the Sentinel-1 metadata files: one per scene, recording the band order
# =================================================================================== #


OTHER_DATE = "2023-12-06-08-39-48"


def s1_metadict(polarizations, date=DATE):
    return {
        "filename": f"{date}_S1_{SITENAME}_{polarizations[0]}.tif",
        "epsg": 32756,
        "im_width": 168,
        "im_height": 436,
        "band_order": polarizations,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def metadata_legacy_metadict(date=OTHER_DATE):
    """A metadata file as written before multi-polarization support."""
    return {
        "filename": f"{date}_S1_{SITENAME}_VH.tif",
        "epsg": 32756,
        "im_width": 168,
        "im_height": 436,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "saved_polarization": "VH",
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def make_meta_dir(root, satname="S1"):
    meta = root / SITENAME / satname / "meta"
    meta.mkdir(parents=True)
    return meta


def metadata_inputs_for(root, sat_list=("S1",)):
    return {
        "filepath": str(root),
        "sitename": SITENAME,
        "sat_list": list(sat_list),
        "dates": ["2020-01-01", "2030-01-01"],
    }


# ------------------------------------------------------------------ write / read


def test_written_without_polar_has_no_suffix(tmp_path):
    write_metadata_file(str(tmp_path), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    assert (tmp_path / f"{DATE}_S1_{SITENAME}.txt").exists()


def test_legacy_polar_suffix_still_available(tmp_path):
    write_metadata_file(
        str(tmp_path), f"{DATE}_S1_{SITENAME}", metadata_legacy_metadict(), polar="VH"
    )

    assert (tmp_path / f"{DATE}_S1_{SITENAME}_VH.txt").exists()


def test_band_order_round_trips_as_a_list(tmp_path):
    write_metadata_file(str(tmp_path), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    meta_info = read_metadata_file(str(tmp_path / f"{DATE}_S1_{SITENAME}.txt"))

    assert meta_info["band_order"] == ["VV", "VH"]


# ------------------------------------------------------------------ get_band_order_from_meta


def test_band_order_is_preferred():
    assert get_band_order_from_meta({"band_order": ["VV", "VH"]}) == ["VV", "VH"]


def test_falls_back_to_saved_polarization():
    assert get_band_order_from_meta({"saved_polarization": "VH"}) == ["VH"]


def test_returns_empty_when_neither_is_recorded():
    assert get_band_order_from_meta({}) == []


def test_accepts_a_comma_separated_band_order():
    assert get_band_order_from_meta({"band_order": "VV, VH"}) == ["VV", "VH"]


# ------------------------------------------------------------------ get_metadata


def test_get_metadata_populates_band_order(tmp_path):
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    metadata = get_metadata(metadata_inputs_for(tmp_path))

    assert metadata["S1"]["band_order"] == [["VV", "VH"]]


def test_one_scene_produces_one_entry(tmp_path):
    # a dual-polarization scene gets ONE metadata file, so it must not be listed twice
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))
    write_metadata_file(
        str(meta), f"{OTHER_DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"], OTHER_DATE)
    )

    metadata = get_metadata(metadata_inputs_for(tmp_path))

    assert len(metadata["S1"]["filenames"]) == 2
    assert metadata["S1"]["filenames"] == [
        f"{DATE}_S1_{SITENAME}_VV.tif",
        f"{OTHER_DATE}_S1_{SITENAME}_VV.tif",
    ]


def test_band_order_is_index_aligned_with_filenames(tmp_path):
    # a session holding both a new dual-pol scene and a legacy VH-only one
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))
    write_metadata_file(
        str(meta), f"{OTHER_DATE}_S1_{SITENAME}", metadata_legacy_metadict(), polar="VH"
    )

    metadata = get_metadata(metadata_inputs_for(tmp_path))

    assert len(metadata["S1"]["band_order"]) == len(metadata["S1"]["filenames"])
    assert metadata["S1"]["band_order"] == [["VV", "VH"], ["VH"]]


def test_optical_metadata_has_no_band_order(tmp_path):
    meta = make_meta_dir(tmp_path, satname="S2")
    write_metadata_file(
        str(meta),
        f"{DATE}_S2_{SITENAME}",
        {
            "filename": f"{DATE}_S2_{SITENAME}_ms.tif",
            "epsg": 32756,
            "acc_georef": -1,
            "image_quality": "PASSED",
            "im_width": 100,
            "im_height": 100,
        },
    )

    metadata = get_metadata(metadata_inputs_for(tmp_path, sat_list=("S2",)))

    assert "band_order" not in metadata["S2"]


# =================================================================================== #
# Tests that resuming a download completes scenes missing a polarization
#
# A site first downloaded as VH-only must not stay single-polarization forever: matching
# on acquisition date alone made every existing scene look complete, so the missing VV was
# never fetched and the scene could never reach the segmentation model.
# =================================================================================== #


DATES = [
    pytz.utc.localize(datetime(2023, 12, 3, 19, 15, 49)),
    pytz.utc.localize(datetime(2023, 12, 6, 8, 39, 48)),
]
STAMPS = ["2023-12-03-19-15-49", "2023-12-06-08-39-48"]


def resume_write_tif(path):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 4, 4, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((4, 4), -20.0))
    dataset = None


def build_session(tmp_path, polarizations_per_scene):
    """Create S1/<POL>/ folders holding the given polarizations for each scene."""
    s1 = tmp_path / SITENAME / "S1"
    for polars in polarizations_per_scene:
        for polar in polars:
            (s1 / polar).mkdir(parents=True, exist_ok=True)
    for index, polars in enumerate(polarizations_per_scene):
        for polar in polars:
            resume_write_tif(s1 / polar / f"{STAMPS[index]}_S1_{SITENAME}_{polar}.tif")
    return {
        "filepath": str(tmp_path),
        "sitename": SITENAME,
        "sat_list": ["S1"],
    }


def metadata_s1_for(polarizations_per_scene):
    """Just the 'S1' entry, which is what the two helpers take."""
    return {
        "dates": list(DATES[: len(polarizations_per_scene)]),
        "filenames": [
            f"{STAMPS[i]}_S1_{SITENAME}_{polars[0]}.tif"
            for i, polars in enumerate(polarizations_per_scene)
        ],
        "band_order": [list(polars) for polars in polarizations_per_scene],
    }


def metadata_for(polarizations_per_scene):
    """The full metadata dict, which is what remove_existing_imagery takes."""
    return {"S1": metadata_s1_for(polarizations_per_scene)}


def resume_image_dict_for(dates):
    return {
        "S1": [
            {"properties": {"system:time_start": date.timestamp() * 1000}}
            for date in dates
        ]
    }


# ------------------------------------------------------- get_downloaded_sar_polarizations


def test_reads_the_polarizations_from_disk(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"], ["VH"]])

    by_date = get_downloaded_sar_polarizations(
        inputs, metadata_s1_for([["VV", "VH"], ["VH"]])
    )

    assert by_date[DATES[0]] == {"VV", "VH"}
    assert by_date[DATES[1]] == {"VH"}


def test_a_deleted_file_makes_the_scene_incomplete(tmp_path):
    # the metadata still claims VV+VH, but the VV .tif is gone
    inputs = build_session(tmp_path, [["VV", "VH"]])
    (tmp_path / SITENAME / "S1" / "VV" / f"{STAMPS[0]}_S1_{SITENAME}_VV.tif").unlink()

    by_date = get_downloaded_sar_polarizations(inputs, metadata_s1_for([["VV", "VH"]]))

    assert by_date[DATES[0]] == {"VH"}


def test_falls_back_to_band_order_when_files_are_unreadable(tmp_path):
    inputs = {"filepath": str(tmp_path / "nothing"), "sitename": SITENAME}

    by_date = get_downloaded_sar_polarizations(inputs, metadata_s1_for([["VV", "VH"]]))

    assert by_date[DATES[0]] == {"VV", "VH"}


# ------------------------------------------------------------- get_incomplete_sar_dates


@pytest.mark.parametrize(
    "on_disk,requested,is_incomplete",
    [
        (["VH"], ["VV", "VH"], True),
        (["VV", "VH"], ["VV", "VH"], False),
        # asking only for VH must not re-download a VH-only site
        (["VH"], ["VH"], False),
        (["VV", "VH"], ["VH"], False),
    ],
    ids=["vh-only-for-vv-vh", "complete", "vh-only-for-vh", "extra-on-disk"],
)
def test_get_incomplete_sar_dates(tmp_path, on_disk, requested, is_incomplete):
    inputs = build_session(tmp_path, [on_disk])

    incomplete = get_incomplete_sar_dates(inputs, metadata_s1_for([on_disk]), requested)

    assert incomplete == ({DATES[0]} if is_incomplete else set())


# ------------------------------------------------------------- remove_existing_imagery


def test_resume_redownloads_the_scene_missing_vv(tmp_path):
    # the regression: this scene used to be skipped and stay VH-only forever
    inputs = build_session(tmp_path, [["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }
    image_dict = resume_image_dict_for(DATES[:1])

    result = remove_existing_imagery(
        image_dict, metadata_for([["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1, "the incomplete scene was not queued for download"


def test_resume_skips_a_complete_scene(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    result = remove_existing_imagery(
        resume_image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 0


def test_resume_downloads_only_the_incomplete_scene(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"], ["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    result = remove_existing_imagery(
        resume_image_dict_for(DATES), metadata_for([["VV", "VH"], ["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1
    kept = datetime.fromtimestamp(
        result["S1"][0]["properties"]["system:time_start"] / 1000, tz=pytz.utc
    )
    assert kept == DATES[1], "the wrong scene was queued"


def test_inputs_is_required(tmp_path):
    # inputs carries the requested polarizations, so without it an incomplete scene
    # cannot be told from a complete one. Failing loudly beats silently leaving a site
    # single-polarization forever, so this must raise rather than fall back.
    with pytest.raises(TypeError):
        remove_existing_imagery(
            resume_image_dict_for(DATES[:1]), metadata_for([["VH"]]), ["S1"]
        )


def test_a_site_with_no_sentinel_1_properties_uses_the_default_polarizations(tmp_path):
    # inputs need not name the polarizations; the VV+VH default still applies, so a
    # legacy VH-only site is completed even by a caller that sets nothing
    inputs = build_session(tmp_path, [["VH"]])

    result = remove_existing_imagery(
        resume_image_dict_for(DATES[:1]), metadata_for([["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1


def test_optical_satellites_are_unaffected(tmp_path):
    metadata = {"L9": {"dates": DATES[:1], "filenames": ["a.tif"]}}
    image_dict = {
        "L9": [
            {"properties": {"system:time_start": date.timestamp() * 1000}}
            for date in DATES
        ]
    }

    result = remove_existing_imagery(image_dict, metadata, ["L9"], {"sitename": "s"})

    assert len(result["L9"]) == 1  # only the second date is new


def test_s1_and_an_optical_satellite_in_one_call(tmp_path):
    # the S1 completeness check must not disturb the date-only filtering of the others
    inputs = build_session(tmp_path, [["VH"]])
    inputs["sat_list"] = ["S1", "L9"]
    metadata = metadata_for([["VH"]])
    metadata["L9"] = {"dates": DATES[:1], "filenames": ["a.tif"]}
    image_dict = resume_image_dict_for(DATES)
    image_dict["L9"] = [
        {"properties": {"system:time_start": date.timestamp() * 1000}}
        for date in DATES
    ]

    result = remove_existing_imagery(image_dict, metadata, ["S1", "L9"], inputs)

    assert len(result["S1"]) == 2  # the incomplete scene, plus the genuinely new one
    assert len(result["L9"]) == 1  # only the second date is new


def test_a_satellite_with_no_metadata_keeps_every_image(tmp_path):
    # nothing downloaded yet: the site has no S1 entry at all
    inputs = build_session(tmp_path, [])

    result = remove_existing_imagery(resume_image_dict_for(DATES), {}, ["S1"], inputs)

    assert len(result["S1"]) == 2


def test_a_satellite_with_no_downloaded_dates_keeps_every_image(tmp_path):
    inputs = build_session(tmp_path, [])
    metadata = {"S1": {"dates": [], "filenames": [], "band_order": []}}

    result = remove_existing_imagery(resume_image_dict_for(DATES), metadata, ["S1"], inputs)

    assert len(result["S1"]) == 2


def test_no_available_images_leaves_the_dict_empty(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    result = remove_existing_imagery(
        {"S1": []}, metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert result["S1"] == []


def test_a_polarization_that_was_never_downloaded_requeues_the_scene(tmp_path):
    # asking for HH on a VV+VH site: the scene is incomplete for that request
    inputs = build_session(tmp_path, [["VV", "VH"]])
    inputs["sentinel_1_properties"] = {"transmitterReceiverPolarisation": ["HH"]}

    result = remove_existing_imagery(
        resume_image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1


def test_it_reports_how_many_scenes_are_incomplete(tmp_path, capsys):
    inputs = build_session(tmp_path, [["VH"], ["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    remove_existing_imagery(
        resume_image_dict_for(DATES), metadata_for([["VH"], ["VH"]]), ["S1"], inputs
    )

    printed = capsys.readouterr().out
    assert "2 existing images are missing one of VV+VH" in printed


def test_a_complete_site_reports_nothing_to_download(tmp_path, capsys):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    remove_existing_imagery(
        resume_image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    printed = capsys.readouterr().out
    assert "missing one of" not in printed
    assert "1 images already exist, 0 to download" in printed


# =================================================================================== #
# End-to-end resume of a legacy VH-only Sentinel-1 site
#
# The resume section above covers the helpers in isolation, against hand-built metadata
# dicts. This drives the whole resume instead: a real VH-only site on disk (VH .tif plus a
# '_VH.txt' metadata file carrying 'saved_polarization', exactly as written before
# multi-polarization support), read back by the real get_metadata, filtered by
# remove_existing_imagery, then downloaded by process_sentinel1_image with Earth Engine
# stubbed out.
#
# The point of the fix is not that a scene is *queued* - it is that VV ends up on disk and
# the scene can finally build the 3-channel [VV, VH, VV-VH] composite the segmentation
# model needs. That is what these tests assert.
# =================================================================================== #


STAMP = "2023-12-03-19-15-49"
OTHER_STAMP = "2023-12-06-08-39-48"

# a distinct constant per polarization, so a band can be identified after stacking
RESUME_VH_VALUES = {"VV": -12.0, "VH": -20.0}


def resume_vh_write_tif(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((SIZE, SIZE), value, dtype="float64"))
    dataset = None
    return str(path)


def resume_vh_legacy_metadict(stamp):
    """A metadata file as written before multi-polarization support: no 'band_order'."""
    return {
        "filename": f"{stamp}_S1_{SITENAME}_VH.tif",
        "epsg": 32756,
        "im_width": SIZE,
        "im_height": SIZE,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "saved_polarization": "VH",
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def build_legacy_vh_site(tmp_path, stamps=(STAMP,)):
    """Create a site as an older coastsat left it: S1/VH/ and '*_VH.txt' only."""
    site = tmp_path / SITENAME
    meta_dir = site / "S1" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        resume_vh_write_tif(
            site / "S1" / "VH" / f"{stamp}_S1_{SITENAME}_VH.tif", RESUME_VH_VALUES["VH"]
        )
        write_metadata_file(
            str(meta_dir), f"{stamp}_S1_{SITENAME}", resume_vh_legacy_metadict(stamp), polar="VH"
        )
    return site


def resume_vh_inputs_for(tmp_path, polarizations=("VV", "VH")):
    return {
        "filepath": str(tmp_path),
        "sitename": SITENAME,
        "sat_list": ["S1"],
        "dates": ["2020-01-01", "2030-01-01"],
        "polygon": [[[151.3, -33.7], [151.4, -33.7], [151.4, -33.8], [151.3, -33.8]]],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": list(polarizations),
            "instrumentMode": "IW",
        },
    }


def resume_vh_band(polarization):
    return {
        "id": polarization,
        "crs": "EPSG:32756",
        "crs_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0],
        "dimensions": [SIZE, SIZE],
    }


def resume_vh_im_meta(polarizations=("VV", "VH")):
    return {
        "id": IMAGE_ID,
        "bands": [resume_vh_band(polar) for polar in polarizations],
        "properties": {
            "orbitProperties_pass": "DESCENDING",
            "transmitterReceiverPolarisation": list(polarizations),
            "resolution": "H",
            "resolution_meters": 10,
            "instrumentMode": "IW",
            "system:time_start": 0,  # replaced per scene by image_dict_for
        },
    }


def resume_vh_image_dict_for(stamps):
    """The 'available on the server' dict, as retrieve_images builds it from GEE."""
    images = []
    for stamp in stamps:
        image = resume_vh_im_meta()
        date = SDS_download.parse_date_from_filename(stamp)
        image["properties"]["system:time_start"] = date.timestamp() * 1000
        images.append(image)
    return {"S1": images}


@pytest.fixture
def resume_vh_fake_download(monkeypatch):
    """Replace the Earth Engine download with a stub that writes 'image.<POLAR>.tif'."""
    calls = []

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        calls.append(polarization)
        return resume_vh_write_tif(
            Path(save_path) / f"image.{polarization}.tif", RESUME_VH_VALUES[polarization]
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)
    return calls


def resume(tmp_path, inputs):
    """Run the resume the way retrieve_images does: filter, then download what is left.

    Returns the image_dict that survived remove_existing_imagery.
    """
    logger = logging.getLogger("test_s1_resume_completes_vh_only_site")
    polarizations = SDS_download.get_sentinel1_polarizations(inputs)
    stamps = [
        SDS_download.parse_date_from_filename(name).strftime("%Y-%m-%d-%H-%M-%S")
        for name in [STAMP, OTHER_STAMP]
    ]
    available = resume_vh_image_dict_for(
        [stamp for stamp in stamps if _scene_exists(tmp_path, stamp)]
    )

    to_download = remove_existing_imagery(
        available, get_metadata(inputs), ["S1"], inputs
    )

    filepaths = SDS_tools.create_folder_structure(
        str(tmp_path / SITENAME), "S1", list(polarizations)
    )
    all_names = []
    for image in to_download["S1"]:
        im_date = SDS_download.parse_date_from_filename(
            _stamp_of(image)
        ).strftime("%Y-%m-%d-%H-%M-%S")
        process_sentinel1_image(
            image_ee=None,
            im_meta=image,
            inputs=inputs,
            filepaths=filepaths,
            im_date=im_date,
            suffix=".tif",
            all_names=all_names,
            logger=logger,
        )
    return to_download


def _stamp_of(image):
    from datetime import datetime

    return datetime.fromtimestamp(
        image["properties"]["system:time_start"] / 1000, tz=pytz.utc
    ).strftime("%Y-%m-%d-%H-%M-%S")


def _scene_exists(tmp_path, stamp):
    """Whether the server offers this scene - here, whether the site was built with it."""
    meta_dir = tmp_path / SITENAME / "S1" / "meta"
    return any(meta_dir.glob(f"{stamp}_S1_{SITENAME}*.txt"))


def tif(tmp_path, polar, stamp=STAMP):
    return tmp_path / SITENAME / "S1" / polar / f"{stamp}_S1_{SITENAME}_{polar}.tif"


# ------------------------------------------------------------------ the precondition


def test_the_legacy_site_starts_out_vh_only(tmp_path):
    build_legacy_vh_site(tmp_path)

    metadata = get_metadata(resume_vh_inputs_for(tmp_path))

    assert metadata["S1"]["band_order"] == [["VH"]]  # via 'saved_polarization'
    assert not (tmp_path / SITENAME / "S1" / "VV").exists()


def test_the_legacy_scene_cannot_build_the_model_composite(tmp_path):
    # this is what the resume has to fix: no VV means no 3-channel composite, so the
    # scene falls back to Otsu and can never reach the segmentation model
    build_legacy_vh_site(tmp_path)
    inputs = resume_vh_inputs_for(tmp_path)
    metadata = get_metadata(inputs)

    im_polar, _ = SDS_tools.read_sar_image(
        SDS_tools.get_filenames(
            metadata["S1"]["filenames"][0], SDS_tools.get_filepath(inputs, "S1"), "S1"
        )
    )

    assert im_polar.shape == (SIZE, SIZE, 1)
    assert SDS_tools.build_sar_composite(im_polar, ["VH"]) is None


# ------------------------------------------------------------------ the resume


def test_resume_completes_the_vh_only_scene(tmp_path, resume_vh_fake_download):
    build_legacy_vh_site(tmp_path)
    inputs = resume_vh_inputs_for(tmp_path)

    to_download = resume(tmp_path, inputs)

    assert len(to_download["S1"]) == 1, "the incomplete scene was not queued"
    assert tif(tmp_path, "VV").exists(), "VV was never downloaded"
    assert resume_vh_fake_download == ["VV", "VH"]
    # the scene is re-downloaded whole, so VH is fetched again - it must overwrite the
    # existing file rather than land beside it as a '_dup0'
    vh_files = sorted(p.name for p in (tmp_path / SITENAME / "S1" / "VH").glob("*.tif"))
    assert vh_files == [f"{STAMP}_S1_{SITENAME}_VH.tif"]
    assert ["VV", "VH"] in get_metadata(inputs)["S1"]["band_order"]


def test_after_the_resume_the_scene_builds_the_model_composite(tmp_path, resume_vh_fake_download):
    # the whole point: the completed scene now stacks to the 3-channel composite the
    # ONNX model needs, instead of falling back to Otsu on VH
    build_legacy_vh_site(tmp_path)
    inputs = resume_vh_inputs_for(tmp_path)
    resume(tmp_path, inputs)

    filepath = SDS_tools.get_filepath(inputs, "S1")
    file_paths = SDS_tools.get_filenames(
        f"{STAMP}_S1_{SITENAME}_VV.tif", filepath, "S1"
    )
    im_polar, _ = SDS_tools.read_sar_image(file_paths)
    composite = SDS_tools.build_sar_composite(im_polar, ["VV", "VH"])

    assert [Path(fp).parent.name for fp in file_paths] == ["VV", "VH"]
    assert im_polar.shape == (SIZE, SIZE, 2)
    assert composite is not None
    assert composite.shape == (SIZE, SIZE, 3)
    # channels in canonical order: VV, VH, VV-VH
    assert np.allclose(composite[:, :, 0], RESUME_VH_VALUES["VV"])
    assert np.allclose(composite[:, :, 1], RESUME_VH_VALUES["VH"])
    assert np.allclose(composite[:, :, 2], RESUME_VH_VALUES["VV"] - RESUME_VH_VALUES["VH"])


def test_a_second_resume_downloads_nothing(tmp_path, resume_vh_fake_download):
    # a completed site must not re-download on every run
    build_legacy_vh_site(tmp_path)
    inputs = resume_vh_inputs_for(tmp_path)
    resume(tmp_path, inputs)
    resume_vh_fake_download.clear()

    to_download = resume(tmp_path, inputs)

    assert to_download["S1"] == []
    assert resume_vh_fake_download == []


def test_a_vh_only_request_leaves_the_legacy_site_alone(tmp_path, resume_vh_fake_download):
    # a user who pins transmitterReceiverPolarisation to VH must not have their whole
    # site re-downloaded by the new completeness check
    build_legacy_vh_site(tmp_path)

    to_download = resume(tmp_path, resume_vh_inputs_for(tmp_path, ["VH"]))

    assert to_download["S1"] == []
    assert resume_vh_fake_download == []
    assert not (tmp_path / SITENAME / "S1" / "VV").exists()


def test_the_stale_legacy_metadata_file_is_left_behind(tmp_path, resume_vh_fake_download):
    """Documents current behaviour, not desired behaviour.

    The legacy file is '<stamp>_S1_<site>_VH.txt'; the resume writes the modern
    '<stamp>_S1_<site>.txt' beside it and never removes the old one, so get_metadata
    lists the completed scene twice - once as ['VV','VH'] and once as ['VH'].

    The resume itself stays correct: get_downloaded_sar_polarizations unions the
    polarizations per date, so both entries resolve to VV+VH and a second run downloads
    nothing (see test_a_second_resume_downloads_nothing). The cost is that extraction
    maps the scene twice, producing duplicate shorelines for that date unless
    SDS_tools.remove_duplicates is called.
    """
    build_legacy_vh_site(tmp_path)
    inputs = resume_vh_inputs_for(tmp_path)

    resume(tmp_path, inputs)

    meta_dir = tmp_path / SITENAME / "S1" / "meta"
    assert sorted(p.name for p in meta_dir.iterdir()) == [
        f"{STAMP}_S1_{SITENAME}.txt",
        f"{STAMP}_S1_{SITENAME}_VH.txt",
    ]
    metadata = get_metadata(inputs)["S1"]
    assert len(metadata["dates"]) == 2
    assert metadata["band_order"] == [["VV", "VH"], ["VH"]]
    # both entries point at the same scene, so both read the same two files
    for filename in metadata["filenames"]:
        paths = SDS_tools.get_filenames(
            filename, SDS_tools.get_filepath(inputs, "S1"), "S1"
        )
        assert [Path(p).parent.name for p in paths] == ["VV", "VH"]


def test_resume_completes_only_the_incomplete_scene(tmp_path, resume_vh_fake_download):
    # a site holding one dual-pol scene and one legacy VH-only scene: only the legacy
    # one is fetched again
    build_legacy_vh_site(tmp_path, stamps=(STAMP, OTHER_STAMP))
    inputs = resume_vh_inputs_for(tmp_path)
    # promote the first scene to VV+VH, as a newer download would have left it
    resume_vh_write_tif(tif(tmp_path, "VV"), RESUME_VH_VALUES["VV"])
    write_metadata_file(
        str(tmp_path / SITENAME / "S1" / "meta"),
        f"{STAMP}_S1_{SITENAME}",
        {**resume_vh_legacy_metadict(STAMP), "band_order": ["VV", "VH"]},
    )
    (tmp_path / SITENAME / "S1" / "meta" / f"{STAMP}_S1_{SITENAME}_VH.txt").unlink()

    to_download = resume(tmp_path, inputs)

    assert len(to_download["S1"]) == 1
    assert _stamp_of(to_download["S1"][0]) == OTHER_STAMP
    assert tif(tmp_path, "VV", OTHER_STAMP).exists()


# =================================================================================== #
# Tests for reading the polarization files of a SAR scene into one stacked array
# =================================================================================== #


def read_write_tif(path, value, size=8, geotransform=None):
    """Write a small single-band Float64 .tif, like an Earth Engine S1 download."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geotransform or GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return str(path)


def test_stacks_polarizations_in_the_order_given(tmp_path):
    vv = read_write_tif(tmp_path / "vv.tif", -12.0)
    vh = read_write_tif(tmp_path / "vh.tif", -20.0)

    im, georef = read_sar_image([vv, vh])

    assert im.shape == (8, 8, 2)
    assert np.all(im[:, :, 0] == -12.0)
    assert np.all(im[:, :, 1] == -20.0)
    assert np.allclose(georef, GEOTRANSFORM)


def test_reversing_the_file_list_reverses_the_channels(tmp_path):
    vv = read_write_tif(tmp_path / "vv.tif", -12.0)
    vh = read_write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image([vh, vv])

    assert np.all(im[:, :, 0] == -20.0)
    assert np.all(im[:, :, 1] == -12.0)


def test_single_file_returns_one_channel(tmp_path):
    # a scene downloaded before multi-polarization support
    vh = read_write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image([vh])

    assert im.shape == (8, 8, 1)
    assert np.all(im[:, :, 0] == -20.0)


def test_read_accepts_a_bare_string_path(tmp_path):
    vh = read_write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image(vh)

    assert im.shape == (8, 8, 1)


def test_raises_when_shapes_differ(tmp_path):
    vv = read_write_tif(tmp_path / "vv.tif", -12.0, size=8)
    vh = read_write_tif(tmp_path / "vh.tif", -20.0, size=6)

    with pytest.raises(ValueError, match="same grid"):
        read_sar_image([vv, vh])


def test_raises_when_geotransforms_differ(tmp_path):
    vv = read_write_tif(tmp_path / "vv.tif", -12.0)
    vh = read_write_tif(
        tmp_path / "vh.tif",
        -20.0,
        geotransform=[600000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0],
    )

    with pytest.raises(ValueError, match="same grid"):
        read_sar_image([vv, vh])


@pytest.mark.parametrize(
    "names,error",
    [(["does_not_exist.tif"], IOError), ([], ValueError)],
    ids=["missing-file", "empty-file-list"],
)
def test_read_raises_on_unreadable_input(tmp_path, names, error):
    with pytest.raises(error):
        read_sar_image([str(tmp_path / name) for name in names])


# =================================================================================== #
# Tests for the per-pixel validity mask of a SAR scene
#
# The segmentation model neutralises invalid pixels on input and blanks them out of the
# prediction, so this mask decides which pixels are allowed to become a land/water call.
# =================================================================================== #


def mask_write_tif(path, array, nodata=None):
    """Write a single-band Float64 .tif, like an Earth Engine S1 download."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array.astype("float64"))
    dataset = None
    return str(path)


def filled(value):
    return np.full((SIZE, SIZE), value, dtype="float64")


def test_all_valid_when_no_nodata_is_declared(tmp_path):
    vv = mask_write_tif(tmp_path / "vv.tif", filled(-12.0))

    valid = read_sar_valid_mask([vv])

    assert valid.shape == (SIZE, SIZE)
    assert valid.dtype == bool
    assert np.all(valid)


def test_nodata_pixels_are_invalid(tmp_path):
    # the downloaded Sentinel-1 rasters declare nodata = 0.0
    array = filled(-12.0)
    array[2, 3] = 0.0
    vv = mask_write_tif(tmp_path / "vv.tif", array, nodata=0.0)

    valid = read_sar_valid_mask([vv])

    assert not valid[2, 3]
    assert valid.sum() == SIZE * SIZE - 1


def test_non_finite_pixels_are_invalid(tmp_path):
    array = filled(-12.0)
    array[0, 0] = np.nan
    array[1, 1] = np.inf
    vv = mask_write_tif(tmp_path / "vv.tif", array)

    valid = read_sar_valid_mask([vv])

    assert not valid[0, 0]
    assert not valid[1, 1]
    assert valid.sum() == SIZE * SIZE - 2


def test_masks_are_combined_across_polarizations(tmp_path):
    # a pixel is only usable if EVERY polarization of the scene has data there,
    # because VV-VH is meaningless when either band is missing
    vv_array = filled(-12.0)
    vv_array[1, 1] = 0.0
    vh_array = filled(-20.0)
    vh_array[5, 5] = 0.0
    vv = mask_write_tif(tmp_path / "vv.tif", vv_array, nodata=0.0)
    vh = mask_write_tif(tmp_path / "vh.tif", vh_array, nodata=0.0)

    valid = read_sar_valid_mask([vv, vh])

    assert not valid[1, 1]
    assert not valid[5, 5]
    assert valid.sum() == SIZE * SIZE - 2


def test_mask_accepts_a_bare_string_path(tmp_path):
    vh = mask_write_tif(tmp_path / "vh.tif", filled(-20.0))

    assert read_sar_valid_mask(vh).shape == (SIZE, SIZE)


@pytest.mark.parametrize(
    "names,error",
    [(["does_not_exist.tif"], IOError), ([], ValueError)],
    ids=["missing-file", "empty-file-list"],
)
def test_mask_raises_on_unreadable_input(tmp_path, names, error):
    with pytest.raises(error):
        read_sar_valid_mask([str(tmp_path / name) for name in names])


# =================================================================================== #
# Tests for the [VV, VH, VV-VH] Sentinel-1 composite and the image Otsu runs on
# =================================================================================== #


COMPOSITE_SIZE = 6


def composite_polarization_stack(values):
    """(H, W, n) stack where channel k is filled with values[k]."""
    shape = (COMPOSITE_SIZE, COMPOSITE_SIZE)
    return np.stack([np.full(shape, value, dtype="float64") for value in values], axis=2)


# ------------------------------------------------------------------ build_sar_composite


def test_third_band_is_the_db_difference():
    im = composite_polarization_stack([-12.0, -20.0])

    composite = SDS_tools.build_sar_composite(im, ["VV", "VH"])

    assert composite.shape == (COMPOSITE_SIZE, COMPOSITE_SIZE, 3)
    assert np.all(composite[:, :, 0] == -12.0)  # VV
    assert np.all(composite[:, :, 1] == -20.0)  # VH
    assert np.all(composite[:, :, 2] == 8.0)  # VV - VH


def test_band_order_follows_the_composite_bands_not_the_input():
    # VH first on the way in, but the composite is always [VV, VH, VV-VH]
    im = composite_polarization_stack([-20.0, -12.0])

    composite = SDS_tools.build_sar_composite(im, ["VH", "VV"])

    assert np.all(composite[:, :, 0] == -12.0)
    assert np.all(composite[:, :, 1] == -20.0)
    assert np.all(composite[:, :, 2] == 8.0)


def test_extra_polarizations_are_ignored():
    im = composite_polarization_stack([-12.0, -20.0, -30.0])

    composite = SDS_tools.build_sar_composite(im, ["VV", "VH", "HH"])

    assert composite.shape == (COMPOSITE_SIZE, COMPOSITE_SIZE, 3)
    assert np.all(composite[:, :, 2] == 8.0)


@pytest.mark.parametrize("polarizations", [["VH"], ["VV"], ["HH", "HV"]])
def test_returns_none_without_both_vv_and_vh(polarizations):
    im = composite_polarization_stack([-20.0] * len(polarizations))

    assert SDS_tools.build_sar_composite(im, polarizations) is None


# ------------------------------------------------------------------ get_sar_otsu_image


def test_thresholds_the_mean_of_vv_and_vh():
    composite = SDS_tools.build_sar_composite(
        composite_polarization_stack([-12.0, -20.0]), ["VV", "VH"]
    )

    im_sar = get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")])

    # (-12 + -20) / 2. The VV-VH band must stay out of it: np.mean over all three bands
    # is (2/3)*VV = -8, which cancels VH entirely and thresholds VV alone.
    assert np.all(im_sar == -16.0)


@pytest.mark.parametrize("fn", [[s1_path("VH")], s1_path("VH")], ids=["list", "bare-string"])
def test_single_polarization_thresholds_that_band(fn):
    im = composite_polarization_stack([-20.0])

    assert np.all(get_sar_otsu_image(im, fn) == -20.0)


def test_result_is_two_dimensional():
    composite = SDS_tools.build_sar_composite(
        composite_polarization_stack([-12.0, -20.0]), ["VV", "VH"]
    )

    # otsu_threshold and find_shoreline_SAR both require a 2D array
    assert get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")]).ndim == 2


def test_extra_polarizations_do_not_enter_the_mean():
    # VV+VH+HH: the mean is still VV and VH only
    im = composite_polarization_stack([-12.0, -20.0, -30.0])
    fn = [s1_path("VV"), s1_path("VH"), s1_path("HH")]

    assert np.all(get_sar_otsu_image(im, fn) == -16.0)


# ------------------------------------------------------------------ no smoothing


def test_otsu_image_is_not_smoothed():
    # the SAR imagery is thresholded as acquired: no median/speckle filter, so a
    # single outlying pixel must survive into the image Otsu sees
    im = composite_polarization_stack([-12.0, -20.0])
    im[3, 3, 0] = 40.0  # a lone bright pixel in VV
    composite = SDS_tools.build_sar_composite(im, ["VV", "VH"])

    im_sar = get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")])

    assert im_sar[3, 3] == pytest.approx((40.0 + -20.0) / 2)
    # a median filter would have replaced it with the surrounding value
    assert im_sar[3, 3] != pytest.approx(-16.0)


def test_extract_shorelines_does_not_filter_the_sar_image():
    # guards against a median/gaussian filter being reintroduced in the S1 branch
    source = inspect.getsource(SDS_shoreline.extract_shorelines)

    for smoother in ("median_filter", "gaussian_filter", "uniform_filter", "convolve"):
        assert smoother not in source, f"{smoother} reintroduced into extract_shorelines"


# ------------------------------------------------------------------ logging


@pytest.mark.parametrize(
    "polarizations,expected",
    [
        (["VV", "VH"], "the mean of VV and VH"),
        (["VV", "VH", "HH"], "the mean of VV and VH"),
        (["VH"], "VH"),
        (["VV"], "VV"),
    ],
)
def test_describe_matches_what_is_thresholded(polarizations, expected):
    assert describe_sar_otsu_image([s1_path(p) for p in polarizations]) == expected


# =================================================================================== #
# End-to-end test of the SAR read path, from a session on disk to the image thresholded
#
# Walks get_filepath -> get_filenames -> preprocess_image -> get_sar_otsu_image against
# real .tif files. A dual-polarization scene must reach Otsu as the mean of VV and VH,
# while a scene with only VH must still threshold VH exactly as it did before.
# =================================================================================== #


# a distinct constant per polarization, so the channel that ends up being read is obvious
PIPELINE_VALUES = {"VV": -12.0, "VH": -22.0, "HH": -32.0}


def pipeline_write_tif(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((SIZE, SIZE), value, dtype="float64"))
    dataset = None


def pipeline_make_session(root, polarizations):
    for polar in polarizations:
        pipeline_write_tif(
            root / SITENAME / "S1" / polar / f"{DATE}_S1_{SITENAME}_{polar}.tif",
            PIPELINE_VALUES[polar],
        )
    return {"filepath": str(root), "sitename": SITENAME}


def read_scene(root, polarizations, primary=None):
    """Resolve and read a scene exactly the way extract_shorelines does."""
    inputs = pipeline_make_session(root, polarizations)
    filepath = SDS_tools.get_filepath(inputs, "S1")
    filename = f"{DATE}_S1_{SITENAME}_{primary or polarizations[0]}.tif"
    fn = SDS_tools.get_filenames(filename, filepath, "S1")
    im_ms, georef, cloud_mask, im_nodata = preprocess_image(fn, "S1", {}, "C02")
    return fn, im_ms, georef, cloud_mask, im_nodata


def test_dual_polarization_scene_returns_the_composite(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV", "VH"])

    assert len(fn) == 2
    assert im_ms.shape == (SIZE, SIZE, 3)  # [VV, VH, VV-VH]
    assert np.all(im_ms[:, :, 0] == PIPELINE_VALUES["VV"])
    assert np.all(im_ms[:, :, 1] == PIPELINE_VALUES["VH"])
    assert np.all(im_ms[:, :, 2] == PIPELINE_VALUES["VV"] - PIPELINE_VALUES["VH"])


def test_dual_polarization_scene_thresholds_the_mean_of_vv_and_vh(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV", "VH"])

    im_sar = get_sar_otsu_image(im_ms, fn)

    assert np.all(im_sar == (PIPELINE_VALUES["VV"] + PIPELINE_VALUES["VH"]) / 2.0)


def test_legacy_vh_only_session_is_unchanged(tmp_path):
    # no VV, so no composite: it stays a single band and Otsu still sees plain VH
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VH"])

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == PIPELINE_VALUES["VH"])


def test_mixed_session_thresholds_vh_for_a_legacy_scene(tmp_path):
    # a VV+VH session that also contains a scene downloaded before VV was added
    inputs = pipeline_make_session(tmp_path, ["VV", "VH"])
    (tmp_path / SITENAME / "S1" / "VV" / f"{DATE}_S1_{SITENAME}_VV.tif").unlink()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(f"{DATE}_S1_{SITENAME}_VH.tif", filepath, "S1")
    im_ms, _, _, _ = preprocess_image(fn, "S1", {}, "C02")

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == PIPELINE_VALUES["VH"])


def test_vv_only_session_thresholds_vv(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV"])

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == PIPELINE_VALUES["VV"])
    assert get_sar_band_index(fn) == 0


def test_masks_match_the_image_footprint(tmp_path):
    _, im_ms, _, cloud_mask, im_nodata = read_scene(tmp_path, ["VV", "VH"])

    # SAR carries no cloud or no-data information, but the masks must still be 2D
    assert cloud_mask.shape == im_ms.shape[:2]
    assert im_nodata.shape == im_ms.shape[:2]
    assert not cloud_mask.any()
    assert not im_nodata.any()


def test_channel_order_does_not_depend_on_the_metadata_filename(tmp_path):
    # the metadata records the VV file, but starting from VH must give the same stack
    _, from_vv, _, _, _ = read_scene(tmp_path, ["VV", "VH"], primary="VV")
    _, from_vh, _, _, _ = read_scene(tmp_path, ["VV", "VH"], primary="VH")

    assert np.array_equal(from_vv, from_vh)


# =================================================================================== #
# Tests that SAR shoreline detection still thresholds VH once VV is also downloaded
#
# im_ms now holds one channel per polarization, so the channel to threshold is chosen
# by polarization name rather than by position.
# =================================================================================== #


def detection_polarization_stack(values, size=4):
    """An (H, W, n) stack with one constant band per value."""
    return np.stack([np.full((size, size), v, dtype="float64") for v in values], axis=2)


@pytest.mark.parametrize(
    "fn,expected,reason",
    [
        ([s1_path("VV"), s1_path("VH")], 1, "VV+VH: VH is the second channel"),
        ([s1_path("VH"), s1_path("VV")], 0, "reversed: VH is the first channel"),
        ([s1_path("VH")], 0, "legacy VH-only scene"),
        ([s1_path("VV")], 0, "VV-only request falls back to the only channel"),
        ([s1_path("HH"), s1_path("HV")], 0, "no VH present, falls back to channel 0"),
    ],
)
def test_get_sar_band_index(fn, expected, reason):
    assert get_sar_band_index(fn) == expected, reason


def test_handles_duplicate_named_files():
    fn = [s1_path("VV", "_dup0"), s1_path("VH", "_dup0")]

    assert get_sar_band_index(fn) == 1


def test_detection_accepts_a_bare_string_path():
    assert get_sar_band_index(s1_path("VH")) == 0


def test_polarization_is_configurable():
    fn = [s1_path("VV"), s1_path("VH")]

    assert get_sar_band_index(fn, polarization="VV") == 0


# ------------------------------------------- settings['sar_detection_polarization']


# a composite, which is what preprocess_image hands over for a dual-polarization scene
COMPOSITE = detection_polarization_stack([-12.0, -20.0, 8.0])


@pytest.mark.parametrize(
    "settings,expected",
    [
        (None, None),
        ({}, None),
        ({"sar_detection_polarization": None}, None),
        ({"sar_detection_polarization": ""}, None),
        ({"sar_detection_polarization": "vv"}, "VV"),
        ({"sar_detection_polarization": " VH "}, "VH"),
    ],
)
def test_get_sar_detection_polarization(settings, expected):
    assert get_sar_detection_polarization(settings) == expected


@pytest.mark.parametrize(
    "settings,value,description",
    [
        # dual-polarization still thresholds the mean of VV and VH, as it always has
        ({}, -16.0, "the mean of VV and VH"),
        ({"sar_detection_polarization": "VV"}, -12.0, "VV"),
        ({"sar_detection_polarization": "VH"}, -20.0, "VH"),
        # a band the scene does not carry: one odd scene in a run must still map rather
        # than raising, and the log reports the band actually thresholded
        ({"sar_detection_polarization": "HV"}, -16.0, "the mean of VV and VH"),
    ],
    ids=["absent", "VV", "VH", "not-in-the-scene"],
)
def test_the_setting_selects_the_thresholded_band(settings, value, description):
    assert np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == value)
    assert describe_sar_otsu_image(DUAL_POL, settings) == description


def test_the_setting_selects_a_band_of_a_single_polarization_scene():
    im = detection_polarization_stack([-12.0, -30.0])
    fn = [s1_path("VV"), s1_path("HH")]  # no VH, so no composite was built

    settings = {"sar_detection_polarization": "HH"}

    assert np.all(get_sar_otsu_image(im, fn, settings) == -30.0)
    assert describe_sar_otsu_image(fn, settings) == "HH"


def test_the_difference_band_cannot_be_selected_by_a_polarization_name():
    # VV-VH is a derived band, not a polarization, so no setting value reaches it
    for polarization in SDS_tools.SAR_POLARIZATIONS:
        settings = {"sar_detection_polarization": polarization}
        assert not np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == 8.0)


def test_an_unknown_polarization_is_warned_about_once():
    logger = RecordingLogger()

    warn_unknown_sar_detection_polarization(
        {"sar_detection_polarization": "VVH"}, logger=logger
    )

    assert len(logger.warnings) == 1
    assert "VVH" in logger.warnings[0]


@pytest.mark.parametrize("settings", [{}, {"sar_detection_polarization": "VV"}])
def test_no_warning_for_the_default_or_a_real_polarization(settings):
    logger = RecordingLogger()

    warn_unknown_sar_detection_polarization(settings, logger=logger)

    assert logger.warnings == []


# =================================================================================== #
# Tests that the SAR Otsu path filters small objects in pixels, not square metres
#
# settings['min_beach_area'] is in m^2. The optical path divides it by pixel_size**2
# before handing it to the classifier; the SAR Otsu path used to pass it straight through
# to morphology.remove_small_objects, which counts pixels. At the 10 m S1 pixel that made
# the filter 100x too aggressive and silently deleted real shorelines.
# =================================================================================== #


def image_with_a_small_and_a_large_bright_region():
    """A 20x20 image with a 9-pixel blob and a 200-pixel region above the threshold."""
    im = np.full((20, 20), -20.0)
    im[2:5, 2:5] = -5.0  # 9 pixels
    im[10:20, :] = -5.0  # 200 pixels
    return im


def test_the_argument_is_counted_in_pixels():
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, 4)

    assert kept[2:5, 2:5].all(), "a 9-pixel object must survive a 4-pixel minimum"
    assert kept[10:, :].all()


def test_objects_smaller_than_the_minimum_are_dropped():
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, 16)

    assert not kept[2:5, 2:5].any(), "a 9-pixel object must not survive a 16-pixel minimum"
    assert kept[10:, :].all(), "the large region must be untouched"


def test_a_float_minimum_is_accepted():
    # np.ceil(min_beach_area / pixel_size**2) is a float, and remove_small_objects
    # rejects a non-integer min_size
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, np.ceil(4500 / 10**2))  # 45 pixels

    assert not kept[2:5, 2:5].any()
    assert kept[10:, :].all()


def test_extract_shorelines_passes_pixels_to_the_sar_otsu_path():
    # guards against settings['min_beach_area'] (m^2) being passed straight through again
    source = inspect.getsource(SDS_shoreline.extract_shorelines)

    call = re.search(r"otsu_threshold\(([^)]*)\)", source)

    assert call is not None, "extract_shorelines no longer calls otsu_threshold"
    assert "min_beach_area_pixels" in call.group(1)
    assert 'settings["min_beach_area"]' not in call.group(1)


def test_the_conversion_matters_at_the_s1_pixel_size():
    # the bug was invisible in review because both values are "min beach area"; at 10 m
    # they differ by 100x, which is the whole point of converting
    pixel_size = get_pixel_size_for_satellite("S1")
    min_beach_area = 4500

    assert np.ceil(min_beach_area / pixel_size**2) == 45
    assert min_beach_area / np.ceil(min_beach_area / pixel_size**2) == 100


# =================================================================================== #
# Tests that S1 scenes are segmented with the ONNX model, falling back to Otsu
#
# The model is the default, but it needs onnxruntime, the .onnx file and a dual-polarization
# scene. Whenever any of those is missing the run must still map a shoreline with the legacy
# Otsu threshold rather than dropping the image.
# =================================================================================== #


def model_composite(height=SIZE, width=SIZE):
    im_vv = np.full((height, width), -12.0)
    im_vh = np.full((height, width), -20.0)
    return np.stack([im_vv, im_vh, im_vv - im_vh], axis=2)


@pytest.fixture(autouse=True)
def clear_cache():
    SDS_sar_model.clear_model_cache()
    yield
    SDS_sar_model.clear_model_cache()


# --------------------------------------------------------------- load_sar_segmenter


def test_loads_the_model_by_default(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: loaded.append(path) or ("session", {"channel_order": ["VV", "VH", "VV-VH"]}),
    )

    segmenter = load_sar_segmenter({}, logger=RecordingLogger())

    assert segmenter is not None
    # None means "the default model", which authorizes load_sar_model to download it
    assert loaded == [None]


def test_sar_segmentation_otsu_skips_the_model_entirely(monkeypatch):
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: pytest.fail("the model must not be loaded when Otsu is requested"),
    )
    logger = RecordingLogger()

    assert load_sar_segmenter({"sar_segmentation": "otsu"}, logger=logger) is None
    assert any("otsu" in message.lower() for message in logger.info_messages)


def test_falls_back_when_onnxruntime_is_missing(monkeypatch):
    def unavailable(_path):
        raise SarModelUnavailable("onnxruntime is not installed")

    monkeypatch.setattr(SDS_sar_model, "load_sar_model", unavailable)
    logger = RecordingLogger()

    assert load_sar_segmenter({}, logger=logger) is None
    assert any("onnxruntime" in warning for warning in logger.warnings)


def test_falls_back_when_the_model_file_is_missing(tmp_path):
    logger = RecordingLogger()
    settings = {"sar_model_path": str(tmp_path / "missing.onnx")}

    assert load_sar_segmenter(settings, logger=logger) is None
    assert any("not found" in warning for warning in logger.warnings)


def test_override_paths_are_never_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SDS_sar_model,
        "fetch_default_sar_model",
        lambda: pytest.fail("an explicit sar_model_path must never trigger a download"),
    )
    logger = RecordingLogger()
    settings = {"sar_model_path": str(tmp_path / "missing.onnx")}

    assert load_sar_segmenter(settings, logger=logger) is None
    assert any("not found" in warning for warning in logger.warnings)


def test_settings_can_point_at_another_model(monkeypatch, tmp_path):
    requested = []
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: requested.append(path) or ("session", {"channel_order": []}),
    )

    load_sar_segmenter({"sar_model_path": str(tmp_path / "other.onnx")})

    assert Path(requested[0]).name == "other.onnx"


# --------------------------------------------------------------- segment_sar_water


def test_returns_none_without_a_segmenter():
    assert segment_sar_water(model_composite(), DUAL_POL, None, {}) is None


def test_returns_the_water_validity_and_probability_masks(monkeypatch):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    water = np.zeros((SIZE, SIZE), dtype=bool)
    water[4:, :] = True
    prob = np.where(water, 0.9, 0.1).astype(np.float32)
    monkeypatch.setattr(SDS_shoreline.SDS_tools, "read_sar_valid_mask", lambda fn: valid)
    monkeypatch.setattr(
        SDS_sar_model, "segment_water", lambda *args, **kwargs: (water, prob)
    )

    im_water, returned_valid, im_prob = segment_sar_water(
        model_composite(), DUAL_POL, ("session", {}), {}
    )

    assert np.array_equal(im_water, water)
    assert np.array_equal(returned_valid, valid)
    # the probability the mask was cut from, for save_sar_probability
    assert np.array_equal(im_prob, prob)


def test_falls_back_for_a_single_polarization_scene(monkeypatch):
    # a legacy VH-only scene is (H, W, 1), which the model cannot take
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )
    logger = RecordingLogger()
    im_ms = np.full((SIZE, SIZE, 1), -20.0)
    spec = SDS_sar_model.read_model_spec(_SpecOnlySession())

    result = segment_sar_water(
        im_ms, [s1_path("VH")], (None, spec), {}, logger=logger, date_str=DATE
    )

    assert result is None
    assert any("missing VV or VH" in warning for warning in logger.warnings)


def test_falls_back_when_inference_fails(monkeypatch):
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )

    def boom(*args, **kwargs):
        raise SarModelUnavailable("SAR model inference failed: out of memory")

    monkeypatch.setattr(SDS_sar_model, "segment_water", boom)
    logger = RecordingLogger()

    result = segment_sar_water(
        model_composite(), DUAL_POL, ("session", {}), {}, logger=logger, date_str=DATE
    )

    assert result is None
    assert any("out of memory" in warning for warning in logger.warnings)


def test_the_water_threshold_setting_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )
    thresholds = []
    monkeypatch.setattr(
        SDS_sar_model,
        "segment_water",
        lambda im, valid, session, spec, threshold: thresholds.append(threshold)
        or (np.zeros((SIZE, SIZE), dtype=bool), None),
    )

    segment_sar_water(model_composite(), DUAL_POL, ("session", {}), {})
    segment_sar_water(
        model_composite(), DUAL_POL, ("session", {}), {"sar_water_threshold": 0.8}
    )

    assert thresholds == [SDS_sar_model.WATER_THRESHOLD, 0.8]


class _SpecOnlySession:
    """Minimal stand-in used only to build a real spec dict.

    It has to carry the metadata a real export carries: read_model_spec rejects a model
    that does not declare the constants it was trained with.
    """

    class _Meta:
        custom_metadata_map = {
            "channel_order": '["VV", "VH", "VV-VH"]',
            "normalization_mean": "[-12.59, -20.26, 10.5465]",
            "normalization_std": "[5.26, 5.91, 7.6855]",
        }

    def get_modelmeta(self):
        return self._Meta()


# --------------------------------------------------------------- shoreline search area


def test_invalid_pixels_are_excluded_from_the_contour_search():
    # find_shoreline_SAR NaNs out everything outside the mask it is given, and
    # process_contours then drops the NaN points, so nodata cannot make a shoreline
    water = np.zeros((10, 10), dtype=bool)
    water[5:, :] = True
    buffer = np.ones((10, 10), dtype=bool)
    valid = np.ones((10, 10), dtype=bool)
    valid[:, 8:] = False
    settings = {"output_epsg": 4326, "min_length_sl": 0}
    georef = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

    with_all = SDS_shoreline.find_shoreline_SAR(water, buffer, georef, 4326, settings)
    with_valid = SDS_shoreline.find_shoreline_SAR(
        water, buffer & valid, georef, 4326, settings
    )

    assert len(with_valid) < len(with_all)


# =================================================================================== #
# Tests that P(water) of a model-segmented S1 scene is written as a GeoTIFF
#
# segment_water always returned the probability but the detection branch used to discard
# it. save_sar_probability writes it to S1/prob/, on the grid of the scene's first
# polarization file, with invalid pixels as NaN. Losing the raster must never cost the
# scene its shoreline, so failures are logged and swallowed.
# =================================================================================== #


PROBABILITY_GEOTRANSFORM = (500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0)


def write_scene_tif(path, size=SIZE):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(PROBABILITY_GEOTRANSFORM)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32756)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), -20.0))
    dataset = None
    return str(path)


@pytest.fixture
def scene(tmp_path):
    """A dual-polarization scene layout: S1/VV and S1/VH each holding one .tif."""
    fn = [
        write_scene_tif(tmp_path / "S1" / polar / f"{DATE}_S1_site_{polar}.tif")
        for polar in ("VV", "VH")
    ]
    return fn


def probability():
    prob = np.linspace(0.0, 1.0, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    return prob


# ------------------------------------------------------------------ the happy path


def test_writes_the_probability_next_to_the_polarization_folders(scene, tmp_path):
    valid = np.ones((SIZE, SIZE), dtype=bool)

    path = save_sar_probability(probability(), valid, scene)

    assert path is not None
    assert Path(path) == tmp_path / "S1" / "prob" / f"{DATE}_S1_site_prob.tif"
    assert Path(path).exists()


def test_values_and_dtype_survive_the_round_trip(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    prob = probability()

    path = save_sar_probability(prob, valid, scene)

    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1).ReadAsArray()
    assert band.dtype == np.float32
    np.testing.assert_allclose(band, prob, rtol=1e-6)


def test_invalid_pixels_are_nan_and_nodata_is_declared(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    valid[:, :2] = False

    path = save_sar_probability(probability(), valid, scene)

    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    assert np.all(np.isnan(data[:, :2]))
    assert np.all(np.isfinite(data[:, 2:]))
    # NaN, not 0.0: a probability can genuinely be 0.0
    assert np.isnan(band.GetNoDataValue())


def test_the_grid_is_copied_from_the_scene(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)

    path = save_sar_probability(probability(), valid, scene)

    dataset = gdal.Open(path)
    assert dataset.GetGeoTransform() == PROBABILITY_GEOTRANSFORM
    assert "32756" in dataset.GetProjection()


def test_the_dup_suffix_is_kept(tmp_path):
    # both polarizations of a duplicate scene share a _dupN token, which is what
    # pairs them; the probability file has to carry it too
    fn = [
        write_scene_tif(tmp_path / "S1" / polar / f"{DATE}_S1_site_{polar}_dup0.tif")
        for polar in ("VV", "VH")
    ]

    path = save_sar_probability(probability(), np.ones((SIZE, SIZE), dtype=bool), fn)

    assert Path(path).name == f"{DATE}_S1_site_prob_dup0.tif"


def test_probability_accepts_a_bare_string_path(tmp_path):
    fn = write_scene_tif(tmp_path / "S1" / "VH" / f"{DATE}_S1_site_VH.tif")

    path = save_sar_probability(probability(), np.ones((SIZE, SIZE), dtype=bool), fn)

    assert Path(path).exists()


# ------------------------------------------------------------------ failure handling


def test_an_unreadable_scene_is_logged_and_swallowed(tmp_path):
    logger = RecordingLogger()
    missing = str(tmp_path / "S1" / "VV" / "gone.tif")

    path = save_sar_probability(
        probability(),
        np.ones((SIZE, SIZE), dtype=bool),
        [missing],
        logger=logger,
        date_str=DATE,
    )

    assert path is None
    assert any(DATE in warning for warning in logger.warnings)


# ------------------------------------------------------------- compute_sar_water_fraction


def test_water_fraction_is_computed_over_the_region():
    water = np.zeros((4, 4), dtype=bool)
    water[2:, :] = True
    region = np.ones((4, 4), dtype=bool)

    assert compute_sar_water_fraction(water, region) == pytest.approx(0.5)


def test_water_fraction_ignores_pixels_outside_the_region():
    water = np.zeros((4, 4), dtype=bool)
    water[2:, :] = True
    region = np.zeros((4, 4), dtype=bool)
    region[2:, :] = True  # only water pixels are in the region

    assert compute_sar_water_fraction(water, region) == pytest.approx(1.0)


def test_water_fraction_of_an_empty_region_is_nan():
    # an empty buffer & valid intersection must not divide by zero
    water = np.zeros((4, 4), dtype=bool)
    region = np.zeros((4, 4), dtype=bool)

    assert np.isnan(compute_sar_water_fraction(water, region))


# =================================================================================== #
# Tests for the SAR jpg preview: a titled false-colour composite of VV, VH and VV-VH
# =================================================================================== #


PREVIEW_SIZE = 12


def gradient(low, high):
    """A band with a real spread, so the percentile stretch has something to work on."""
    values = np.linspace(low, high, PREVIEW_SIZE * PREVIEW_SIZE)
    return values.reshape(PREVIEW_SIZE, PREVIEW_SIZE).astype("float64")


def preview_composite():
    """A [VV, VH, VV-VH] composite with three genuinely different dB ranges."""
    vv = gradient(-25.0, -2.0)
    vh = gradient(-32.0, -12.0)
    return np.dstack([vv, vh, vv - vh])


# ------------------------------------------------------------------ the displayed array


def test_composite_is_shown_as_independently_stretched_rgb():
    im_composite = preview_composite()

    im_display, _ = build_sar_preview(im_composite, DATE, "S1")

    assert im_display.shape == (PREVIEW_SIZE, PREVIEW_SIZE, 3)
    for channel in range(3):
        band = im_display[:, :, channel]
        # VV-VH spans a completely different dB range; a shared stretch would flatten it
        assert band.min() == pytest.approx(0.0, abs=1e-6)
        assert band.max() == pytest.approx(1.0, abs=1e-6)
        # the stretch is monotonic, so the ordering of pixels must be preserved
        assert np.array_equal(
            np.argsort(band, axis=None),
            np.argsort(im_composite[:, :, channel], axis=None),
        )


@pytest.mark.parametrize("ndim", [3, 2], ids=["single-channel-stack", "two-dimensional"])
def test_single_polarization_is_shown_as_greyscale(ndim):
    im = gradient(-32.0, -12.0)

    im_display, _ = build_sar_preview(im[:, :, None] if ndim == 3 else im, DATE, "S1")

    assert im_display.ndim == 2


def test_flat_band_does_not_divide_by_zero():
    flat = np.full((PREVIEW_SIZE, PREVIEW_SIZE), -20.0)

    im_display, _ = build_sar_preview(np.dstack([flat, flat, flat - flat]), DATE, "S1")

    assert np.all(np.isfinite(im_display))


def test_nan_pixels_do_not_propagate():
    im_composite = preview_composite()
    im_composite[0, 0, :] = np.nan

    im_display, _ = build_sar_preview(im_composite, DATE, "S1")

    assert np.all(np.isfinite(im_display))


# ------------------------------------------------------------------ the title


def test_composite_title_names_the_bands_their_channels_the_date_and_satellite():
    _, title = build_sar_preview(preview_composite(), DATE, "S1")

    assert "composite" in title.lower()
    for band in SDS_tools.SAR_COMPOSITE_BANDS:
        assert band in title
    assert "R=VV" in title
    assert "G=VH" in title
    assert "B=VV-VH" in title
    assert DATE in title
    assert "S1" in title


def test_single_polarization_title_makes_no_composite_claim():
    _, title = build_sar_preview(gradient(-32.0, -12.0), DATE, "S1")

    assert "composite" not in title.lower()
    assert DATE in title


# ------------------------------------------------------------------ the saved file


@pytest.mark.parametrize(
    "make_image",
    [preview_composite, lambda: gradient(-32.0, -12.0)[:, :, None]],
    ids=["composite", "single-polarization"],
)
def test_saves_a_jpg_into_the_rgb_folder(tmp_path, make_image):
    save_sar_image(make_image(), DATE, "S1", str(tmp_path))

    assert (tmp_path / "RGB" / f"{DATE}_RGB_S1.jpg").exists()


def test_rejects_an_unexpected_shape(tmp_path):
    with pytest.raises(ValueError):
        save_sar_image(np.zeros((2, 2, 2, 2)), DATE, "S1", str(tmp_path))


# =================================================================================== #
# Tests that SAR jpg previews are written from the images on disk
#
# The S1 branch of save_jpg used to be handed a bare filename rather than a full path,
# so it only worked when the current directory happened to be the .tif folder.
# =================================================================================== #


def jpg_write_tif(path, value, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    # a gradient, so rescale_intensity has a range to work with
    array = np.linspace(value, value + 15.0, size * size).reshape(size, size)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


def jpg_make_session(root, polarizations):
    for polar in polarizations:
        jpg_write_tif(
            root / SITENAME / "S1" / polar / f"{DATE}_S1_{SITENAME}_{polar}.tif",
            value=-30.0 if polar == "VV" else -25.0,
        )
    metadata = {
        "S1": {"filenames": [f"{DATE}_S1_{SITENAME}_{polarizations[0]}.tif"]}
    }
    settings = {
        "inputs": {"filepath": str(root), "sitename": SITENAME, "landsat_collection": "C02"},
        "cloud_thresh": 0.5,
        "cloud_mask_issue": False,
        "pan_off": False,
    }
    return metadata, settings


def preview_path(root):
    return root / SITENAME / "jpg_files" / "preprocessed" / "RGB" / f"{DATE}_RGB_S1.jpg"


def test_writes_a_preview_for_a_dual_polarization_scene(tmp_path, monkeypatch):
    # run from somewhere else entirely: a bare filename would not resolve
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = jpg_make_session(tmp_path, ["VV", "VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_writes_a_preview_for_a_legacy_vh_only_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = jpg_make_session(tmp_path, ["VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_writes_a_preview_for_a_duplicate_named_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = jpg_make_session(tmp_path, ["VV", "VH"])
    for polar in ("VV", "VH"):
        folder = tmp_path / SITENAME / "S1" / polar
        (folder / f"{DATE}_S1_{SITENAME}_{polar}.tif").rename(
            folder / f"{DATE}_S1_{SITENAME}_{polar}_dup0.tif"
        )
    metadata["S1"]["filenames"] = [f"{DATE}_S1_{SITENAME}_VV_dup0.tif"]

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_no_composite_tif_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = jpg_make_session(tmp_path, ["VV", "VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    s1_dir = tmp_path / SITENAME / "S1"
    assert sorted(p.name for p in s1_dir.iterdir()) == ["VH", "VV"]


# =================================================================================== #
# Tests that the SAR detection figure handles any number of polarization channels
#
# matplotlib's imshow only accepts 1, 3 or 4 trailing channels, so a 2-channel VV+VH
# array has to be reduced to one band before it is displayed.
# =================================================================================== #


@pytest.fixture
def detection_args():
    height, width = 12, 10
    return {
        "im_labels": np.zeros((height, width), dtype=bool),
        "sl_pix": np.array([[1.0, 1.0], [2.0, 2.0]]),
        "date": DATE,
        "satname": "S1",
        "im_ref_buffer": np.ones((height, width), dtype=bool),
        "shoreline_extraction_area": [],
    }


def sar_image(n_channels, height=12, width=10):
    """A SAR image whose channels differ, so the displayed one is identifiable."""
    channels = [
        np.linspace(-30 - 10 * c, -5 - 10 * c, height * width).reshape(height, width)
        for c in range(n_channels)
    ]
    return np.stack(channels, axis=2)


@pytest.mark.parametrize("n_channels", [1, 2, 3, 4])
def test_handles_any_number_of_channels(tmp_path, detection_args, n_channels):
    # 2 channels is VV+VH, 1 is a legacy VH-only scene, 3 is the pre-change array shape
    SDS_plotting.plot_sar_detection(
        sar_image(n_channels),
        output_path=str(tmp_path),
        settings={"save_figure": True},
        **detection_args,
    )

    assert (tmp_path / f"{DATE}_S1.jpg").exists()


def test_handles_a_two_dimensional_image(tmp_path, detection_args):
    SDS_plotting.plot_sar_detection(
        sar_image(1)[:, :, 0],
        output_path=str(tmp_path),
        settings={"save_figure": True},
        **detection_args,
    )

    assert (tmp_path / f"{DATE}_S1.jpg").exists()


@pytest.mark.parametrize(
    "kwargs,channel", [({}, 0), ({"band_index": 1}, 1)], ids=["default", "band_index"]
)
def test_displays_the_requested_channel(tmp_path, detection_args, monkeypatch, kwargs, channel):
    im_ms = sar_image(2)
    displayed = []
    monkeypatch.setattr(
        SDS_plotting,
        "_normalize_grayscale",
        lambda im: displayed.append(im) or np.zeros_like(im),
    )

    SDS_plotting.plot_sar_detection(
        im_ms,
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **kwargs,
        **detection_args,
    )

    assert len(displayed) == 1
    assert np.array_equal(displayed[0], im_ms[:, :, channel])


@pytest.mark.parametrize("covers_everything", [True, False], ids=["all-true", "partial"])
def test_buffer_is_drawn_only_when_it_constrains_the_search(
    tmp_path, detection_args, monkeypatch, covers_everything
):
    # a buffer covering the whole image means no reference shoreline was given, so
    # tinting the entire figure with it would only obscure the imagery
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    if covers_everything:
        buffer[:] = True
    detection_args["im_ref_buffer"] = buffer
    overlays = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "imshow", lambda self, im, **kw: overlays.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(2),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    drawn = any(kw.get("cmap") == "PiYG" for kw in overlays)
    assert drawn != covers_everything


def axis_calls(monkeypatch):
    """Record which axis every imshow lands on, so the panels can be told apart."""
    calls = []

    def record(self, im, **kw):
        calls.append((self, im, kw))

    monkeypatch.setattr(SDS_plotting.plt.Axes, "imshow", record)
    return calls


def test_draws_imagery_segmentation_and_buffer_panels(tmp_path, detection_args, monkeypatch):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    assert len(axes) == 3, "expected imagery, segmentation and buffer panels"
    # the middle panel has to stay readable, so the buffer goes on its own copy
    buffer_axes = {axis for axis, _, kw in calls if kw.get("cmap") == "PiYG"}
    assert buffer_axes == {axes[2]}, "the buffer must appear only on the third panel"
    # the class overlay is the one drawn with alpha=0.3
    class_axes = {axis for axis, _, kw in calls if kw.get("alpha") == 0.3}
    assert class_axes == {axes[1], axes[2]}, "the first panel shows the imagery only"


def test_three_panels_without_a_reference_shoreline(tmp_path, detection_args, monkeypatch):
    # an all-True buffer constrains nothing; the panel still appears but is not tinted
    detection_args["im_ref_buffer"] = np.ones((12, 10), dtype=bool)
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    assert len(axes) == 3
    assert not any(kw.get("cmap") == "PiYG" for _, _, kw in calls)


def test_buffer_appears_in_the_legend(tmp_path, detection_args, monkeypatch):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    legends = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: legends.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    labels = [handle.get_label() for handle in legends[0]["handles"]]
    assert "reference shoreline buffer" in labels


def test_the_buffer_swatch_matches_the_colour_actually_drawn():
    # the buffer is an all-True masked array under PiYG, which normalises to 0.0 and so
    # takes the PINK end of the colormap. A swatch showing the green end would send the
    # reader looking for something that is not on the image.
    buffer = np.zeros((6, 6), dtype=bool)
    buffer[2:4, :] = True
    masked = np.ma.masked_where(buffer == False, buffer)

    figure, axis = SDS_plotting.plt.subplots()
    image = axis.imshow(masked, cmap="PiYG", alpha=SDS_plotting.REF_BUFFER_ALPHA)
    drawn = matplotlib.colors.to_hex(image.cmap(image.norm(1.0)))
    SDS_plotting.plt.close(figure)

    assert drawn == SDS_plotting.REF_BUFFER_COLOR


def test_class_labels_appear_in_the_legend(tmp_path, detection_args, monkeypatch):
    # the two segmentation methods label opposite classes: Otsu marks the
    # high-backscatter class (land) as 1, the model marks water as 1
    figures = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: figures.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        class_labels=("land", "water"),
        **detection_args,
    )

    labels = [handle.get_label() for handle in figures[0]["handles"]]
    assert "land" in labels and "water" in labels
    assert "Otsu class 1" not in labels


def test_legend_defaults_to_the_otsu_labels(tmp_path, detection_args, monkeypatch):
    figures = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: figures.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    labels = [handle.get_label() for handle in figures[0]["handles"]]
    assert "Otsu class 1" in labels and "Otsu class 2" in labels
