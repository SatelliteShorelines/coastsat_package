"""Tests for SDS_shoreline: cloud metrics, the shoreline buffer, extraction, classifiers."""

from datetime import datetime
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_preprocess, SDS_shoreline
from coastsat.SDS_shoreline import compute_cloud_metrics, create_shoreline_buffer


# =================================================================================== #
# compute_cloud_metrics
# =================================================================================== #


def test_compute_cloud_metrics_returns_expected_values():
    cloud_mask = np.array([[True, False], [True, False]])
    im_nodata = np.array([[False, False], [True, False]])

    cloud_cover_combined, cloud_cover, cloud_mask_adv = compute_cloud_metrics(
        cloud_mask=cloud_mask,
        im_nodata=im_nodata,
    )

    expected_cloud_mask_adv = np.array([[True, False], [False, False]])

    assert cloud_cover_combined == pytest.approx(0.5)
    assert cloud_cover == pytest.approx(1 / 3)
    assert np.array_equal(cloud_mask_adv, expected_cloud_mask_adv)


def test_compute_cloud_metrics_raises_for_empty_nodata_list():
    cloud_mask = np.array([[True, False], [False, True]])

    with pytest.raises(ValueError, match="Cloud/no-data mask shape mismatch"):
        compute_cloud_metrics(cloud_mask=cloud_mask, im_nodata=[])


# =================================================================================== #
# Tests for the reference shoreline buffer, including the no-reference-shoreline case
# =================================================================================== #


IM_SHAPE = (40, 30)
PIXEL_SIZE = 10
EPSG = 32756
# a georeference whose world coordinates map 1:1 onto pixels at PIXEL_SIZE metres
GEOREF = np.array([500000.0, PIXEL_SIZE, 0.0, 6000000.0, 0.0, -PIXEL_SIZE])


def world_xy(col, row):
    """World coordinates of a pixel centre, for the georeference above."""
    return [GEOREF[0] + col * PIXEL_SIZE, GEOREF[3] - row * PIXEL_SIZE]


def vertical_reference_line():
    """A reference shoreline running down the middle of the image."""
    return np.array([world_xy(15, 5), world_xy(15, 35)])


def test_no_reference_shoreline_searches_the_whole_image():
    # regression: the dilation used to run outside the reference-shoreline branch,
    # raising UnboundLocalError whenever settings had no 'reference_shoreline'
    settings = {"output_epsg": EPSG, "max_dist_ref": 300}

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.shape == IM_SHAPE
    assert buffer.dtype == bool
    assert buffer.all()


def test_no_reference_shoreline_does_not_need_max_dist_ref():
    # max_dist_ref is only meaningful alongside a reference shoreline
    settings = {"output_epsg": EPSG}

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.all()


def test_reference_shoreline_restricts_the_buffer():
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,  # 3 pixels
        "reference_shoreline": vertical_reference_line(),
    }

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.shape == IM_SHAPE
    assert buffer.any()
    assert not buffer.all()  # a buffer that covers everything would be pointless
    # the line itself is inside the buffer, the far corner is outside
    assert buffer[20, 15]
    assert not buffer[0, 0]


def test_buffer_widens_with_max_dist_ref():
    def buffer_for(max_dist_ref):
        settings = {
            "output_epsg": EPSG,
            "max_dist_ref": max_dist_ref,
            "reference_shoreline": vertical_reference_line(),
        }
        return create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer_for(20).sum() < buffer_for(60).sum()


def test_reference_shoreline_with_three_columns_is_accepted():
    # reference shorelines are often stored as x, y, z
    line = vertical_reference_line()
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,
        "reference_shoreline": np.column_stack([line, np.zeros(len(line))]),
    }

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.any()
    assert not buffer.all()


def two_separate_lines():
    """Two short reference lines at opposite corners, like a shoreline that leaves
    the image and comes back in somewhere else."""
    top_left = np.array([world_xy(3, 3), world_xy(8, 3)])
    bottom_right = np.array([world_xy(22, 36), world_xy(27, 36)])
    return top_left, bottom_right


def test_separate_parts_are_not_joined_by_a_straight_line():
    # regression: the parts used to be stacked into one array, so the end of one
    # line was connected to the start of the next and a stripe was buffered
    top_left, bottom_right = two_separate_lines()
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,  # 3 pixels
        "reference_shoreline": np.vstack([top_left, bottom_right]),
    }

    joined = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)
    settings["reference_shoreline_parts"] = [top_left, bottom_right]
    separate = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    # both lines are buffered, the middle of the image between them is not
    assert separate[3, 5]
    assert separate[36, 25]
    assert not separate[20, 15]
    # without the parts the old single-line behaviour is kept
    assert joined[20, 15]


def test_parts_match_buffering_each_line_on_its_own():
    top_left, bottom_right = two_separate_lines()

    def buffer_for(line, parts=None):
        settings = {"output_epsg": EPSG, "max_dist_ref": 30, "reference_shoreline": line}
        if parts is not None:
            settings["reference_shoreline_parts"] = parts
        return create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    both = buffer_for(np.vstack([top_left, bottom_right]), [top_left, bottom_right])

    assert np.array_equal(both, buffer_for(top_left) | buffer_for(bottom_right))


@pytest.mark.parametrize("parts", [None, []])
def test_missing_or_empty_parts_fall_back_to_the_single_line(parts):
    line = vertical_reference_line()
    settings = {"output_epsg": EPSG, "max_dist_ref": 30, "reference_shoreline": line}
    expected = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    settings["reference_shoreline_parts"] = parts
    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert np.array_equal(buffer, expected)


def test_single_part_matches_the_single_line():
    line = vertical_reference_line()
    settings = {"output_epsg": EPSG, "max_dist_ref": 30, "reference_shoreline": line}
    expected = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    settings["reference_shoreline_parts"] = [line]
    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert np.array_equal(buffer, expected)


def test_part_with_a_single_point_is_skipped():
    # clipping a shoreline to an ROI can leave a part that is a single point
    line = vertical_reference_line()
    settings = {"output_epsg": EPSG, "max_dist_ref": 30, "reference_shoreline": line}
    expected = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    settings["reference_shoreline_parts"] = [line, np.array([world_xy(2, 2)])]
    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert np.array_equal(buffer, expected)


def test_parts_given_as_plain_lists_are_accepted():
    # parts read back from a saved settings json are lists, not arrays
    top_left, bottom_right = two_separate_lines()
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,
        "reference_shoreline": np.vstack([top_left, bottom_right]),
        "reference_shoreline_parts": [top_left, bottom_right],
    }
    expected = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    settings["reference_shoreline_parts"] = [top_left.tolist(), bottom_right.tolist()]
    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert np.array_equal(buffer, expected)


# =================================================================================== #
# extract_shorelines on a scene whose multispectral image is all zeros
# =================================================================================== #


def test_extract_shorelines_skips_all_zero_s2_image_and_continues(
    monkeypatch, tmp_path
):
    logger = MagicMock()
    saved_outputs = []

    filenames = [
        "2024-01-01-00-00-00_S2_mock.tif",
        "2024-01-02-00-00-00_S2_mock.tif",
    ]

    settings = {
        "inputs": {
            "sitename": "unit-test-site",
            "filepath": str(tmp_path),
            "landsat_collection": "C02",
        },
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": 32633,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": False,
        "min_beach_area": 25,
        "min_length_sl": 10,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": False,
    }

    metadata = {
        "S2": {
            "filenames": filenames,
            "dates": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "epsg": [32633, 32633],
            "acc_georef": [5.0, 5.0],
        }
    }

    monkeypatch.setattr(SDS_shoreline, "setup_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(SDS_shoreline, "release_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "get_filepath",
        lambda inputs, satname: str(tmp_path / satname),
    )
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "get_filenames",
        lambda filename, filepath, satname: [
            os.path.join(filepath, filename),
            os.path.join(filepath, f"swir_{filename}"),
            os.path.join(filepath, f"mask_{filename}"),
        ],
    )
    monkeypatch.setattr(SDS_shoreline, "load_classifier", lambda **kwargs: object())
    monkeypatch.setattr(
        SDS_shoreline,
        "create_shoreline_buffer",
        lambda im_shape, georef, image_epsg, pixel_size, settings: np.ones(
            im_shape, dtype=bool
        ),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "classify_image_NN",
        lambda im_ms, cloud_mask, min_beach_area_pixels, clf: (
            np.zeros(cloud_mask.shape, dtype=int),
            np.zeros((*cloud_mask.shape, 3), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "find_shoreline_optical",
        lambda *args, **kwargs: (np.array([[0.0, 0.0], [1.0, 1.0]]), 0.12),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "filter_shoreline",
        lambda shoreline, shoreline_extraction_area, output_epsg: shoreline,
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "get_extract_shoreline_extraction_area_array",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        SDS_preprocess,
        "create_gdf_from_image_extent",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        SDS_preprocess,
        "write_to_json",
        lambda path, output: saved_outputs.append((path, output)),
    )

    def fake_preprocess_image(fn, satname, settings, collection):
        if os.path.basename(fn[0]).startswith("2024-01-01-00-00-00"):
            raise SDS_preprocess.SkipImageError(
                "Skipped image because Sentinel-2 multispectral file contains only zeros"
            )

        im_ms = np.ones((2, 2, 5), dtype=float)
        georef = np.array([0.0, 10.0, 0.0, 0.0, 0.0, -10.0])
        cloud_mask = np.zeros((2, 2), dtype=bool)
        im_nodata = np.zeros((2, 2), dtype=bool)
        return im_ms, georef, cloud_mask, im_nodata

    monkeypatch.setattr(SDS_shoreline, "preprocess_image", fake_preprocess_image)

    output = SDS_shoreline.extract_shorelines(metadata, settings)

    assert output["filename"] == [filenames[1]]
    assert output["dates"] == [metadata["S2"]["dates"][1]]
    assert output["idx"] == [1]
    assert output["satname"] == ["S2"]
    assert len(output["shorelines"]) == 1
    np.testing.assert_array_equal(
        output["shorelines"][0], np.array([[0.0, 0.0], [1.0, 1.0]])
    )
    assert saved_outputs and saved_outputs[0][1] == output
    assert any(
        "Skipped during preprocessing" in call.args[0]
        for call in logger.warning.call_args_list
    )


# =================================================================================== #
# load_classifier fallback for the 'latest' model
# =================================================================================== #


@pytest.fixture
def loaded_paths(monkeypatch):
    """Record the model path passed to joblib.load instead of loading it."""
    paths = []

    def fake_load(path):
        paths.append(Path(path).name)
        return object()

    monkeypatch.setattr(SDS_shoreline.joblib, "load", fake_load)
    return paths


def test_load_classifier_latest_falls_back_to_default_when_missing(
    loaded_paths, tmp_path, capsys
):
    (tmp_path / "NN_4classes_Landsat_new.pkl").touch()

    SDS_shoreline.load_classifier("L8", "latest", str(tmp_path), "1.5.0")

    assert loaded_paths == ["NN_4classes_Landsat_new.pkl"]
    assert "Using the default model" in capsys.readouterr().out


@pytest.mark.parametrize(
    "default_file,sklearn_version",
    [("NN_4classes_Landsat_new.pkl", "1.5.0"), ("NN_4classes_Landsat.pkl", "0.20.3")],
    ids=["new-sklearn", "old-sklearn"],
)
def test_load_classifier_latest_is_used_when_present(
    loaded_paths, tmp_path, default_file, sklearn_version
):
    (tmp_path / default_file).touch()
    (tmp_path / "NN_4classes_Landsat_latest_new.pkl").touch()

    SDS_shoreline.load_classifier("L8", "latest", str(tmp_path), sklearn_version)

    assert loaded_paths == ["NN_4classes_Landsat_latest_new.pkl"]


def test_load_model_latest_falls_back_to_default_when_missing(
    loaded_paths, tmp_path, monkeypatch
):
    monkeypatch.setattr(SDS_shoreline, "get_model_locations", lambda: str(tmp_path))
    monkeypatch.setattr(SDS_shoreline.sklearn, "__version__", "1.5.0")
    (tmp_path / "NN_4classes_Landsat_new.pkl").touch()

    SDS_shoreline.load_model("L8", {"sand_color": "latest"})

    assert loaded_paths == ["NN_4classes_Landsat_new.pkl"]


def test_packaged_latest_model_loads_without_fallback(capsys):
    """The real packaged latest model must load for sand_color 'latest' on Landsat."""
    clf = SDS_shoreline.load_classifier(
        "L8",
        "latest",
        SDS_shoreline.get_model_locations(),
        SDS_shoreline.sklearn.__version__,
    )
    assert hasattr(clf, "predict")
    assert "Using the default model" not in capsys.readouterr().out
