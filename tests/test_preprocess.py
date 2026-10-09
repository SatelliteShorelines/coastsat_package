from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_preprocess, SDS_tools

@pytest.mark.parametrize(
    "value", ["false", "False", " FALSE ", "f", "no", "n", "0", "off", "none", ""]
)
def test_to_bool_false_strings(value):
    assert SDS_tools.to_bool(value) is False


@pytest.mark.parametrize("value", ["true", "True", " TRUE ", "t", "yes", "y", "1", "on"])
def test_to_bool_true_strings(value):
    assert SDS_tools.to_bool(value) is True


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
        (None, False),
        (1, True),
        (0, False),
    ],
)
def test_to_bool_non_strings(value, expected):
    assert SDS_tools.to_bool(value) is expected


def test_to_bool_raises_on_unrecognized_string():
    with pytest.raises(ValueError, match="do_cloud_mask"):
        SDS_tools.to_bool("maybe", "do_cloud_mask")


class _FakeDataset:
    RasterCount = 1

    def __init__(self, band):
        self._band = band

    def GetGeoTransform(self):
        return (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)

    def GetRasterBand(self, k):
        band = self._band

        class _Band:
            def ReadAsArray(self):
                return band

        return _Band()


@pytest.fixture
def mock_landsat(monkeypatch):
    """Mock the file reads so preprocess_single runs on a 20x20 image with a 10x10 cloud."""
    im_QA = np.zeros((20, 20), dtype=np.uint16)
    im_QA[5:15, 5:15] = 8  # bit 3 = cloud
    ms_bands = [np.ones((20, 20), dtype=float) for _ in range(5)]
    im_pan = np.ones((20, 20), dtype=float)

    def fake_read_bands(filename, satname=""):
        return [im_QA] if "mask" in filename else ms_bands

    monkeypatch.setattr(SDS_preprocess, "read_bands", fake_read_bands)
    monkeypatch.setattr(
        SDS_preprocess, "raise_exception_if_file_is_empty", lambda *a, **k: None
    )
    monkeypatch.setattr(
        SDS_preprocess.gdal, "Open", lambda *a, **k: _FakeDataset(im_pan)
    )
    return im_QA == 8


def _cloud_mask(satname, fn, pan_off=False, do_cloud_mask=True):
    return SDS_preprocess.preprocess_single(
        fn, satname, False, pan_off, "C02", do_cloud_mask
    )


L5_FN = ["2020-01-01-00-00-00_L5_ms.tif", "2020-01-01-00-00-00_L5_mask.tif"]
L8_FN = [
    "2020-01-01-00-00-00_L8_ms.tif",
    "2020-01-01-00-00-00_L8_pan.tif",
    "2020-01-01-00-00-00_L8_mask.tif",
]


@pytest.mark.parametrize("value", [False, "false", "False", "FALSE", "0", "no"])
def test_preprocess_single_do_cloud_mask_false_strings(mock_landsat, value):
    cloud_mask = _cloud_mask("L5", L5_FN, do_cloud_mask=value)[2]
    assert not cloud_mask.any()


@pytest.mark.parametrize("value", [True, "true", "True"])
def test_preprocess_single_do_cloud_mask_true_strings(mock_landsat, value):
    cloud_mask = _cloud_mask("L5", L5_FN, do_cloud_mask=value)[2]
    np.testing.assert_array_equal(cloud_mask, mock_landsat)


def test_preprocess_single_pan_off_false_string_still_pansharpens(mock_landsat):
    im_extra = _cloud_mask("L8", L8_FN, pan_off="False")[3]
    assert isinstance(im_extra, np.ndarray)

    im_extra = _cloud_mask("L8", L8_FN, pan_off="True")[3]
    assert im_extra == []


def test_preprocess_single_raises_on_unrecognized_string(mock_landsat):
    with pytest.raises(ValueError, match="do_cloud_mask"):
        _cloud_mask("L5", L5_FN, do_cloud_mask="maybe")


@pytest.mark.parametrize(
    "check_detection, expected_calls", [("False", 0), (False, 0), ("True", 1)]
)
def test_plotting_check_detection_false_string_skips_user_input(
    monkeypatch, check_detection, expected_calls
):
    from coastsat import SDS_plotting

    calls = []

    def fake_waitforbuttonpress():
        calls.append(1)
        raise RuntimeError("stop waiting for user input")

    monkeypatch.setattr(
        SDS_plotting.plt, "waitforbuttonpress", fake_waitforbuttonpress
    )
    fig, ax = SDS_plotting.plt.subplots()
    try:
        SDS_plotting._handle_user_input(
            fig, ax, {"check_detection": check_detection}
        )
    except RuntimeError:
        pass
    finally:
        SDS_plotting.plt.close(fig)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "include_T2, expected", [("False", ["t1"]), (False, ["t1"]), ("True", ["t1", "t2"])]
)
def test_merge_image_tiers_include_T2_strings(include_T2, expected):
    from coastsat import SDS_download

    inputs = {"sat_list": ["L8"], "include_T2": include_T2}
    merged = SDS_download.merge_image_tiers(inputs, {"L8": ["t1"]}, {"L8": ["t2"]})
    assert merged["L8"] == expected
