import os
import pytest
from coastsat.SDS_tools import get_filenames


# @pytest.fixture
# def temp_dir(tmpdir):
#     # Create a temporary directory with sample files
#     temp_dir = tmpdir.mkdir("satellite_images")
#     temp_dir.join("sample_ms.tif").write("sample data")
#     temp_dir.join("sample_pan.tif").write("sample data")
#     temp_dir.join("sample_mask.tif").write("sample data")
#     return temp_dir


@pytest.fixture
def temp_dir(tmpdir):
    # Create a temporary directory with sample files
    temp_dir = tmpdir.mkdir("satellite_images")
    temp_dir.join("sample_ms.tif").write("sample data")
    temp_dir.join("sample_pan.tif").write("sample data")
    temp_dir.join("sample_mask.tif").write("sample data")
    temp_dir.join("sample_swir.tif").write("sample data")
    return temp_dir


def test_get_filenames1(temp_dir):
    # Test the get_filenames function with sample data
    filename = "sample_ms.tif"
    filepath = [str(temp_dir), str(temp_dir), str(temp_dir)]
    satname = "L8"

    expected_filenames = [
        os.path.join(str(temp_dir), "sample_ms.tif"),
        os.path.join(str(temp_dir), "sample_pan.tif"),
        os.path.join(str(temp_dir), "sample_mask.tif"),
    ]

    result_filenames = get_filenames(filename, filepath, satname)
    assert (
        result_filenames == expected_filenames
    ), f"Expected {expected_filenames}, but got {result_filenames}"

    # Test with an unrecognized satellite name
    with pytest.raises(ValueError):
        get_filenames(filename, filepath, "INVALID_SATNAME")


def test_get_filenames(temp_dir):
    # Test the get_filenames function with sample data for L5
    filename = "sample_ms.tif"
    filepath = [str(temp_dir), str(temp_dir)]
    satname = "L5"

    expected_filenames_l5 = [
        os.path.join(str(temp_dir), "sample_ms.tif"),
        os.path.join(str(temp_dir), "sample_mask.tif"),
    ]

    result_filenames_l5 = get_filenames(filename, filepath, satname)
    assert (
        result_filenames_l5 == expected_filenames_l5
    ), f"Expected {expected_filenames_l5}, but got {result_filenames_l5}"

    # Test the get_filenames function with sample data for L7, L8, and L9
    filepath = [str(temp_dir), str(temp_dir), str(temp_dir)]
    expected_filenames_l7_l8_l9 = [
        os.path.join(str(temp_dir), "sample_ms.tif"),
        os.path.join(str(temp_dir), "sample_pan.tif"),
        os.path.join(str(temp_dir), "sample_mask.tif"),
    ]

    for satname in ["L7", "L8", "L9"]:
        result_filenames_l7_l8_l9 = get_filenames(filename, filepath, satname)
        assert (
            result_filenames_l7_l8_l9 == expected_filenames_l7_l8_l9
        ), f"Expected {expected_filenames_l7_l8_l9}, but got {result_filenames_l7_l8_l9}"

    # Test the get_filenames function with sample data for S2
    filename = "sample_ms.tif"
    satname = "S2"

    expected_filenames_s2 = [
        os.path.join(str(temp_dir), "sample_ms.tif"),
        os.path.join(str(temp_dir), "sample_swir.tif"),
        os.path.join(str(temp_dir), "sample_mask.tif"),
    ]

    result_filenames_s2 = get_filenames(filename, filepath, satname)
    assert (
        result_filenames_s2 == expected_filenames_s2
    ), f"Expected {expected_filenames_s2}, but got {result_filenames_s2}"

    # Test with an unrecognized satellite name
    with pytest.raises(ValueError):
        get_filenames(filename, filepath, "INVALID_SATNAME")
