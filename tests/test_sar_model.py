"""Tests for SDS_sar_model: the ONNX preprocessing/inference contract and the real model."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_sar_model, SDS_tools
from coastsat.SDS_sar_model import (
    build_model_input,
    pad_to_multiple,
    read_model_spec,
    SarModelUnavailable,
    segment_water,
)


# =================================================================================== #
# Tests for the ONNX preprocessing and inference contract in SDS_sar_model
#
# The ONNX graph starts at the normalized input tensor, so every one of these steps lives
# in our code and none of them fails loudly when it is wrong - it just degrades the
# prediction. These tests pin each step of the model's inference guide.
#
# A stub session stands in for onnxruntime so the suite needs neither onnxruntime nor the
# 130 MB model file.
# =================================================================================== #


MEAN = np.array([-12.59, -20.26, 10.5465], dtype=np.float32)
STD = np.array([5.26, 5.91, 7.6855], dtype=np.float32)
STRIDE = 32

# the real size of a downloaded scene: neither dimension is a multiple of 32
REAL_HEIGHT, REAL_WIDTH = 436, 168


class FakeIO:
    def __init__(self, name):
        self.name = name


class FakeModelMeta:
    def __init__(self, metadata):
        self.custom_metadata_map = metadata


class FakeSession:
    """Stands in for onnxruntime.InferenceSession.

    `water_field` is the P(water) the model should "predict"; it is broadcast to the
    padded input size so cropping behaviour is observable.
    """

    def __init__(self, metadata=None, output_names=("logits", "water_prob"), water_field=None):
        self.metadata = DEFAULT_METADATA if metadata is None else metadata
        self.output_names = list(output_names)
        self.water_field = water_field
        self.inputs_seen = []

    def get_modelmeta(self):
        return FakeModelMeta(self.metadata)

    def get_inputs(self):
        return [FakeIO("image")]

    def get_outputs(self):
        return [FakeIO(name) for name in self.output_names]

    def run(self, _output_names, feeds):
        image = feeds["image"]
        self.inputs_seen.append(image)
        _, _, height, width = image.shape

        if self.water_field is None:
            prob = np.zeros((1, 1, height, width), dtype=np.float32)
        else:
            prob = np.zeros((1, 1, height, width), dtype=np.float32)
            field = self.water_field
            prob[0, 0, : field.shape[0], : field.shape[1]] = field

        logits = np.zeros((1, 2, height, width), dtype=np.float32)
        # channel 0 = land, channel 1 = water; make softmax(logits)[:, 1] == prob
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6)))
        logits[:, 1:2] = odds

        by_name = {"logits": logits, "water_prob": prob}
        # an unrecognised output name still returns an array, so that resolving the
        # output by name is what fails rather than this stub
        return [by_name.get(name, prob) for name in self.output_names]


DEFAULT_METADATA = {
    "channel_order": json.dumps(["VV", "VH", "VV-VH"]),
    "normalization_mean": json.dumps([-12.59, -20.26, 10.5465]),
    "normalization_std": json.dumps([5.26, 5.91, 7.6855]),
    "speckle_filter": json.dumps({"type": "none", "window": 7}),
    "output_stride": "32",
    "nodata": "255",
    "input_scale": "dB",
    "classes": json.dumps({"0": "land", "1": "water"}),
}


@pytest.fixture
def spec():
    return read_model_spec(FakeSession())


def composite(height=6, width=6, vv=-12.0, vh=-20.0):
    """A [VV, VH, VV-VH] composite, as SDS_tools.build_sar_composite returns."""
    im_vv = np.full((height, width), vv, dtype="float64")
    im_vh = np.full((height, width), vh, dtype="float64")
    return np.stack([im_vv, im_vh, im_vv - im_vh], axis=2)


def all_valid(im):
    return np.ones(im.shape[:2], dtype=bool)


# ------------------------------------------------------------------------ model spec


def test_reads_the_constants_from_the_embedded_metadata(spec):
    assert list(spec["channel_order"]) == ["VV", "VH", "VV-VH"]
    assert np.allclose(spec["mean"], MEAN)
    assert np.allclose(spec["std"], STD)
    assert spec["output_stride"] == STRIDE
    assert spec["nodata"] == 255


def test_embedded_metadata_overrides_the_fallback_constants():
    # a retrained model must bring its own constants, not silently use ours
    metadata = dict(DEFAULT_METADATA, normalization_mean=json.dumps([-1.0, -2.0, 3.0]))

    spec = read_model_spec(FakeSession(metadata))

    assert np.allclose(spec["mean"], [-1.0, -2.0, 3.0])
    assert not np.allclose(spec["mean"], MEAN)


def test_rejects_a_model_that_carries_no_metadata():
    # substituting the shipped model's constants here cannot be detected at run time:
    # the graph would accept the input and return a plausible but degraded mask
    with pytest.raises(SarModelUnavailable, match="does not declare"):
        read_model_spec(FakeSession(metadata={}))


@pytest.mark.parametrize("key", SDS_sar_model.REQUIRED_METADATA_KEYS)
def test_rejects_a_model_missing_any_constant_it_was_trained_with(key):
    metadata = {k: v for k, v in DEFAULT_METADATA.items() if k != key}

    with pytest.raises(SarModelUnavailable, match=key):
        read_model_spec(FakeSession(metadata))


def test_the_required_keys_are_the_ones_that_fail_silently():
    # channel order and normalization say what the model was trained on; the rest only
    # describe how to run the graph and fail loudly when wrong, so they keep defaults
    assert set(SDS_sar_model.REQUIRED_METADATA_KEYS) == {
        "channel_order",
        "normalization_mean",
        "normalization_std",
    }
    assert not set(SDS_sar_model.DEFAULT_SPEC) & {"channel_order", "mean", "std"}


def test_defaults_still_fill_in_the_keys_that_only_describe_the_graph():
    metadata = {
        key: DEFAULT_METADATA[key] for key in SDS_sar_model.REQUIRED_METADATA_KEYS
    }

    spec = read_model_spec(FakeSession(metadata))

    assert spec["output_stride"] == STRIDE
    assert spec["nodata"] == 255


@pytest.mark.parametrize(
    "override,match",
    [
        ({"normalization_mean": "not a list of numbers"}, "cannot be read as numbers"),
        # feeding this model our composite would silently transpose VH and VV-VH
        ({"channel_order": json.dumps(["VV", "VV-VH", "VH"])}, "expects channels"),
        # the Lee-family filters do not export into the graph, so they would have to be
        # applied here; running unfiltered data through a filtered model is wrong
        ({"speckle_filter": json.dumps({"type": "lee_sigma"})}, "speckle filter"),
        ({"input_scale": "linear"}, "dB"),
        ({"normalization_std": json.dumps([5.26, 0.0, 7.6855])}, "zero normalization std"),
    ],
    ids=[
        "non-numeric-normalization",
        "reordered-channels",
        "speckle-filter",
        "linear-input",
        "zero-std",
    ],
)
def test_rejects_a_model_this_pipeline_cannot_feed(override, match):
    metadata = dict(DEFAULT_METADATA, **override)

    with pytest.raises(SarModelUnavailable, match=match):
        read_model_spec(FakeSession(metadata))


def test_channel_order_must_match_the_composite_bands(spec):
    assert list(spec["channel_order"]) == list(SDS_tools.SAR_COMPOSITE_BANDS)


# ------------------------------------------------------------------------ padding


@pytest.mark.parametrize(
    "height,width",
    [(REAL_HEIGHT, REAL_WIDTH), (32, 32), (1, 1), (33, 64)],
)
def test_padded_dimensions_are_multiples_of_the_stride(height, width):
    padded = pad_to_multiple(np.zeros((3, height, width), dtype="float32"), STRIDE)

    assert padded.shape[1] % STRIDE == 0
    assert padded.shape[2] % STRIDE == 0
    assert padded.shape[1] >= height and padded.shape[2] >= width


def test_the_real_scene_size_pads_to_the_next_multiple():
    padded = pad_to_multiple(np.zeros((3, REAL_HEIGHT, REAL_WIDTH), "float32"), STRIDE)

    assert padded.shape == (3, 448, 192)


def test_padding_goes_on_the_bottom_and_right_with_zeros():
    image = np.ones((3, 33, 33), dtype="float32")

    padded = pad_to_multiple(image, STRIDE)

    assert padded.shape == (3, 64, 64)
    # the original content stays in the top-left corner, untouched
    assert np.all(padded[:, :33, :33] == 1.0)
    assert np.all(padded[:, 33:, :] == 0.0)
    assert np.all(padded[:, :, 33:] == 0.0)


def test_an_already_aligned_image_is_not_padded():
    image = np.ones((3, 64, 32), dtype="float32")

    assert pad_to_multiple(image, STRIDE).shape == image.shape


# ------------------------------------------------------------------------ model input


def test_input_tensor_is_float32_nchw(spec):
    im = composite(REAL_HEIGHT, REAL_WIDTH)

    image, size = build_model_input(im, all_valid(im), spec)

    assert image.dtype == np.float32
    assert image.shape == (1, 3, 448, 192)
    assert size == (REAL_HEIGHT, REAL_WIDTH)


def test_normalization_is_per_channel(spec):
    im = composite(vv=-12.0, vh=-20.0)  # VV-VH = 8.0

    image, (height, width) = build_model_input(im, all_valid(im), spec)

    expected = (np.array([-12.0, -20.0, 8.0], dtype="float32") - MEAN) / STD
    for channel in range(3):
        assert np.allclose(image[0, channel, :height, :width], expected[channel])


def test_invalid_pixels_are_exactly_zero_after_normalization(spec):
    # filling with the channel mean puts them at a neutral 0, not an extreme outlier
    im = composite()
    valid = all_valid(im)
    valid[2, 3] = False

    image, _ = build_model_input(im, valid, spec)

    assert np.allclose(image[0, :, 2, 3], 0.0)
    assert not np.allclose(image[0, :, 0, 0], 0.0)


def test_a_nan_at_an_invalid_pixel_does_not_leak_into_the_input(spec):
    im = composite()
    im[2, 3, :] = np.nan
    valid = all_valid(im)
    valid[2, 3] = False

    image, _ = build_model_input(im, valid, spec)

    assert np.all(np.isfinite(image))


def test_the_difference_band_is_the_raw_db_difference(spec):
    # VV-VH is derived before the invalid-pixel fill, so no fill value leaks into it
    im = composite(vv=-8.0, vh=-22.0)

    image, (height, width) = build_model_input(im, all_valid(im), spec)

    assert np.allclose(image[0, 2, :height, :width], (14.0 - MEAN[2]) / STD[2])


def test_the_source_composite_is_not_modified(spec):
    im = composite()
    original = im.copy()
    valid = all_valid(im)
    valid[0, 0] = False

    build_model_input(im, valid, spec)

    assert np.array_equal(im, original)


def test_rejects_a_single_polarization_scene(spec):
    # a legacy VH-only scene arrives as (H, W, 1) and cannot be segmented
    im = np.full((6, 6, 1), -20.0)

    with pytest.raises(SarModelUnavailable, match="missing VV or VH"):
        build_model_input(im, all_valid(im), spec)


def test_rejects_a_mismatched_validity_mask(spec):
    im = composite(6, 6)

    with pytest.raises(SarModelUnavailable, match="does not match"):
        build_model_input(im, np.ones((4, 4), dtype=bool), spec)


# ------------------------------------------------------------------------ inference


def test_output_is_cropped_back_to_the_original_size(spec):
    im = composite(REAL_HEIGHT, REAL_WIDTH)
    session = FakeSession()

    im_water, prob = segment_water(im, all_valid(im), session, spec)

    assert im_water.shape == (REAL_HEIGHT, REAL_WIDTH)
    assert prob.shape == (REAL_HEIGHT, REAL_WIDTH)


def test_never_resizes(spec):
    # the model was trained at the native ~10 m GSD; rescaling degrades accuracy
    im = composite(REAL_HEIGHT, REAL_WIDTH)
    session = FakeSession()

    im_water, _ = segment_water(im, all_valid(im), session, spec)

    assert im_water.shape == im.shape[:2]
    assert session.inputs_seen[0].shape[2:] == (448, 192)  # padded, not resized


def test_the_padded_region_is_cropped_off_not_thresholded(spec):
    im = composite(33, 33)
    # the stub predicts water everywhere in the padded tensor
    session = FakeSession(water_field=np.ones((64, 64), dtype="float32"))

    im_water, _ = segment_water(im, all_valid(im), session, spec)

    assert im_water.shape == (33, 33)
    assert np.all(im_water)


@pytest.mark.parametrize(
    "probability,expected", [(0.0, False), (0.49, False), (0.5, True), (1.0, True)]
)
def test_threshold_is_inclusive_at_the_cutoff(spec, probability, expected):
    im = composite(6, 6)
    session = FakeSession(water_field=np.full((32, 32), probability, dtype="float32"))

    im_water, _ = segment_water(im, all_valid(im), session, spec)

    assert bool(im_water[0, 0]) is expected


def test_a_custom_threshold_is_applied(spec):
    im = composite(6, 6)
    session = FakeSession(water_field=np.full((32, 32), 0.6, dtype="float32"))

    assert np.all(segment_water(im, all_valid(im), session, spec, 0.5)[0])
    assert not np.any(segment_water(im, all_valid(im), session, spec, 0.9)[0])


def test_nodata_is_stamped_back_onto_the_outputs(spec):
    # a blank input area must never be mistaken for a land or water call
    im = composite(6, 6)
    valid = all_valid(im)
    valid[1, 1] = False
    session = FakeSession(water_field=np.ones((32, 32), dtype="float32"))

    im_water, prob = segment_water(im, valid, session, spec)

    assert not im_water[1, 1]
    assert np.isnan(prob[1, 1])
    assert im_water[0, 0]
    assert prob[0, 0] == pytest.approx(1.0)


def test_water_prob_is_selected_by_name_not_position(spec):
    # a re-export that reorders the outputs must not silently return the logits
    im = composite(6, 6)
    field = np.full((32, 32), 0.9, dtype="float32")
    forward = FakeSession(output_names=("logits", "water_prob"), water_field=field)
    reversed_ = FakeSession(output_names=("water_prob", "logits"), water_field=field)

    _, prob_forward = segment_water(im, all_valid(im), forward, spec)
    _, prob_reversed = segment_water(im, all_valid(im), reversed_, spec)

    assert np.allclose(prob_forward, prob_reversed)
    assert np.allclose(prob_forward, 0.9)


def test_logits_only_model_is_softmaxed(spec):
    im = composite(6, 6)
    session = FakeSession(
        output_names=("logits",), water_field=np.full((32, 32), 0.8, dtype="float32")
    )

    _, prob = segment_water(im, all_valid(im), session, spec)

    assert np.allclose(prob, 0.8, atol=1e-4)


def test_rejects_a_model_with_no_usable_output(spec):
    im = composite(6, 6)
    session = FakeSession(output_names=("something_else",))

    with pytest.raises(SarModelUnavailable, match="none of which"):
        segment_water(im, all_valid(im), session, spec)


# ------------------------------------------------------------------------ loading


def test_missing_model_file_raises_sar_model_unavailable(tmp_path):
    SDS_sar_model.clear_model_cache()

    with pytest.raises(SarModelUnavailable, match="not found"):
        SDS_sar_model.load_sar_model(str(tmp_path / "missing.onnx"))


def test_settings_can_override_the_model_path(tmp_path):
    custom = tmp_path / "custom.onnx"

    path = SDS_sar_model.get_sar_model_path({"sar_model_path": str(custom)})

    assert Path(path) == custom.resolve()


def test_the_bundled_model_wins_when_present(monkeypatch, tmp_path):
    import coastsat.SDS_shoreline

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bundled = models_dir / SDS_sar_model.DEFAULT_SAR_MODEL_FILENAME
    bundled.write_bytes(b"stub")
    monkeypatch.setattr(
        coastsat.SDS_shoreline, "get_model_locations", lambda: str(models_dir)
    )

    assert Path(SDS_sar_model.get_sar_model_path({})) == bundled


def test_the_default_falls_back_to_the_download_cache(monkeypatch, tmp_path):
    import coastsat.SDS_shoreline

    # an empty models dir, like an installed wheel: the .onnx is not shipped
    monkeypatch.setattr(
        coastsat.SDS_shoreline, "get_model_locations", lambda: str(tmp_path)
    )

    path = Path(SDS_sar_model.get_sar_model_path({}))

    import os

    assert path.name == SDS_sar_model.DEFAULT_SAR_MODEL_FILENAME
    assert path.parent == Path(os.path.abspath(SDS_sar_model._model_cache_dir()))


# ----------------------------------------------------------------------- download


def make_fake_pooch(tmp_path, retrieve):
    """A stand-in for the pooch module: os_cache points into tmp_path."""
    import types

    fake = types.ModuleType("pooch")
    fake.os_cache = lambda name: tmp_path / name
    fake.retrieve = retrieve
    return fake


def test_fetch_calls_pooch_with_the_pinned_hash(monkeypatch, tmp_path):
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["path"]) / kwargs["fname"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stub")
        return str(target)

    monkeypatch.setitem(sys.modules, "pooch", make_fake_pooch(tmp_path, retrieve))

    path = SDS_sar_model.fetch_default_sar_model()

    assert Path(path) == tmp_path / "coastsat" / SDS_sar_model.DEFAULT_SAR_MODEL_FILENAME
    (call,) = calls
    assert call["url"] == SDS_sar_model.SAR_MODEL_CONFIG["url"]
    assert call["known_hash"] == SDS_sar_model.SAR_MODEL_CONFIG["sha256"]
    assert call["fname"] == SDS_sar_model.DEFAULT_SAR_MODEL_FILENAME
    assert call["progressbar"] is True


def test_fetch_wraps_download_failures(monkeypatch, tmp_path):
    def retrieve(**kwargs):
        raise OSError("network unreachable")

    monkeypatch.setitem(sys.modules, "pooch", make_fake_pooch(tmp_path, retrieve))

    with pytest.raises(SarModelUnavailable, match="could not download"):
        SDS_sar_model.fetch_default_sar_model()


def test_fetch_without_pooch_is_unavailable(monkeypatch):
    # None in sys.modules makes `import pooch` raise ImportError
    monkeypatch.setitem(sys.modules, "pooch", None)

    with pytest.raises(SarModelUnavailable, match="pooch is not installed"):
        SDS_sar_model.fetch_default_sar_model()


def test_default_load_downloads_when_missing(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    SDS_sar_model.clear_model_cache()
    monkeypatch.setattr(
        SDS_sar_model,
        "get_sar_model_path",
        lambda settings=None: str(tmp_path / "absent.onnx"),
    )

    def fetch():
        raise SarModelUnavailable("download attempted")

    monkeypatch.setattr(SDS_sar_model, "fetch_default_sar_model", fetch)

    with pytest.raises(SarModelUnavailable, match="download attempted"):
        SDS_sar_model.load_sar_model(None)


def test_explicit_missing_path_does_not_download(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    SDS_sar_model.clear_model_cache()
    monkeypatch.setattr(
        SDS_sar_model,
        "fetch_default_sar_model",
        lambda: pytest.fail("an explicit path must never trigger a download"),
    )

    with pytest.raises(SarModelUnavailable, match="not found"):
        SDS_sar_model.load_sar_model(str(tmp_path / "missing.onnx"))


# =================================================================================== #
# End-to-end check of the real ONNX model on a real downloaded scene
#
# Skipped unless onnxruntime is installed, the 130 MB model is present, and a
# dual-polarization session exists on disk. Everything else about the model path is
# covered by the stubbed tests above.
# =================================================================================== #


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION = REPO_ROOT / "data" / "test_s1_downloads_7_24" / "S1"


def scene_paths():
    """The VV/VH pair of the first scene of the session, or None."""
    if not (SESSION / "VV").is_dir() or not (SESSION / "VH").is_dir():
        return None
    for vv in sorted((SESSION / "VV").glob("*_VV.tif")):
        vh = SESSION / "VH" / vv.name.replace("_VV.tif", "_VH.tif")
        if vh.exists():
            return str(vv), str(vh)
    return None


PAIR = scene_paths()


def real_model_skip_reason():
    """Why the real model cannot run here, or None when it can."""
    if importlib.util.find_spec("onnxruntime") is None:
        return "onnxruntime is not installed"
    if not Path(SDS_sar_model.get_sar_model_path({})).is_file():
        return "the SAR model .onnx is not present"
    if PAIR is None:
        return "no dual-polarization scene on disk"
    return None


# a per-test mark rather than a module-level pytestmark: the stubbed tests above share
# this file and must keep running when the real model is unavailable
SKIP_REASON = real_model_skip_reason()
requires_real_model = pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))


@pytest.fixture(scope="module")
def segmenter():
    return SDS_sar_model.load_sar_model()


@pytest.fixture(scope="module")
def scene():
    im_ms, _georef = SDS_tools.read_sar_image(list(PAIR))
    composite = SDS_tools.build_sar_composite(im_ms, ["VV", "VH"])
    valid = SDS_tools.read_sar_valid_mask(list(PAIR))
    return composite, valid


@requires_real_model
def test_the_packaged_model_declares_the_expected_contract(segmenter):
    _session, spec = segmenter

    assert list(spec["channel_order"]) == list(SDS_tools.SAR_COMPOSITE_BANDS)
    assert np.allclose(spec["mean"], [-12.59, -20.26, 10.5465])
    assert np.allclose(spec["std"], [5.26, 5.91, 7.6855])
    assert spec["output_stride"] == 32
    assert str(spec["input_scale"]).lower() == "db"


@requires_real_model
def test_the_downloaded_rasters_are_in_db(scene):
    composite, _valid = scene

    # sanity check from the model's guide: VV over land is roughly -5 to -15 dB and
    # VH lower still. A linear-power raster would be near 0 and predict one class.
    assert -30 < np.median(composite[:, :, 0]) < 0
    assert np.median(composite[:, :, 1]) < np.median(composite[:, :, 0])


@requires_real_model
def test_segments_a_real_scene_into_land_and_water(segmenter, scene):
    session, spec = segmenter
    composite, valid = scene

    im_water, prob = SDS_sar_model.segment_water(composite, valid, session, spec)

    assert im_water.shape == composite.shape[:2]
    assert prob.shape == composite.shape[:2]
    # a coastal scene must contain both classes; all-one-class means the input scale
    # or the channel order is wrong
    assert 0.05 < im_water.mean() < 0.95
    assert np.nanmin(prob) >= 0.0 and np.nanmax(prob) <= 1.0


@requires_real_model
def test_a_higher_threshold_never_grows_the_water_class(segmenter, scene):
    session, spec = segmenter
    composite, valid = scene

    lenient, _ = SDS_sar_model.segment_water(composite, valid, session, spec, 0.5)
    strict, _ = SDS_sar_model.segment_water(composite, valid, session, spec, 0.9)

    assert strict.sum() <= lenient.sum()
    assert np.all(lenient[strict])  # strict water is a subset of lenient water


@requires_real_model
def test_the_session_is_cached(segmenter):
    session, _spec = segmenter

    again, _ = SDS_sar_model.load_sar_model()

    assert again is session
