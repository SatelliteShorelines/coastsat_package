"""Tests for SDS_transects: transect/shoreline intersections and the QC filters."""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_tools, SDS_transects
from coastsat.SDS_transects import (
    compute_intersection,
    compute_intersection_QC,
    DEFAULT_SAR_WATER_FRACTION_RANGE,
    reject_outliers,
    sar_qc_keep,
    threshold_filter_applies,
)


# =================================================================================== #
# Constants and helpers shared by more than one section
# =================================================================================== #


MNDWI = SDS_tools.SEGMENTATION_MNDWI
SAR_OTSU = SDS_tools.SEGMENTATION_SAR_OTSU
SAR_MODEL = SDS_tools.SEGMENTATION_SAR_MODEL


# =================================================================================== #
# Equivalence and performance tests for the transect/shoreline intersection functions
#
# `compute_intersection_QC` and `compute_intersection` were optimized to stop measuring the
# distance from every shoreline point to every transect origin in a Python loop. Both were
# pure performance changes: the results are meant to be bitwise identical to what the
# previous implementations produced.
#
# These tests pin that claim down by keeping frozen copies of the pre-optimization code in
# this module and diffing the optimized functions against them, plus a handful of
# hand-computed values so the whole suite does not rest on one shared expression being
# right.
# =================================================================================== #


# Real shorelines live in a projected (metre) CRS, so coordinates are ~1e5-1e6 with
# metre-scale variation on top. Build the synthetic data at that magnitude rather than
# near the origin: the cancellation behaviour there is what production actually hits.
EASTING = 400000.0
NORTHING = 3800000.0


# --------------------------------------------------------------------------------- #
# Frozen pre-optimization implementations (the oracles)
# --------------------------------------------------------------------------------- #
# These are the shipped implementations as of the commit before the optimization, with
# one deliberate substitution: `np.cross(p2 - p1, sl - p1)` is written out as its
# z-component so the oracles keep working once NumPy removes 2D cross products. The
# substitution is algebraically exact and is pinned independently by
# test_np_cross_on_2d_vectors_matches_the_explicit_scalar_form.
#
# Do not "clean these up" - being a faithful record of the old behaviour is their entire
# purpose. They keep the slow per-point Python loops on purpose.


def _reference_compute_intersection_QC(output, transects, settings):
    """Frozen pre-optimization copy of compute_intersection_QC."""
    cross_dist = dict([])

    shorelines = output["shorelines"]
    along_dist = settings["along_dist"]

    for key in transects.keys():
        std_intersect = np.zeros(len(shorelines))
        med_intersect = np.zeros(len(shorelines))
        max_intersect = np.zeros(len(shorelines))
        min_intersect = np.zeros(len(shorelines))
        n_intersect = np.zeros(len(shorelines))

        for i in range(len(shorelines)):
            sl = shorelines[i]

            if len(sl) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
                continue

            X0 = transects[key][0, 0]
            Y0 = transects[key][0, 1]
            temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

            p1 = np.array([X0, Y0])
            p2 = transects[key][-1, :]
            # was: np.abs(np.cross(p2 - p1, sl - p1) / np.linalg.norm(p2 - p1))
            vec = p2 - p1
            w = sl - p1
            with np.errstate(invalid="ignore", divide="ignore"):
                d_line = np.abs(
                    (vec[0] * w[:, 1] - vec[1] * w[:, 0]) / np.linalg.norm(vec)
                )
            d_origin = np.array([np.linalg.norm(sl[k, :] - p1) for k in range(len(sl))])
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            idx_close = np.where(idx_dist)[0]

            if len(idx_close) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
            else:
                xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
                    np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
                )
                xy_rot = np.matmul(Mrot, xy_close)
                xy_rot[0, xy_rot[0, :] < settings["min_chainage"]] = np.nan

                if not np.all(np.isnan(xy_rot[0, :])):
                    std_intersect[i] = np.nanstd(xy_rot[0, :])
                    med_intersect[i] = np.nanmedian(xy_rot[0, :])
                    max_intersect[i] = np.nanmax(xy_rot[0, :])
                    min_intersect[i] = np.nanmin(xy_rot[0, :])
                    n_intersect[i] = np.sum(~np.isnan(xy_rot[0, :]))
                else:
                    std_intersect[i] = np.nan
                    med_intersect[i] = np.nan
                    max_intersect[i] = np.nan
                    min_intersect[i] = np.nan
                    n_intersect[i] = 0

        condition1 = std_intersect <= settings["max_std"]
        condition2 = (max_intersect - min_intersect) <= settings["max_range"]
        condition3 = n_intersect >= settings["min_points"]
        idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)

        if settings["multiple_inter"] == "auto":
            prc_over = np.sum(std_intersect > settings["max_std"]) / len(std_intersect)

            prc_multiple = settings.get("prc_multiple")
            if prc_multiple is None:
                prc_multiple = settings.get("auto_prc")
                if prc_multiple is None:
                    raise KeyError(
                        "Neither 'prc_multiple' nor 'auto_prc' exist in the settings."
                    )

            if prc_over > prc_multiple:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            else:
                med_intersect[~idx_good] = np.nan

        elif settings["multiple_inter"] == "max":
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan

        elif settings["multiple_inter"] == "nan":
            med_intersect[~idx_good] = np.nan

        else:
            raise Exception("the multiple_inter parameter can only be: nan, max or auto")

        cross_dist[key] = med_intersect

    return cross_dist


def _reference_compute_intersection(output, transects, settings):
    """Frozen pre-vectorization copy of compute_intersection."""
    intersections = np.zeros((len(output["shorelines"]), len(transects)))
    for i in range(len(output["shorelines"])):
        sl = output["shorelines"][i]

        for j, key in enumerate(list(transects.keys())):
            X0 = transects[key][0, 0]
            Y0 = transects[key][0, 1]
            temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

            p1 = np.array([X0, Y0])
            p2 = transects[key][-1, :]
            # was: np.abs(np.cross(p2 - p1, sl - p1) / np.linalg.norm(p2 - p1))
            vec = p2 - p1
            w = sl - p1
            with np.errstate(invalid="ignore", divide="ignore"):
                d_line = np.abs(
                    (vec[0] * w[:, 1] - vec[1] * w[:, 0]) / np.linalg.norm(vec)
                )
            d_origin = np.array([np.linalg.norm(sl[k, :] - p1) for k in range(len(sl))])
            idx_dist = np.logical_and(d_line <= settings["along_dist"], d_origin <= 1000)
            temp_sl = sl - np.array(transects[key][0, :])
            phi_sl = np.array(
                [np.arctan2(temp_sl[k, 1], temp_sl[k, 0]) for k in range(len(temp_sl))]
            )
            diff_angle = phi - phi_sl
            idx_angle = np.abs(diff_angle) < np.pi / 2
            idx_close = np.where(np.logical_and(idx_dist, idx_angle))[0]

            if len(idx_close) == 0:
                intersections[i, j] = np.nan
            else:
                xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
                    np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
                )
                xy_rot = np.matmul(Mrot, xy_close)
                intersections[i, j] = np.nanmedian(xy_rot[0, :])

    cross_dist = dict([])
    for j, key in enumerate(list(transects.keys())):
        cross_dist[key] = intersections[:, j]

    return cross_dist


# --------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------- #
def _make_settings(**overrides):
    """CoastSat's documented defaults (see example.py), overridable per test."""
    settings = {
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15,
        "max_range": 30,
        "min_chainage": -100,
        "multiple_inter": "auto",
        "prc_multiple": 0.1,
    }
    settings.update(overrides)
    return settings


def _make_shoreline(n_points, coast_length, seed=0, cross_shore_offset=0.0, noise=4.0):
    """A wiggly shore-parallel shoreline running east from (EASTING, NORTHING)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, coast_length, n_points)
    y = (
        60.0
        + 8.0 * np.sin(x / 400.0)
        + rng.normal(0.0, noise, n_points)
        + cross_shore_offset
    )
    return np.column_stack([EASTING + x, NORTHING + y])


def _make_transects(n_transects, coast_length):
    """Shore-normal transects spread evenly along the coast, pointing north."""
    transects = {}
    for t in range(n_transects):
        x = EASTING + coast_length * (t + 0.5) / n_transects
        transects["T%04d" % t] = np.array(
            [[x, NORTHING - 40.0], [x, NORTHING + 160.0]]
        )
    return transects


def _make_site(n_shorelines, n_points, n_transects, coast_length, seed=0):
    shorelines = [
        _make_shoreline(
            n_points, coast_length, seed=seed + k, cross_shore_offset=0.3 * k
        )
        for k in range(n_shorelines)
    ]
    return {"shorelines": shorelines}, _make_transects(n_transects, coast_length)


def assert_cross_dist_identical(actual, expected):
    """Assert two cross_dist dicts are bitwise identical, NaNs included.

    Bitwise rather than approximate on purpose: the optimization claims exact equality,
    so a tolerance would quietly accept a version that included one extra boundary point
    (which moves a median far less than any sane rtol) and the test would prove nothing.
    """
    assert list(actual.keys()) == list(expected.keys()), (
        "transect keys or their order changed: "
        f"{list(actual.keys())} != {list(expected.keys())}"
    )
    for key in expected:
        exp, act = expected[key], actual[key]
        assert act.shape == exp.shape, f"transect {key!r}: shape {act.shape} != {exp.shape}"
        assert act.dtype == exp.dtype, f"transect {key!r}: dtype {act.dtype} != {exp.dtype}"
        if not np.array_equal(act, exp, equal_nan=True):
            both_nan = np.isnan(act) & np.isnan(exp)
            bad = np.where(~((act == exp) | both_nan))[0]
            raise AssertionError(
                f"transect {key!r}: {len(bad)} of {len(exp)} values differ. "
                f"first indices={bad[:10].tolist()} "
                f"expected={exp[bad[:10]].tolist()} actual={act[bad[:10]].tolist()}"
            )


def _both(output, transects, settings):
    """Run the optimized and the frozen implementation over identical inputs."""
    actual = compute_intersection_QC(output, transects, settings, use_progress_bar=False)
    expected = _reference_compute_intersection_QC(output, transects, settings)
    return actual, expected


# --------------------------------------------------------------------------------- #
# Oracle integrity - guards against the oracle and the implementation sharing a bug
# --------------------------------------------------------------------------------- #
def _numpy_still_crosses_2d_vectors():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.cross(np.array([1.0, 2.0]), np.array([[3.0, 4.0]]))
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _numpy_still_crosses_2d_vectors(),
    reason="this numpy no longer supports 2D cross products, so there is nothing to pin",
)
def test_np_cross_on_2d_vectors_matches_the_explicit_scalar_form():
    """The substitution both the oracle and the implementation rely on.

    Probes numpy's capability rather than sniffing its version, so this simply stops
    running once numpy drops 2D cross products - having been checked by every run
    until then.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        p1 = rng.normal(0.0, 1e5, 2)
        p2 = rng.normal(0.0, 1e5, 2)
        sl = rng.normal(0.0, 1e5, (rng.integers(1, 40), 2))
        vec, w = p2 - p1, sl - p1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = np.cross(vec, w)
        scalar = vec[0] * w[:, 1] - vec[1] * w[:, 0]
        assert np.array_equal(legacy, scalar)


@pytest.mark.parametrize(
    "transect_end, points, expected_chainage",
    [
        # Transect points east, so chainage is just the easting offset.
        (
            (EASTING + 100.0, NORTHING),
            [(EASTING + 10.0, NORTHING), (EASTING + 20.0, NORTHING + 5.0),
             (EASTING + 30.0, NORTHING - 5.0)],
            20.0,
        ),
        # Same, plus a point 200 m landward of the origin: min_chainage=-100 NaNs it out,
        # so it must not drag the median.
        (
            (EASTING + 100.0, NORTHING),
            [(EASTING - 200.0, NORTHING), (EASTING + 10.0, NORTHING),
             (EASTING + 20.0, NORTHING + 5.0), (EASTING + 30.0, NORTHING - 5.0)],
            20.0,
        ),
        # Transect points north, so chainage is the northing offset.
        (
            (EASTING, NORTHING + 100.0),
            [(EASTING, NORTHING + 40.0), (EASTING + 10.0, NORTHING + 50.0),
             (EASTING - 10.0, NORTHING + 60.0)],
            50.0,
        ),
        # Transect at 45 degrees: a point 50 m east and 50 m north sits 50*sqrt(2) along it.
        (
            (EASTING + 100.0, NORTHING + 100.0),
            [(EASTING + 40.0, NORTHING + 40.0), (EASTING + 50.0, NORTHING + 50.0),
             (EASTING + 60.0, NORTHING + 60.0)],
            50.0 * np.sqrt(2.0),
        ),
    ],
)
def test_optimized_matches_hand_computed_chainage(transect_end, points, expected_chainage):
    """Values derived on paper, so a shared algebra bug cannot hide behind the oracle."""
    output = {"shorelines": [np.array(points, dtype=float)]}
    transects = {"T": np.array([[EASTING, NORTHING], list(transect_end)], dtype=float)}
    settings = _make_settings(multiple_inter="nan")

    actual = compute_intersection_QC(output, transects, settings, use_progress_bar=False)

    assert actual["T"][0] == pytest.approx(expected_chainage)


# --------------------------------------------------------------------------------- #
# Equivalence with the frozen implementation
# --------------------------------------------------------------------------------- #
class TestEquivalenceWithReferenceImplementation:
    def test_realistic_site_matches_reference(self):
        output, transects = _make_site(12, 1200, 15, 6000.0)
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)

    def test_long_coast_low_in_range_fraction_matches_reference(self):
        # Far longer than the coasts this package normally sees, so most points are
        # nowhere near any transect and the 1 km origin filter rejects almost everything.
        output, transects = _make_site(4, 3000, 20, 40000.0)
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)

    @pytest.mark.parametrize("multiple_inter", ["auto", "max", "nan"])
    def test_each_multiple_inter_mode_matches_reference(self, multiple_inter):
        # Noisy shorelines so the dispersion QC actually rejects points.
        shorelines = [
            _make_shoreline(600, 3000.0, seed=k, noise=40.0) for k in range(10)
        ]
        output = {"shorelines": shorelines}
        transects = _make_transects(6, 3000.0)
        settings = _make_settings(multiple_inter=multiple_inter)
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)

    @pytest.mark.parametrize("prc_multiple", [0.0, 0.5, 0.99])
    def test_auto_mode_on_both_sides_of_prc_multiple_matches_reference(self, prc_multiple):
        shorelines = [
            _make_shoreline(600, 3000.0, seed=k, noise=40.0) for k in range(10)
        ]
        output = {"shorelines": shorelines}
        transects = _make_transects(6, 3000.0)
        settings = _make_settings(multiple_inter="auto", prc_multiple=prc_multiple)
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)

    @pytest.mark.parametrize("seed", range(12))
    def test_randomized_geometries_match_reference(self, seed):
        rng = np.random.default_rng(seed)
        coast = float(rng.uniform(500.0, 20000.0))
        output = {
            "shorelines": [
                _make_shoreline(
                    int(rng.integers(10, 800)),
                    coast,
                    seed=seed * 100 + k,
                    noise=float(rng.uniform(1.0, 50.0)),
                )
                for k in range(int(rng.integers(1, 8)))
            ]
        }
        transects = _make_transects(int(rng.integers(1, 10)), coast)
        settings = _make_settings(
            along_dist=float(rng.uniform(5.0, 100.0)),
            max_std=float(rng.uniform(2.0, 40.0)),
            max_range=float(rng.uniform(5.0, 80.0)),
            min_chainage=float(rng.uniform(-200.0, 0.0)),
            multiple_inter=str(rng.choice(["auto", "max", "nan"])),
        )
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)

    def test_compute_intersection_matches_its_reference(self):
        output, transects = _make_site(8, 1200, 10, 6000.0)
        settings = _make_settings()
        actual = compute_intersection(output, transects, settings)
        expected = _reference_compute_intersection(output, transects, settings)
        assert_cross_dist_identical(actual, expected)

    @pytest.mark.parametrize("seed", range(6))
    def test_compute_intersection_matches_its_reference_on_random_geometries(self, seed):
        rng = np.random.default_rng(1000 + seed)
        coast = float(rng.uniform(500.0, 20000.0))
        output = {
            "shorelines": [
                _make_shoreline(
                    int(rng.integers(10, 600)),
                    coast,
                    seed=seed * 50 + k,
                    noise=float(rng.uniform(1.0, 50.0)),
                )
                for k in range(int(rng.integers(1, 6)))
            ]
        }
        transects = _make_transects(int(rng.integers(1, 8)), coast)
        settings = _make_settings(along_dist=float(rng.uniform(5.0, 100.0)))
        actual = compute_intersection(output, transects, settings)
        expected = _reference_compute_intersection(output, transects, settings)
        assert_cross_dist_identical(actual, expected)


# --------------------------------------------------------------------------------- #
# Edge-case branches
# --------------------------------------------------------------------------------- #
class TestEdgeCaseBranches:
    def test_empty_shoreline_yields_nan_and_matches_reference(self):
        output = {"shorelines": [np.empty((0, 2)), _make_shoreline(200, 2000.0)]}
        transects = _make_transects(3, 2000.0)
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)
        for key in transects:
            assert np.isnan(actual[key][0])

    def test_shoreline_with_non_finite_coordinates_matches_reference(self):
        """A NaN/inf vertex must drop out on its own, leaving the rest processed.

        `np.linalg.norm` gives a non-finite distance for these, which fails `<= 1000` and
        excludes them for free. Nothing in the current implementation needs to special-case
        them -- but a spatial index would: `scipy.spatial.cKDTree` raises
        `ValueError: data must be finite` on construction. So this guards against anyone
        reintroducing one without masking, which would turn NaN-carrying shorelines from
        working into a crash.
        """
        sl = _make_shoreline(200, 2000.0)
        sl[10] = [np.nan, np.nan]
        sl[20] = [np.inf, NORTHING + 60.0]
        sl[30] = [EASTING + 300.0, -np.inf]
        output = {"shorelines": [sl]}
        transects = _make_transects(4, 2000.0)

        actual, expected = _both(output, transects, _make_settings())

        assert_cross_dist_identical(actual, expected)
        # and it must still find real intersections, not just agree on all-NaN
        assert any(np.isfinite(actual[key]).any() for key in transects)

    def test_shoreline_of_only_non_finite_coordinates_yields_nan(self):
        output = {"shorelines": [np.full((5, 2), np.nan)]}
        transects = _make_transects(2, 2000.0)
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)
        for key in transects:
            assert np.isnan(actual[key][0])

    def test_no_shoreline_points_within_1km_of_origin_yields_nan(self):
        output = {"shorelines": [_make_shoreline(200, 500.0)]}
        # transect origin sits 50 km east of the shoreline
        transects = {
            "far": np.array(
                [[EASTING + 50000.0, NORTHING], [EASTING + 50000.0, NORTHING + 200.0]]
            )
        }
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)
        assert np.isnan(actual["far"][0])

    def test_points_within_1km_but_none_within_along_dist_yields_nan(self):
        output = {"shorelines": [_make_shoreline(200, 500.0)]}
        transects = _make_transects(1, 500.0)
        # along_dist far smaller than the point spacing -> nothing survives the 2nd filter
        settings = _make_settings(along_dist=1e-9)
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)
        assert np.isnan(actual["T0000"][0])

    def test_all_candidates_rejected_by_min_chainage_yields_nan(self):
        output = {"shorelines": [_make_shoreline(200, 2000.0)]}
        transects = _make_transects(3, 2000.0)
        # every intersection is ~60 m along the transect, so demand far more than that
        settings = _make_settings(min_chainage=10000)
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)
        for key in transects:
            assert np.isnan(actual[key][0])

    def test_point_exactly_1000m_from_origin_is_included(self):
        # (600, 800) is a Pythagorean triple, so the distance is exactly 1000.0 in binary
        pts = np.array(
            [
                [EASTING + 600.0, NORTHING + 800.0],
                [EASTING + 600.0, NORTHING + 800.0],
                [EASTING + 600.0, NORTHING + 800.0],
            ]
        )
        assert np.linalg.norm(pts[0] - np.array([EASTING, NORTHING])) == 1000.0
        output = {"shorelines": [pts]}
        # transect pointing at the cluster so d_line == 0
        transects = {"T": np.array([[EASTING, NORTHING], [EASTING + 60.0, NORTHING + 80.0]])}
        settings = _make_settings(multiple_inter="nan")
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)
        assert actual["T"][0] == pytest.approx(1000.0)

    def test_point_exactly_at_along_dist_is_included(self):
        # transect points east; d_line is then simply the northing offset
        pts = np.array(
            [
                [EASTING + 50.0, NORTHING + 25.0],
                [EASTING + 50.0, NORTHING + 25.0],
                [EASTING + 50.0, NORTHING + 25.0],
            ]
        )
        output = {"shorelines": [pts]}
        transects = {"T": np.array([[EASTING, NORTHING], [EASTING + 100.0, NORTHING]])}
        settings = _make_settings(along_dist=25, multiple_inter="nan")
        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)
        assert actual["T"][0] == pytest.approx(50.0)

    def test_transect_with_extra_vertices_uses_only_first_and_last(self):
        output = {"shorelines": [_make_shoreline(300, 2000.0)]}
        two_vertex = {"T": np.array([[EASTING + 1000.0, NORTHING - 40.0],
                                     [EASTING + 1000.0, NORTHING + 160.0]])}
        many_vertex = {"T": np.array([[EASTING + 1000.0, NORTHING - 40.0],
                                      [EASTING + 1000.0, NORTHING + 10.0],
                                      [EASTING + 1000.0, NORTHING + 90.0],
                                      [EASTING + 1000.0, NORTHING + 160.0]])}
        settings = _make_settings()

        a = compute_intersection_QC(output, two_vertex, settings, use_progress_bar=False)
        b = compute_intersection_QC(output, many_vertex, settings, use_progress_bar=False)

        assert_cross_dist_identical(b, a)

    def test_zero_length_transect_yields_nan(self):
        output = {"shorelines": [_make_shoreline(200, 2000.0)]}
        transects = {
            "degenerate": np.array([[EASTING + 500.0, NORTHING], [EASTING + 500.0, NORTHING]])
        }
        actual, expected = _both(output, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)
        assert np.isnan(actual["degenerate"][0])


# --------------------------------------------------------------------------------- #
# Settings handling
# --------------------------------------------------------------------------------- #
class TestSettingsHandling:
    def test_invalid_multiple_inter_raises(self):
        output = {"shorelines": [_make_shoreline(100, 1000.0)]}
        transects = _make_transects(2, 1000.0)
        settings = _make_settings(multiple_inter="bogus")
        with pytest.raises(Exception, match="can only be: nan, max or auto"):
            compute_intersection_QC(output, transects, settings, use_progress_bar=False)

    def test_missing_prc_multiple_and_auto_prc_raises_key_error(self):
        output = {"shorelines": [_make_shoreline(100, 1000.0)]}
        transects = _make_transects(2, 1000.0)
        settings = _make_settings(multiple_inter="auto")
        del settings["prc_multiple"]
        with pytest.raises(KeyError, match="Neither 'prc_multiple' nor 'auto_prc'"):
            compute_intersection_QC(output, transects, settings, use_progress_bar=False)

    def test_auto_prc_is_used_when_prc_multiple_is_absent(self):
        """The example scripts pass auto_prc; the notebook passes prc_multiple."""
        shorelines = [_make_shoreline(400, 2000.0, seed=k, noise=40.0) for k in range(8)]
        output = {"shorelines": shorelines}
        transects = _make_transects(4, 2000.0)

        with_auto_prc = _make_settings(multiple_inter="auto")
        del with_auto_prc["prc_multiple"]
        with_auto_prc["auto_prc"] = 0.1

        actual, expected = _both(output, transects, with_auto_prc)
        assert_cross_dist_identical(actual, expected)
        # and it behaves like the equivalent prc_multiple setting
        assert_cross_dist_identical(
            actual,
            compute_intersection_QC(
                output, transects, _make_settings(prc_multiple=0.1), use_progress_bar=False
            ),
        )

    def test_prc_multiple_takes_precedence_over_auto_prc(self):
        shorelines = [_make_shoreline(400, 2000.0, seed=k, noise=40.0) for k in range(8)]
        output = {"shorelines": shorelines}
        transects = _make_transects(4, 2000.0)
        # auto_prc would take the other branch; prc_multiple must win
        settings = _make_settings(multiple_inter="auto", prc_multiple=0.0, auto_prc=0.99)

        actual, expected = _both(output, transects, settings)
        assert_cross_dist_identical(actual, expected)
        assert_cross_dist_identical(
            actual,
            compute_intersection_QC(
                output, transects, _make_settings(prc_multiple=0.0), use_progress_bar=False
            ),
        )

    def test_empty_transects_returns_empty_dict_without_raising(self):
        # the KeyError above is raised inside the transect loop, so no transects -> no raise
        output = {"shorelines": [_make_shoreline(100, 1000.0)]}
        settings = _make_settings(multiple_inter="auto")
        del settings["prc_multiple"]
        assert compute_intersection_QC(output, {}, settings, use_progress_bar=False) == {}

    def test_no_shorelines_returns_empty_array_per_transect(self):
        transects = _make_transects(3, 1000.0)
        actual, expected = _both({"shorelines": []}, transects, _make_settings())
        assert_cross_dist_identical(actual, expected)
        for key in transects:
            assert actual[key].shape == (0,)


# --------------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------------- #
class TestOutputContract:
    def test_keys_preserve_transect_insertion_order(self):
        output = {"shorelines": [_make_shoreline(200, 3000.0)]}
        transects = _make_transects(6, 3000.0)
        shuffled = {k: transects[k] for k in reversed(list(transects))}

        actual = compute_intersection_QC(
            output, shuffled, _make_settings(), use_progress_bar=False
        )

        assert list(actual.keys()) == list(shuffled.keys())

    def test_each_array_is_float64_with_one_value_per_shoreline(self):
        output, transects = _make_site(7, 300, 4, 3000.0)
        actual = compute_intersection_QC(
            output, transects, _make_settings(), use_progress_bar=False
        )
        for key in transects:
            assert actual[key].dtype == np.float64
            assert actual[key].shape == (7,)

    def test_output_is_invariant_to_shoreline_point_order(self):
        """What the candidate sort is really about.

        Every consumer of the candidate indices is a permutation-invariant reduction, so
        shuffling the input points must not move the output.
        """
        output, transects = _make_site(6, 800, 8, 4000.0)
        settings = _make_settings()
        ordered = compute_intersection_QC(
            output, transects, settings, use_progress_bar=False
        )

        rng = np.random.default_rng(3)
        shuffled = {
            "shorelines": [
                sl[rng.permutation(len(sl))] for sl in output["shorelines"]
            ]
        }
        actual = compute_intersection_QC(
            shuffled, transects, settings, use_progress_bar=False
        )

        assert_cross_dist_identical(actual, ordered)

    @pytest.mark.parametrize("use_progress_bar", [True, False])
    def test_progress_bar_flag_does_not_change_results(self, use_progress_bar):
        output, transects = _make_site(4, 300, 3, 2000.0)
        settings = _make_settings()
        actual = compute_intersection_QC(
            output, transects, settings, use_progress_bar=use_progress_bar
        )
        assert_cross_dist_identical(
            actual, _reference_compute_intersection_QC(output, transects, settings)
        )


# --------------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------------- #
def _min_runtime(fn, repeats=3):
    fn()  # warm-up: pay the one-time import/dispatch costs outside the measurement
    return min(_time_once(fn) for _ in range(repeats))


def _time_once(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


@pytest.mark.perf
def test_optimized_is_substantially_faster_than_reference():
    """The speedup is the whole point of the change, so guard it.

    This asserts a ratio, never a wall-clock budget: a loaded machine slows both sides
    roughly equally, so the ratio stays stable while an absolute threshold would flake.
    The floor is deliberately far below what we measure (~50-70x, fairly flat across
    geometries) because its only job is catching a per-point Python loop coming back.
    Anything that keeps the distance maths vectorized will clear it comfortably.
    """
    output, transects = _make_site(8, 3000, 40, 40000.0)
    settings = _make_settings()

    t_new = _min_runtime(
        lambda: compute_intersection_QC(output, transects, settings, use_progress_bar=False)
    )
    t_old = _min_runtime(
        lambda: _reference_compute_intersection_QC(output, transects, settings), repeats=1
    )

    speedup = t_old / t_new
    # the speedup must not have been bought with wrong answers
    assert_cross_dist_identical(*_both(output, transects, settings))
    assert speedup > 5.0, (
        f"expected a large speedup, got {speedup:.1f}x "
        f"(reference {t_old:.2f}s vs optimized {t_new:.3f}s)"
    )


# =================================================================================== #
# Tests that the otsu_threshold QC filter cannot silently delete SAR shorelines
#
# output['MNDWI_threshold'] holds a different quantity depending on how the shoreline was
# mapped: an MNDWI value (optical), an Otsu threshold in dB (SAR), or NaN (SAR segmentation
# model, which has no threshold). Comparing the latter two against an MNDWI range drops
# them - NaN fails every comparison and a dB value near -20 is outside any sensible MNDWI
# range - so they have to skip the filter instead.
# =================================================================================== #


# wide enough that no optical shoreline is rejected on its own merits
OPTICAL_RANGE = (-0.5, 0.5)


def threshold_make_output(methods, thresholds, satnames=None):
    start = pytz.utc.localize(datetime(2023, 1, 1))
    return {
        "dates": [start + timedelta(days=k) for k in range(len(methods))],
        "MNDWI_threshold": list(thresholds),
        "segmentation_method": list(methods),
        "satname": list(satnames) if satnames else ["S2"] * len(methods),
    }


# ------------------------------------------------------------ threshold_filter_applies


@pytest.mark.parametrize(
    "methods,thresholds,expected",
    [
        ([MNDWI, MNDWI], [-0.1, 0.2], [True, True]),
        # the whole point: a NaN threshold must not be compared against a range
        ([MNDWI, SAR_MODEL], [-0.1, np.nan], [True, False]),
        # its threshold is in dB, which no MNDWI range can meaningfully bound
        ([MNDWI, SAR_OTSU], [-0.1, -19.3], [True, False]),
        # even if something mislabels the method, NaN can never satisfy a range
        ([MNDWI], [np.nan], [False]),
    ],
    ids=["optical", "sar-model", "sar-otsu", "nan-threshold"],
)
def test_threshold_filter_applies(methods, thresholds, expected):
    output = threshold_make_output(methods, thresholds)

    assert threshold_filter_applies(output).tolist() == expected


def test_falls_back_to_satname_for_older_outputs():
    # outputs written before segmentation_method existed
    output = threshold_make_output([MNDWI, SAR_OTSU], [-0.1, -19.3], satnames=["S2", "S1"])
    del output["segmentation_method"]

    assert threshold_filter_applies(output).tolist() == [True, False]


def test_optical_only_outputs_without_either_key_are_unchanged():
    output = {"MNDWI_threshold": [-0.1, 0.2], "dates": [None, None]}

    assert threshold_filter_applies(output).tolist() == [True, True]


# ------------------------------------------------------------------- reject_outliers


def threshold_transects(n, method, threshold, satname="S1"):
    """n shorelines, all mapped the same way, with a clean linear chainage."""
    output = threshold_make_output([method] * n, [threshold] * n, satnames=[satname] * n)
    # a gently varying series so the despiking step keeps everything
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}
    return cross_distance, output


THRESHOLD_SETTINGS = {
    "otsu_threshold": OPTICAL_RANGE,
    "max_cross_change": 40,
    "plot_fig": False,
}


@pytest.mark.parametrize(
    "method,threshold,satname",
    [(SAR_MODEL, np.nan, "S1"), (SAR_OTSU, -19.3, "S1"), (MNDWI, -0.1, "S2")],
    ids=["sar-model", "sar-otsu", "optical-inside-the-range"],
)
def test_shorelines_the_filter_must_not_reject_all_survive(method, threshold, satname):
    cross_distance, output = threshold_transects(40, method, threshold, satname=satname)

    chain_dict = reject_outliers(cross_distance, output, THRESHOLD_SETTINGS)

    assert "transect_1" in chain_dict, "every shoreline was dropped by the QC filter"
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_optical_shorelines_outside_the_range_are_still_rejected():
    # the filter must keep doing its job where it is meaningful
    cross_distance, output = threshold_transects(40, MNDWI, 0.9, satname="S2")

    chain_dict = reject_outliers(cross_distance, output, THRESHOLD_SETTINGS)

    # fewer than 30 survive the range, so the transect is dropped entirely
    assert "transect_1" not in chain_dict


def test_a_mixed_time_series_keeps_sar_and_filters_optical():
    n = 60
    methods = [MNDWI if k % 2 else SAR_MODEL for k in range(n)]
    # optical thresholds are out of range, SAR is NaN
    thresholds = [0.9 if k % 2 else np.nan for k in range(n)]
    satnames = ["S2" if k % 2 else "S1" for k in range(n)]
    output = threshold_make_output(methods, thresholds, satnames=satnames)
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}

    chain_dict = reject_outliers(cross_distance, output, dict(THRESHOLD_SETTINGS))

    # the 30 SAR shorelines survive, the 30 out-of-range optical ones do not
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 30


def test_a_nan_otsu_threshold_setting_still_disables_the_filter():
    cross_distance, output = threshold_transects(40, MNDWI, 0.9, satname="S2")
    settings = dict(THRESHOLD_SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


# --------------------------------------------------------------- the output column


def test_merge_output_carries_the_new_column():
    # merge_output builds its keys from the first satellite, so the column has to be
    # present for every satellite or it is silently dropped for the others
    output = {
        "S1": {
            "dates": [1],
            "shorelines": [np.zeros((2, 2))],
            "MNDWI_threshold": [np.nan],
            "segmentation_method": [SAR_MODEL],
        },
        "S2": {
            "dates": [2],
            "shorelines": [np.zeros((2, 2))],
            "MNDWI_threshold": [-0.1],
            "segmentation_method": [MNDWI],
        },
    }

    merged = SDS_tools.merge_output(output)

    assert merged["segmentation_method"] == [SAR_MODEL, MNDWI]
    assert merged["satname"] == ["S1", "S2"]


# =================================================================================== #
# Tests for the SAR-specific transect QC
#
# SAR shorelines skip the MNDWI threshold filter (see the otsu_threshold section above), which
# left them with no threshold-based QC at all. sar_qc_keep gives them their own checks:
# the water fraction of the reference buffer (a scene segmented as ~all land or ~all
# water has no real interface to contour) and, optionally, a dB range for Otsu
# thresholds - the SAR counterpart of settings['otsu_threshold'].
# =================================================================================== #


def sar_qc_make_output(methods, thresholds=None, water_fractions=None, satnames=None):
    start = pytz.utc.localize(datetime(2023, 1, 1))
    n = len(methods)
    output = {
        "dates": [start + timedelta(days=k) for k in range(n)],
        "MNDWI_threshold": list(thresholds) if thresholds else [np.nan] * n,
        "segmentation_method": list(methods),
        "satname": list(satnames) if satnames else ["S1"] * n,
    }
    if water_fractions is not None:
        output["water_fraction"] = list(water_fractions)
    return output


SAR_QC_SETTINGS = {
    "otsu_threshold": (-0.5, 0.5),
    "max_cross_change": 40,
    "plot_fig": False,
}


# ------------------------------------------------------------------------ sar_qc_keep


def test_healthy_sar_rows_pass():
    output = sar_qc_make_output([SAR_MODEL, SAR_OTSU], water_fractions=[0.5, 0.3])

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True, True]


def test_degenerate_water_fractions_fail():
    # ~0 or ~1 means the segmentation found no land/water interface
    output = sar_qc_make_output(
        [SAR_MODEL, SAR_MODEL, SAR_MODEL], water_fractions=[0.001, 0.999, 0.5]
    )

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [False, False, True]


def test_optical_rows_always_pass():
    # water_fraction is NaN for optical rows, and even a bogus value must not judge them
    output = sar_qc_make_output(
        [MNDWI, MNDWI], water_fractions=[np.nan, 0.999], satnames=["S2", "S2"]
    )

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True, True]


def test_nan_water_fractions_are_kept():
    # an empty reference buffer produces NaN; that is not evidence of a bad scene
    output = sar_qc_make_output([SAR_MODEL], water_fractions=[np.nan])

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True]


def test_outputs_without_the_column_are_kept():
    # outputs written before water_fraction existed
    output = sar_qc_make_output([SAR_MODEL, SAR_OTSU])

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True, True]


def test_the_range_can_be_tightened():
    output = sar_qc_make_output([SAR_MODEL, SAR_MODEL], water_fractions=[0.1, 0.5])
    settings = dict(SAR_QC_SETTINGS, sar_water_fraction_range=(0.3, 0.7))

    assert sar_qc_keep(output, settings).tolist() == [False, True]


def test_none_disables_the_water_fraction_check():
    output = sar_qc_make_output([SAR_MODEL], water_fractions=[0.999])
    settings = dict(SAR_QC_SETTINGS, sar_water_fraction_range=None)

    assert sar_qc_keep(output, settings).tolist() == [True]


def test_the_default_range_is_deliberately_loose():
    low, high = DEFAULT_SAR_WATER_FRACTION_RANGE

    assert low <= 0.05
    assert high >= 0.95


def test_db_threshold_range_is_off_by_default():
    output = sar_qc_make_output([SAR_OTSU], thresholds=[-40.0], water_fractions=[0.5])

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True]


def test_db_threshold_range_judges_otsu_rows():
    output = sar_qc_make_output(
        [SAR_OTSU, SAR_OTSU, SAR_MODEL],
        thresholds=[-40.0, -20.0, np.nan],
        water_fractions=[0.5, 0.5, 0.5],
    )
    settings = dict(SAR_QC_SETTINGS, sar_otsu_threshold=(-30.0, -10.0))

    # the model row has no threshold, so the dB range cannot judge it
    assert sar_qc_keep(output, settings).tolist() == [False, True, True]


def test_older_outputs_fall_back_to_satname():
    output = sar_qc_make_output(
        [MNDWI, MNDWI], water_fractions=[0.999, 0.999], satnames=["S1", "S2"]
    )
    del output["segmentation_method"]

    # only the S1 row is treated as SAR
    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [False, True]


def test_outputs_without_any_key_pass():
    output = {"dates": [1, 2], "MNDWI_threshold": [np.nan, np.nan]}

    assert sar_qc_keep(output, SAR_QC_SETTINGS).tolist() == [True, True]


# -------------------------------------------------------------------- reject_outliers


def sar_qc_transects(n, method, water_fractions, thresholds=None):
    output = sar_qc_make_output(
        [method] * n,
        thresholds=thresholds if thresholds else [np.nan] * n,
        water_fractions=water_fractions,
    )
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}
    return cross_distance, output


def test_degenerate_sar_scenes_are_dropped_from_the_time_series():
    n = 40
    fractions = [0.999 if k < 5 else 0.5 for k in range(n)]
    cross_distance, output = sar_qc_transects(n, SAR_MODEL, fractions)

    chain_dict = reject_outliers(cross_distance, output, SAR_QC_SETTINGS)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_healthy_sar_scenes_all_survive():
    cross_distance, output = sar_qc_transects(40, SAR_MODEL, [0.5] * 40)

    chain_dict = reject_outliers(cross_distance, output, SAR_QC_SETTINGS)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_sar_qc_runs_even_when_the_mndwi_filter_is_disabled():
    # the MNDWI range being NaN used to skip step 2 entirely
    n = 40
    fractions = [0.999 if k < 5 else 0.5 for k in range(n)]
    cross_distance, output = sar_qc_transects(n, SAR_MODEL, fractions)
    settings = dict(SAR_QC_SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_sar_qc_does_not_drop_short_transects_entirely():
    # the <30-points guard belongs to the MNDWI range filter; SAR QC must not
    # start deleting whole transects of short time-series
    n = 10
    cross_distance, output = sar_qc_transects(n, SAR_MODEL, [0.5] * n)
    settings = dict(SAR_QC_SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert "transect_1" in chain_dict
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n


def test_db_range_drops_otsu_outliers_in_the_time_series():
    n = 40
    thresholds = [-40.0 if k < 5 else -20.0 for k in range(n)]
    cross_distance, output = sar_qc_transects(n, SAR_OTSU, [0.5] * n, thresholds=thresholds)
    settings = dict(SAR_QC_SETTINGS, sar_otsu_threshold=(-30.0, -10.0))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_merge_output_carries_the_water_fraction_column():
    # merge_output builds its keys from the first satellite, so the column has to
    # be appended for every satellite or it is silently dropped for the others
    output = {
        "S1": {
            "dates": [1],
            "shorelines": [np.zeros((2, 2))],
            "water_fraction": [0.5],
        },
        "S2": {
            "dates": [2],
            "shorelines": [np.zeros((2, 2))],
            "water_fraction": [np.nan],
        },
    }

    merged = SDS_tools.merge_output(output)

    assert merged["water_fraction"][0] == 0.5
    assert np.isnan(merged["water_fraction"][1])
