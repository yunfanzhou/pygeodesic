"""Basic unit tests for the pygeodesic library."""
import os
import pathlib
import pygeodesic
import pygeodesic.geodesic as geodesic
import numpy as np
import pytest

igl = pytest.importorskip("igl", reason="igl not installed")
def load_mesh(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return geodesic.read_mesh_from_file(filename)

    v, f = igl.read_triangle_mesh(filename)
    points = np.asarray(v, dtype=np.float64)
    faces = np.asarray(f, dtype=np.int32)

    assert points.ndim == 2 and points.shape[1] == 3, "Mesh vertices should have shape (N,3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "Mesh faces should have shape (M,3)"

    return points, faces


THIS_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = os.path.join(THIS_PATH, "data")


def test_version():
    assert isinstance(pygeodesic.__version__, str)


@pytest.fixture()
def mesh():
    filename = os.path.join(DATA_PATH, "cat_head.obj")
    return load_mesh(filename)


@pytest.fixture()
def geoalg(mesh):
    points, faces = mesh
    return geodesic.PyGeodesicAlgorithmExact(points, faces)


@pytest.fixture()
def random_source_target_pairs(mesh):
    points, _ = mesh
    n = points.shape[0]
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, n, size=(50, 2), dtype=np.int32)

    # avoid degenerate source == target by nudging target
    same = pairs[:, 0] == pairs[:, 1]
    pairs[same, 1] = (pairs[same, 1] + 1) % n
    return pairs


@pytest.mark.parametrize("pair_idx", range(50))
def test_distance(geoalg, random_source_target_pairs, pair_idx):
    source_index, target_index = random_source_target_pairs[pair_idx]
    distance, path = geoalg.geodesicDistance(int(source_index), int(target_index))
    assert np.isfinite(distance)
    assert distance >= 0.0
    assert path.ndim == 2 and path.shape[1] == 3
    assert path.shape[0] >= 1


def test_distances(geoalg, mesh, random_source_target_pairs):
    points, _ = mesh
    source_indices = np.unique(random_source_target_pairs[:, 0]).astype(np.int32)
    distances, best_source = geoalg.geodesicDistances(source_indices)
    assert distances.ndim == 1
    assert best_source.ndim == 1
    assert distances.shape[0] == points.shape[0]
    assert best_source.shape[0] == distances.shape[0]
    for source_index in source_indices:
        assert np.isfinite(distances[source_index])
        assert np.isclose(distances[source_index], 0.0)
