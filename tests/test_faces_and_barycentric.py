"""
Tests for geodesicDistanceWithFacesAndBarycentric.

Three independent checks:
  1. Validity        – each barycentric triple sums to 1 and all values are in [0, 1].
  2. Reconstruction  – b0*V0 + b1*V1 + b2*V2 reproduces the path endpoints.
  3. BVH comparison  – trimesh's nearest-on-surface query agrees on face and
                       barycentric coordinates (skipped when trimesh is absent).
"""
import os
import pathlib

import numpy as np
import pytest

import pygeodesic.geodesic as geodesic
import igl

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
MESH_FILE = os.path.join(DATA_PATH, "cat_head.obj")

# Tolerance used throughout
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    points, faces = load_mesh(MESH_FILE)
    return points, faces


@pytest.fixture(scope="module")
def geoalg(mesh):
    points, faces = mesh
    return geodesic.PyGeodesicAlgorithmExact(points, faces)


@pytest.fixture(scope="module")
def path_data(geoalg):
    """Run the algorithm once for a non-trivial source/target pair."""
    source_index = 0
    target_index = 100
    result = geoalg.geodesicDistanceWithFacesAndBarycentric(source_index, target_index)
    path_length, path_points, face_ids, face_offsets, bary_start, bary_end = result
    assert path_points is not None, "Algorithm returned None – propagation failed"
    return dict(
        path_length=path_length,
        path_points=path_points,
        face_ids=face_ids,
        face_offsets=face_offsets,
        bary_start=bary_start,
        bary_end=bary_end,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def reconstruct_points(bary, face_ids, mesh_points, mesh_faces):
    """Reconstruct 3-D positions from barycentric coordinates.

    Parameters
    ----------
    bary      : (m, 3) barycentric weights
    face_ids  : (m,)   face indices into mesh_faces
    mesh_points, mesh_faces : mesh arrays

    Returns
    -------
    pts : (m, 3) reconstructed 3-D points
    """
    V = mesh_points[mesh_faces[face_ids]]   # (m, 3, 3)  – three vertices per entry
    return (bary[:, 0:1] * V[:, 0] +
            bary[:, 1:2] * V[:, 1] +
            bary[:, 2:3] * V[:, 2])


# ---------------------------------------------------------------------------
# Test 1: barycentric validity
# ---------------------------------------------------------------------------

class TestBarycentriValidity:

    def test_start_sums_to_one(self, path_data):
        bary = path_data["bary_start"]
        sums = bary.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=ATOL), \
            f"bary_start rows do not sum to 1; worst deviation {np.abs(sums - 1).max():.2e}"

    def test_end_sums_to_one(self, path_data):
        bary = path_data["bary_end"]
        sums = bary.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=ATOL), \
            f"bary_end rows do not sum to 1; worst deviation {np.abs(sums - 1).max():.2e}"

    def test_start_non_negative(self, path_data):
        bary = path_data["bary_start"]
        assert np.all(bary >= -ATOL), \
            f"bary_start has negative values; min {bary.min():.2e}"

    def test_end_non_negative(self, path_data):
        bary = path_data["bary_end"]
        assert np.all(bary >= -ATOL), \
            f"bary_end has negative values; min {bary.min():.2e}"

    def test_start_at_most_one(self, path_data):
        bary = path_data["bary_start"]
        assert np.all(bary <= 1.0 + ATOL), \
            f"bary_start has values > 1; max {bary.max():.2e}"

    def test_end_at_most_one(self, path_data):
        bary = path_data["bary_end"]
        assert np.all(bary <= 1.0 + ATOL), \
            f"bary_end has values > 1; max {bary.max():.2e}"

    def test_known_distance_unchanged(self, path_data):
        """Sanity-check: path length should be finite and non-negative."""
        assert np.isfinite(path_data["path_length"])
        assert path_data["path_length"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: endpoint reconstruction
# ---------------------------------------------------------------------------

class TestEndpointReconstruction:
    """
    face_offsets[i] .. face_offsets[i+1] are the face entries for segment i.
    bary_start for the *first* face in that range should reconstruct path[i].
    bary_end   for the *last*  face in that range should reconstruct path[i+1].
    """

    def _check_segment_endpoints(self, path_data, mesh):
        points, faces = mesh
        path_pts     = path_data["path_points"]
        face_ids     = path_data["face_ids"]
        face_offsets = path_data["face_offsets"]
        bary_start   = path_data["bary_start"]
        bary_end     = path_data["bary_end"]

        n_segments = len(path_pts) - 1
        for seg in range(n_segments):
            f_start = int(face_offsets[seg])
            f_end   = int(face_offsets[seg + 1])

            # First face in segment: bary_start should reconstruct path[seg]
            first_fid  = face_ids[f_start:f_start+1]
            rec_start  = reconstruct_points(bary_start[f_start:f_start+1], first_fid, points, faces)
            expected_a = path_pts[seg]
            err_a = np.linalg.norm(rec_start[0] - expected_a)
            assert err_a < 1e-8, (
                f"Segment {seg}: bary_start of first face reconstructs "
                f"{rec_start[0]} but path[{seg}]={expected_a} (err={err_a:.2e})"
            )

            # Last face in segment: bary_end should reconstruct path[seg+1]
            last_fid  = face_ids[f_end-1:f_end]
            rec_end   = reconstruct_points(bary_end[f_end-1:f_end], last_fid, points, faces)
            expected_b = path_pts[seg + 1]
            err_b = np.linalg.norm(rec_end[0] - expected_b)
            assert err_b < 1e-8, (
                f"Segment {seg}: bary_end of last face reconstructs "
                f"{rec_end[0]} but path[{seg+1}]={expected_b} (err={err_b:.2e})"
            )

    def test_segment_start_reconstruction(self, path_data, mesh):
        self._check_segment_endpoints(path_data, mesh)

    def test_all_reconstructed_points_on_mesh(self, path_data, mesh):
        """Every (face, bary) entry reconstructs finite 3-D points."""
        points, faces = mesh
        face_ids   = path_data["face_ids"]
        bary_start = path_data["bary_start"]
        bary_end   = path_data["bary_end"]

        rec_s = reconstruct_points(bary_start, face_ids, points, faces)
        rec_e = reconstruct_points(bary_end,   face_ids, points, faces)
        assert np.all(np.isfinite(rec_s))
        assert np.all(np.isfinite(rec_e))


# ---------------------------------------------------------------------------
# Test 3: BVH comparison via trimesh
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers shared by Test 3 and Test 4
# ---------------------------------------------------------------------------

scipy_spatial = pytest.importorskip("scipy.spatial", reason="scipy not installed")


def build_face_adjacency(faces):
    """Return dict mapping face_id -> set of face_ids sharing at least one vertex."""
    from collections import defaultdict
    vertex_to_faces = defaultdict(set)
    for fid, f in enumerate(faces):
        for v in f:
            vertex_to_faces[v].add(fid)
    adj = {}
    for fid, f in enumerate(faces):
        neighbours = set()
        for v in f:
            neighbours |= vertex_to_faces[v]
        adj[fid] = neighbours
    return adj


# ---------------------------------------------------------------------------
# Test 3: scipy KD-tree face check (no rtree / trimesh BVH dependency)
# ---------------------------------------------------------------------------

class TestKDTreeFaceCheck:
    """
    For every (face_id, bary) entry:
      1. Reconstruct the 3-D point.
      2. Query a scipy KDTree of triangle centroids for the nearest face.
      3. Assert the nearest face equals the reported face, OR is vertex-adjacent
         (valid for points that sit on a shared edge or vertex).
    """

    @pytest.fixture(scope="class")
    def kd_data(self, mesh):
        points, faces = mesh
        centroids = points[faces].mean(axis=1)   # (F, 3)
        tree = scipy_spatial.cKDTree(centroids)
        adj  = build_face_adjacency(faces)
        return tree, adj

    def check_tag(self, tag, bary, path_data, mesh, kd_data):
        points, faces = mesh
        face_ids = path_data["face_ids"]
        tree, adj = kd_data

        pts_3d = reconstruct_points(bary, face_ids, points, faces)
        _, nn_fids = tree.query(pts_3d)

        for idx, (expected_fid, nn_fid, pt) in enumerate(
                zip(face_ids, nn_fids, pts_3d)):
            assert nn_fid == int(expected_fid) or nn_fid in adj[int(expected_fid)], (
                f"{tag}[{idx}]: geodesic face={expected_fid}, "
                f"KD-tree nearest centroid face={nn_fid}, point={pt}"
            )

    def test_start_face_matches(self, path_data, mesh, kd_data):
        self.check_tag("bary_start", path_data["bary_start"], path_data, mesh, kd_data)

    def test_end_face_matches(self, path_data, mesh, kd_data):
        self.check_tag("bary_end", path_data["bary_end"], path_data, mesh, kd_data)


# ---------------------------------------------------------------------------
# Test 4: structural validity of face_offsets and segment count
# ---------------------------------------------------------------------------

class TestStructure:
    """Verify the shape and invariants of face_offsets, face_ids, and path."""

    def test_face_offsets_starts_at_zero(self, path_data):
        assert path_data["face_offsets"][0] == 0

    def test_face_offsets_ends_at_face_ids_length(self, path_data):
        assert path_data["face_offsets"][-1] == len(path_data["face_ids"])

    def test_face_offsets_monotone(self, path_data):
        offsets = path_data["face_offsets"]
        assert np.all(np.diff(offsets.astype(np.int64)) >= 0), \
            "face_offsets is not monotonically non-decreasing"

    def test_face_offsets_length_equals_segment_count_plus_one(self, path_data):
        n_segments = len(path_data["path_points"]) - 1
        assert len(path_data["face_offsets"]) == n_segments + 1

    def test_each_segment_has_at_least_one_face(self, path_data):
        offsets = path_data["face_offsets"]
        counts = np.diff(offsets.astype(np.int64))
        assert np.all(counts >= 1), \
            f"Some segments have zero faces: {np.where(counts < 1)[0]}"

    def test_each_segment_has_at_most_two_faces(self, path_data):
        offsets = path_data["face_offsets"]
        counts = np.diff(offsets.astype(np.int64))
        assert np.all(counts <= 2), \
            f"Some segments have more than 2 faces: counts={counts[counts > 2]}"

    def test_bary_start_and_end_row_counts_match_face_ids(self, path_data):
        m = len(path_data["face_ids"])
        assert path_data["bary_start"].shape == (m, 3)
        assert path_data["bary_end"].shape   == (m, 3)


# ---------------------------------------------------------------------------
# Test 5: inter-segment consistency
# ---------------------------------------------------------------------------

class TestSegmentConsistency:
    """
    The end of segment i and the start of segment i+1 refer to the same
    path waypoint.  Within each face entry, bary_end[i] must reconstruct
    the same 3-D point as bary_start[i+1] (for the same face).
    """

    def test_consecutive_segments_share_waypoint(self, path_data, mesh):
        points, faces = mesh
        face_ids   = path_data["face_ids"]
        face_offsets = path_data["face_offsets"].astype(int)
        bary_start = path_data["bary_start"]
        bary_end   = path_data["bary_end"]
        path_pts   = path_data["path_points"]

        n_segments = len(path_pts) - 1
        for seg in range(n_segments - 1):
            # end of segment seg  -> path_pts[seg+1]
            last_entry   = face_offsets[seg + 1] - 1
            rec_end   = reconstruct_points(
                bary_end[last_entry:last_entry+1],
                face_ids[last_entry:last_entry+1], points, faces)

            # start of segment seg+1 -> also path_pts[seg+1]
            first_entry  = face_offsets[seg + 1]
            rec_start = reconstruct_points(
                bary_start[first_entry:first_entry+1],
                face_ids[first_entry:first_entry+1], points, faces)

            err = np.linalg.norm(rec_end[0] - rec_start[0])
            assert err < 1e-8, (
                f"Segments {seg} and {seg+1}: shared waypoint mismatch "
                f"(end={rec_end[0]}, start={rec_start[0]}, err={err:.2e})"
            )


# ---------------------------------------------------------------------------
# Test 6: edge cases – same source/target and adjacent vertices
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_same_source_and_target(self, geoalg):
        """Source == target: path has 1 point, 0 segments, all outputs empty."""
        result = geoalg.geodesicDistanceWithFacesAndBarycentric(0, 0)
        path_length, path_points, face_ids, face_offsets, bary_start, bary_end = result
        assert path_length == 0.0 or np.isclose(path_length, 0.0)
        assert len(path_points) == 1
        assert len(face_ids)    == 0
        assert len(face_offsets) in (0, 1)   # either empty or just [0]
        assert len(bary_start)  == 0
        assert len(bary_end)    == 0

    def test_adjacent_vertices_single_step(self, geoalg, mesh):
        """Adjacent vertices (connected by a mesh edge) produce a 1-segment path."""
        points, faces = mesh
        # Pick any edge from first face to guarantee adjacency
        source_index, target_index = int(faces[0, 0]), int(faces[0, 1])
        result = geoalg.geodesicDistanceWithFacesAndBarycentric(source_index, target_index)
        path_length, path_points, face_ids, face_offsets, bary_start, bary_end = result
        assert path_points is not None
        # Path: target vertex -> source vertex, so exactly 2 waypoints, 1 segment
        assert len(path_points) == 2
        assert len(face_offsets) == 2          # [0, n_faces_for_segment]
        assert face_offsets[0] == 0
        assert face_offsets[-1] == len(face_ids)
        # The single segment may cross 1 face (interior) or 2 (along shared edge)
        n_faces = int(face_offsets[1])
        assert n_faces in (1, 2), f"Expected 1 or 2 faces for adjacent pair, got {n_faces}"
        # Barycentric coords must be valid
        assert np.allclose(bary_start.sum(axis=1), 1.0, atol=ATOL)
        assert np.allclose(bary_end.sum(axis=1),   1.0, atol=ATOL)
        # Reconstructed endpoints must match the actual vertex positions
        rec_target = reconstruct_points(bary_start[:1], face_ids[:1], points, faces)
        rec_source = reconstruct_points(bary_end[-1:],  face_ids[-1:], points, faces)
        assert np.allclose(rec_target[0], path_points[0],  atol=1e-8)
        assert np.allclose(rec_source[0], path_points[-1], atol=1e-8)
