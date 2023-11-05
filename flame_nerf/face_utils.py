import sys
import numpy as np
import torch
from torch.autograd.function import once_differentiable

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.distributions import Normal
sys.path.append('../../FLAME/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

_DEFAULT_MIN_TRIANGLE_AREA: float = 1e-4


class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idxs):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th triangular face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


face_point_distance = _FacePointDistance.apply


def point_mesh_face_distance(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    # max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, idx = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return point_to_face, idx


def mesh_face_point_distance(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # face to point distance: shape (T,)
    face_to_point, idx = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )

    return face_to_point, idx


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)


# -----------------------------------------------------------------------------


def recover_homogenous_affine_transformation(p, p_prime):
    batch_size = p.shape[0]

    p_ext = torch.cat((p, torch.ones((batch_size, 3, 1))), dim=-1)
    p_prime_ext = torch.cat((p_prime, torch.ones((batch_size, 3, 1))), dim=-1)

    # construct intermediate matrix
    Q = p[:, 1:, :] - p[:, 0:1, :]
    Q_prime = p_prime[:, 1:, :] - p_prime[:, 0:1, :]

    # # move tensors to CPU
    # Q = Q.cpu()
    # Q_prime = Q_prime.cpu()

    # calculate rotation matrix
    R = torch.inverse(torch.cat((Q, torch.cross(Q[:, 0, :], Q[:, 1, :], dim=1).unsqueeze(1)), dim=1)) @ \
        torch.cat((Q_prime, torch.cross(Q_prime[:, 0, :], Q_prime[:, 1, :], dim=1).unsqueeze(1)), dim=1)

    # # move R back to the original device
    # R = R.to(device)

    # calculate translation vector
    t = p_prime[:, 0, :] - torch.einsum("ik,ikj->ij", p[:, 0, :], R)

    # calculate affine transformation matrix
    a = torch.cat((R, t[:, None, :]), dim=1)
    b = torch.tensor([0, 0, 0, 1])[None, :, None].expand(a.shape[0], -1, -1)
    return torch.cat((a, b), dim=2)


def transform_pt(point, trans_mat):
    a = np.array([point[0], point[1], point[2], 1])
    ap = np.dot(a, trans_mat)[:3]
    return [ap[0], ap[1], ap[2]]


def distance_from_triangle(point, triangle):
    # Define vectors for triangle edges
    v0 = triangle[..., 2, :] - triangle[..., 0, :]
    v1 = triangle[..., 1, :] - triangle[..., 0, :]
    v2 = point - triangle[..., 0, :]

    # Find normal of the triangle plane
    plane_normal = torch.cross(v0, v1)
    plane_normal = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)
    # Find distance from point to plane using the formula:
    # d = |(P-A).N| / |N|
    d = torch.abs(torch.sum(v2 * plane_normal, dim=-1))

    # Find the closest point on the plane to the given point
    closest_point = point - torch.abs(plane_normal * d.unsqueeze(-1))
    distance_2d = distance_from_triangle_2d(closest_point, triangle)
    return torch.sqrt(torch.pow(d, 2) + torch.pow(distance_2d, 2))


def distance_from_triangle_2d(point, triangle):
    # Calculate vectors for edges of triangle
    v0 = triangle[..., 1, :] - triangle[..., 0, :]
    v1 = triangle[..., 2, :] - triangle[..., 0, :]
    v2 = point - triangle[..., 0, :]
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    inside_triangle = (u >= 0) & (v >= 0) & (w >= 0)

    # Point is outside the triangle
    # Find closest point on each edge and return distance to closest one
    d1 = distance_point_to_line_segment(point, triangle[..., 0, :], triangle[..., 1, :])
    d2 = distance_point_to_line_segment(point, triangle[..., 0, :], triangle[..., 2, :])
    d3 = distance_point_to_line_segment(point, triangle[..., 1, :], triangle[..., 2, :])
    return torch.where(inside_triangle, torch.tensor([0], dtype=torch.float), torch.minimum(torch.minimum(d1, d2), d3))


def distance_point_to_line_segment(point, line_point1, line_point2):
    # Create the line vector
    line_vec = line_point2 - line_point1
    # Normalize the line vector
    line_vec = line_vec / torch.norm(line_vec, dim=-1, keepdim=True)
    # Create a vector from point to line_point1
    point_vec = point - line_point1
    # Get the scalar projection of point_vec onto line_vec
    scalar_projection = torch.sum(point_vec * line_vec, dim=-1)
    # Multiply line_vec by the scalar projection to get the projection vector
    projection_vec = scalar_projection.unsqueeze(-1) * line_vec
    # Add the projection vector to line_point1 to get the projected point
    projected_point = line_point1 + projection_vec
    # Check if the projection is inside or outside the line segment
    inside_segment = (scalar_projection >= 0) & (scalar_projection <= torch.norm(line_point2 - line_point1, dim=-1))

    return torch.where(
        inside_segment, torch.norm(point - projected_point, dim=-1),
        torch.min(torch.norm(point - line_point1, dim=-1), torch.norm(point - line_point2, dim=-1))
    )


"""
def sigma(x, epsilon):
    # Create a Normal distribution with mean 0 and standard deviation epsilon
    dist = Normal(0, epsilon)
    return torch.exp(dist.log_prob(x)) / torch.exp(dist.log_prob(torch.tensor(0)))
"""


def flame_based_alpha_calculator_f_relu(min_d, m, e):
    alpha = 1 - ((m(min_d) / e) - (m(min_d - e) / e))
    return alpha


def flame_based_alpha_calculator_3_face_version(set_of_coordinates, mesh_points, mesh_faces):
    mesh_points = [mesh_points]
    mesh_faces = [mesh_faces]

    set_of_coordinates_size = set_of_coordinates.size()
    set_of_coordinates = [torch.flatten(set_of_coordinates, 0, 1)]

    p_mesh = Meshes(verts=mesh_points, faces=mesh_faces)
    p_points = Pointclouds(points=set_of_coordinates)

    dists, idxs = point_mesh_face_distance(p_mesh, p_points)
    dists = torch.sqrt(dists)
    dists, idxs = torch.reshape(dists, (set_of_coordinates_size[0], set_of_coordinates_size[1])), \
                  torch.reshape(idxs, (set_of_coordinates_size[0], set_of_coordinates_size[1]))

    return dists, idxs
