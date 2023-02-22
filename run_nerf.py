import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

# FLAME zone

sys.path.append('./FLAME/')
from FLAME import FLAME
from os.path import join
from pytorch3d import _C
# from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
import tensorflow as tf
from torch.distributions import Normal
import datetime

# FLAME zone


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(f_vert, f_faces, rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(f_vert, f_faces, rays_flat[i:i + chunk], **kwargs)
        torch.cuda.empty_cache()
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(f_vert, f_faces, H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(f_vert, f_faces, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(f_vert, f_faces, render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(f_vert, f_faces, H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # output_ch = 5 if args.N_importance > 0 else 4
    output_ch = 3
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'use_vert': False,
        'epsilon': args.epsilon,
        'fake_epsilon': args.fake_epsilon,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = 1.
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, alpha_overide=None, fake_alpha=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        if alpha_overide is None:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        else:
            noise = torch.randn(alpha_overide.shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # np.savetxt('dists.txt', dists.cpu().detach().numpy())
    # np.savetxt('alpha_overide.txt', alpha_overide.cpu().detach().numpy())
    # np.savetxt('raw2alpha.txt', raw2alpha(alpha_overide, dists).cpu().detach().numpy())
    # np.savetxt('raw2alpha_noise.txt', raw2alpha(alpha_overide + noise, dists).cpu().detach().numpy())

    if alpha_overide is None:
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    else:
        # dists_c = dists.clone()
        # dists_c[..., -1]=torch.mean(dists[..., :-1],dim=1)
        # alpha = torch.clamp((alpha_overide + noise) * (dists_c / torch.mean(dists[..., :-1],dim=1)[...,None]) , min=0.0, max=1.0) # [N_rays, N_samples]
        alpha = alpha_overide

    if fake_alpha is None:
        fake_alpha = alpha
    # np.savetxt('alpha_d.txt', alpha.cpu().detach().numpy())
    # np.savetxt('alpha_overide_d.txt', alpha_overide.cpu().detach().numpy())

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    fake_weights = fake_alpha * torch.cumprod(torch.cat([torch.ones((fake_alpha.shape[0], 1)), 1. - fake_alpha + 1e-10], -1),
                                        -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, fake_weights, depth_map


# def singular_distance_calculator(coordinates, mesh_points):
#     distances = torch.sqrt(torch.sum((mesh_points - coordinates) ** 2, 1))
#     return distances
# def distance_calculator(set_of_coordinates, mesh_points):
#     distances = torch.empty([set_of_coordinates.shape[0], set_of_coordinates.shape[1], mesh_points.shape[0]])
#     for ray_i in range(distances.shape[0]):
#         for point_i in range(distances.shape[1]):
#             distances[ray_i, point_i] = singular_distance_calculator(set_of_coordinates[ray_i, point_i],
#                                                                           mesh_points)
#     return distances

# PointFaceDistance


# PointFaceDistance

# _DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3
_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-4
# _DEFAULT_MIN_TRIANGLE_AREA: float = 3e-4
# _DEFAULT_MIN_TRIANGLE_AREA: float = 1e-5
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





# Stolen from pytorch3d
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
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, idx = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    # dists, idxs = _C.point_face_dist_forward(
    #     points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    # )
    return point_to_face, idx


def distance_calculator(set_of_coordinates, mesh_points):
    return torch.cdist(set_of_coordinates, mesh_points)


def FLAME_based_alpha_calculator_v_relu(distances, m, e):
    min_d, _ = torch.min(distances, 2)
    # fun = lambda dis_min: 1 - ((m(dis_min) / e) - (m(dis_min - e) / e))
    alpha = 1 - ((m(min_d) / e) - (m(min_d - e) / e))
    return alpha


def sigma(x, epsilon):
    # Create a Normal distribution with mean 0 and standard deviation epsilon
    dist = Normal(0, epsilon)

    return torch.exp(dist.log_prob(x)) / torch.exp(dist.log_prob(torch.tensor(0)))


def FLAME_based_alpha_calculator_v_gauss(distances, m, e):
    min_d, _ = torch.min(distances, 2)
    alpha = sigma(min_d, epsilon=e)
    return alpha


def FLAME_based_alpha_calculator_v_solid(distances, m, e):
    min_d, _ = torch.min(distances, 2)

    alpha = torch.where(min_d <= e, torch.tensor(1), torch.tensor(0))
    return alpha


def FLAME_based_alpha_calculator_f_relu(min_d, m, e):
    alpha = 1 - ((m(min_d) / e) - (m(min_d - e) / e))
    return alpha


def FLAME_based_alpha_calculator_f_gauss(min_d, m, e):
    alpha = sigma(min_d, epsilon=e)
    return alpha


def FLAME_based_alpha_calculator_f_solid(min_d, m, e):
    alpha = torch.where(min_d <= e, torch.tensor(1), torch.tensor(0))
    return alpha


# def FLAME_based_alpha_calculator_2(set_of_coordinates, mesh_points, m, e):
#     min_d, _ = torch.min(torch.cdist(set_of_coordinates, mesh_points), 2)
#     # fun = lambda dis_min: 1 - ((m(dis_min) / e) - (m(dis_min - e) / e))
#     alpha = 1 - ((m(min_d) / e) - (m(min_d - e) / e))
#     return alpha


def FLAME_based_alpha_calculator_3_face_version(set_of_coordinates, mesh_points, mesh_faces):
    # print('mesh_points',mesh_points)
    # print('mesh_faces', mesh_faces)
    mesh_points = [mesh_points]
    mesh_faces = [mesh_faces]
    # print('mesh_points', mesh_points)
    # print('mesh_faces', mesh_faces)
    set_of_coordinates_size = set_of_coordinates.size()
    set_of_coordinates = [torch.flatten(set_of_coordinates, 0, 1)]

    p_mesh = Meshes(verts=mesh_points, faces=mesh_faces)
    p_points = Pointclouds(points=set_of_coordinates)

    # dists, idxs = point_mesh_face_distance(p_mesh, p_points)
    # dists = torch.sqrt(dists)
    # dists, idxs = torch.reshape(dists, (set_of_coordinates_size[0], set_of_coordinates_size[1])), \
    #               torch.reshape(idxs, (set_of_coordinates_size[0], set_of_coordinates_size[1]))
    dists, idxs = point_mesh_face_distance(p_mesh, p_points)
    dists = torch.sqrt(dists)
    dists, idxs = torch.reshape(dists, (set_of_coordinates_size[0], set_of_coordinates_size[1])), \
                  torch.reshape(idxs, (set_of_coordinates_size[0], set_of_coordinates_size[1]))
    # dists = torch.reshape(dists, (set_of_coordinates_size[0], set_of_coordinates_size[1]))

    return dists, idxs

def distance_from_triangle(point, triangle):
    # Define vectors for triangle edges
    v0 = triangle[..., 2,:] - triangle[..., 0,:]
    v1 = triangle[..., 1,:] - triangle[..., 0,:]
    v2 = point - triangle[..., 0,:]

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
    # return d

def distance_from_triangle_2d(point, triangle):
    # Calculate vectors for edges of triangle
    v0 = triangle[..., 1,:] - triangle[..., 0,:]
    v1 = triangle[..., 2,:] - triangle[..., 0,:]
    v2 = point - triangle[..., 0,:]
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
    d1 = distance_point_to_line_segment(point, triangle[..., 0,:], triangle[..., 1,:])
    d2 = distance_point_to_line_segment(point, triangle[..., 0,:], triangle[..., 2,:])
    d3 = distance_point_to_line_segment(point, triangle[..., 1,:], triangle[..., 2,:])
    return torch.where(inside_triangle, torch.tensor([0], dtype=torch.float),torch.minimum(torch.minimum(d1,d2),d3))

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

    return torch.where(inside_segment, torch.norm(point - projected_point, dim=-1), torch.min(torch.norm(point - line_point1, dim=-1), torch.norm(point - line_point2, dim=-1)))

# def distance_from_triangle_2d(points, triangles):
#     # Calculate vectors for edges of triangle
#     v0 = triangles[:,:,1] - triangles[:,:,0]
#     v1 = triangles[:,:,2] - triangles[:,:,0]
#     v2 = points - triangles[:,:,0]
#     d00 = torch.sum(v0*v0,dim=-1)
#     d01 = torch.sum(v0*v1,dim=-1)
#     d11 = torch.sum(v1*v1,dim=-1)
#     d20 = torch.sum(v2*v0,dim=-1)
#     d21 = torch.sum(v2*v1,dim=-1)
#     denom = d00 * d11 - d01 * d01
#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1 - v - w
#     inside_triangle = (u >= 0) & (v >= 0) & (w >= 0)
#     inside_triangle = inside_triangle.unsqueeze(-1)
#     closest_point = v.unsqueeze(-1)*v0 + w.unsqueeze(-1)*v1 + triangles[:,:,0]
#     return torch.where(inside_triangle,torch.tensor(0.0),distance_point_to_line_segment(points,closest_point,triangles))
#
# def distance_point_to_line_segment(points, closest_points,triangles):
#     # Create the line vector
#     line_vec = triangles[:,:,1]-triangles[:,:,0]
#     # Normalize the line vector
#     line_vec = line_vec / torch.norm(line_vec,dim=-1,keepdim=True)
#     # Create a vector from point to line_point1
#     point_vec = points - triangles[:,:,0]
#     # Get the scalar projection of point_vec onto line_vec
#     scalar_projection = torch.sum(point_vec*line_vec,dim=-1)
#     # Multiply line_vec by the scalar projection to get the projection vector
#     projection_vec = scalar_projection.unsqueeze(-1)*line_vec
#     # Add the projection vector to line_point1 to get the projected point
#     projected_point = triangles[:,:,0] + projection_vec
#     # Check if the projection is inside or outside the line segment
#     inside_segment = (scalar_projection >= 0) & (scalar_projection <= torch.norm(triangles[:,:,1]-triangles[:,:,0],dim=-1))
#     inside_segment = inside_segment.unsqueeze(-1)
#     return torch.where(inside_segment,torch.norm(points-projected_point,dim=-1),torch.min(torch.norm(points-triangles[:,:,0],dim=-1),torch.norm(points-triangles[:,:,1],dim=-1)))

def FLAME_based_alpha_calculator_4_face_version(set_of_coordinates, mesh_points, mesh_faces ):
    f_distances = torch.empty(set_of_coordinates.shape[0], set_of_coordinates.shape[1], mesh_faces.shape[0])
    for i, points in enumerate(set_of_coordinates):
        # Iterate through all the faces in the mesh
        for j, point in enumerate(points):
            for k, face in enumerate(mesh_faces):
                # Get the coordinates of the triangle
                triangle = mesh_points[face]
                # Apply the distance function to the point and triangle
                f_distances[i][j][k] = distance_from_triangle(point, triangle)
    return f_distances

# def FLAME_based_alpha_calculator_5_face_version(set_of_coordinates, mesh_points, mesh_faces ,v_distances):
#     f_distances = torch.empty(set_of_coordinates.shape[0], set_of_coordinates.shape[1])
#     top_i = torch.topk(v_distances, k=3, dim=2).indices
#
#     for i, points in enumerate(set_of_coordinates):
#         # Iterate through all the faces in the mesh
#         for j, point in enumerate(points):
#             triangle = mesh_points[top_i[i][j]]
#             # Apply the distance function to the point and triangle
#             f_distances[i][j] = distance_from_triangle(point, triangle)
#     return f_distances


def FLAME_based_alpha_calculator_5_face_version(set_of_coordinates, mesh_points, mesh_faces ,v_distances):
    # Get the top 3 closest mesh points for each point in set_of_coordinates
    top_i = torch.topk(v_distances, k=3, dim=2, largest=False).indices

    # Get the coordinates of the top 3 closest mesh points for each point in set_of_coordinates
    triangle = mesh_points[top_i]

    # Calculate the distance from each point in set_of_coordinates to the corresponding triangle
    f_distances = distance_from_triangle(set_of_coordinates, triangle)

    return f_distances

def transform_pt(point, trans_mat):
    a  = np.array([point[0], point[1], point[2], 1])
    ap = np.dot(a, trans_mat)[:3]
    return [ap[0], ap[1], ap[2]]

def render_rays(f_vert,
                f_faces,
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                use_vert=False,
                epsilon=0.04,
                fake_epsilon=0.06,
                trans_mat=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    m = torch.nn.ReLU()
    # epsilon_v = torch.tensor(0.025)
    # epsilon_f = torch.tensor(0.025)
    # fake_epsilon_f = torch.tensor(0.05)
    # distances_v = distance_calculator(pts, f_vert)
    if use_vert:
        pass
        # alpha = FLAME_based_alpha_calculator_v_relu(distances_v, m, epsilon_v)
        # alpha = FLAME_based_alpha_calculator_v_gauss(distances_v, m, epsilon_v)
        # alpha = FLAME_based_alpha_calculator_v_solid(distances_v, m, epsilon_v)
    else:
        distances_f, idx_f = FLAME_based_alpha_calculator_3_face_version(pts, f_vert, f_faces)
        # distances_f = FLAME_based_alpha_calculator_5_face_version(pts, f_vert, f_faces,distances_v)
        alpha = FLAME_based_alpha_calculator_f_relu(distances_f, m, epsilon)
        fake_alpha = FLAME_based_alpha_calculator_f_relu(distances_f, m, fake_epsilon)
        # alpha = FLAME_based_alpha_calculator_f_gauss(distances_f, m, epsilon_f)
        # alpha = FLAME_based_alpha_calculator_f_solid(distances_f, m, epsilon_f)

    # distances_v = torch.cat((distances_v, torch.unsqueeze(alpha_f, 2)), dim=-1)
    # torch.save(distances_v, 'distances_v.txt')
    # torch.save(alpha_v, 'alpha_v.txt')
    # torch.save(distances_f, 'distances_f.txt')
    # torch.save(alpha_f, 'alpha_f.txt')
    # alpha=FLAME_based_alpha_calculator_2(pts, f_vert, m, e)
    # print('pts',pts)
    # print('pts_s', pts.size())
    # print('viewdirs', viewdirs)
    # print('viewdirs_s', viewdirs.size())

    if trans_mat is not None:
        trans_mat_organ = trans_mat[idx_f, :]
        pts = transform_pt(pts, trans_mat_organ)

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, fake_weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest, alpha_overide=alpha,fake_alpha=fake_alpha)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, fake_weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        # distances_v = distance_calculator(pts, f_vert)
        if use_vert:
            pass
            # alpha = FLAME_based_alpha_calculator_v_relu(distances_v, m, epsilon_v)
            # alpha = FLAME_based_alpha_calculator_v_gauss(distances_v, m, epsilon_v)
            # alpha = FLAME_based_alpha_calculator_v_solid(distances_v, m, epsilon_v)
        else:
            distances_f, idx_f = FLAME_based_alpha_calculator_3_face_version(pts, f_vert, f_faces)
            # distances_f = FLAME_based_alpha_calculator_5_face_version(pts, f_vert, f_faces, distances_v)
            alpha = FLAME_based_alpha_calculator_f_relu(distances_f, m, epsilon)
            # alpha = FLAME_based_alpha_calculator_f_gauss(distances_f, m, epsilon_f)
            # alpha = FLAME_based_alpha_calculator_f_solid(distances_f, m, epsilon_f)

        # distances_v = torch.cat((distances_v, torch.unsqueeze(alpha_f, 2)), dim=-1)
        # alpha = FLAME_based_alpha_calculator_2(pts, f_vert, m, e)

        run_fn = network_fn if network_fine is None else network_fine

        if trans_mat is not None:
            trans_mat_organ = trans_mat[idx_f, :]
            pts = transform_pt(pts, trans_mat_organ)

        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, fake_weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest, alpha_overide=alpha)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=1,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=1,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--chunk_render", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_render", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--fake_epsilon", type=float, default=0.06)

    # FLAME zone
    parser.add_argument(
        '--flame_model_path',
        type=str,
        default='./FLAME/model/generic_model.pkl',
        help='flame model path'
    )

    parser.add_argument(
        '--static_landmark_embedding_path',
        type=str,
        default='./FLAME/model/flame_static_embedding.pkl',
        help='Static landmark embeddings path for FLAME'
    )

    parser.add_argument(
        '--dynamic_landmark_embedding_path',
        type=str,
        default='./FLAME/model/flame_dynamic_embedding.npy',
        help='Dynamic contour embedding path for FLAME'
    )

    # FLAME hyper-parameters

    parser.add_argument(
        '--shape_params',
        type=int,
        default=100,
        help='the number of shape parameters'
    )

    parser.add_argument(
        '--expression_params',
        type=int,
        default=50,
        help='the number of expression parameters'
    )

    parser.add_argument(
        '--pose_params',
        type=int,
        default=6,
        help='the number of pose parameters'
    )

    # Training hyper-parameters

    parser.add_argument(
        '--use_face_contour',
        default=True,
        type=bool,
        help='If true apply the landmark loss on also on the face contour.'
    )

    parser.add_argument(
        '--use_3D_translation',
        default=True,  # Flase for RingNet project
        type=bool,
        help='If true apply the landmark loss on also on the face contour.'
    )

    parser.add_argument(
        '--optimize_eyeballpose',
        default=True,  # False for For RingNet project
        type=bool,
        help='If true optimize for the eyeball pose.'
    )

    parser.add_argument(
        '--optimize_neckpose',
        default=True,  # False For RingNet project
        type=bool,
        help='If true optimize for the neck pose.'
    )

    parser.add_argument(
        '--num_worker',
        type=int,
        default=4,
        help='pytorch number worker.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Training batch size.'
    )

    parser.add_argument(
        '--ring_margin',
        type=float,
        default=0.5,
        help='ring margin.'
    )

    parser.add_argument(
        '--ring_loss_weight',
        type=float,
        default=1.0,
        help='weight on ring loss.'
    )
    # FLAME zone
    return parser


# FLAME zone
def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)


# -----------------------------------------------------------------------------

def safe_mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

def recover_homogenous_affine_transformation(p, p_prime):
    batch_size = p.shape[0]

    p_ext = torch.cat((p, torch.ones((batch_size, 3, 1))), dim=-1)
    p_prime_ext = torch.cat((p_prime, torch.ones((batch_size, 3, 1))), dim=-1)

    # construct intermediate matrix
    Q       = p[:, 1:, :]       - p[:, 0:1, :]
    Q_prime = p_prime[:, 1:, :] - p_prime[:, 0:1, :]

    # calculate rotation matrix
    R = torch.inverse(torch.cat((Q, torch.cross(Q[:, 0, :], Q[:, 1, :], dim=1).unsqueeze(1)), dim=1)) @ \
        torch.cat((Q_prime, torch.cross(Q_prime[:, 0, :], Q_prime[:, 1, :], dim=1).unsqueeze(1)), dim=1)

    # calculate translation vector
    t = p_prime[:, 0, :] - torch.einsum("ik,ikj->ij", p[:, 0, :], R)

    # calculate affine transformation matrix
    a=torch.cat((R, t[:,None,:]), dim=1 )
    b=torch.tensor([0,0,0,1])[None, :,None].expand(a.shape[0],-1,-1)
    return torch.cat((a, b), dim=2)

def transform_pt(point, trans_mat):
    a = torch.cat((point, torch.ones(point.shape[0],point.shape[1],1)), dim=-1)
    # ap = (a @ trans_mat)[..., :3]torch.einsum("ijk,ik->ij", trans_mat, a)
    ap = torch.einsum("tik,tikj->tij", a,trans_mat)[..., :3]
    return ap.squeeze(0)
# FLAME zone

def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        # near = 3.
        # far = 5.5

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # FLAME zone
    w_shape_reg = 1e-4
    w_shape_reg_trans = 1e-2
    flamelayer = FLAME(args).to(device)
    f_shape = nn.Parameter(torch.zeros(1, 100).float().to(device))
    f_exp = nn.Parameter(torch.zeros(1, 50).float().to(device))
    f_pose = nn.Parameter(torch.zeros(1, 6).float().to(device))
    f_neck_pose = nn.Parameter(torch.zeros(1, 3).float().to(device))
    f_trans = nn.Parameter(torch.zeros(1, 3).float().to(device))

    f_lr = 0.001
    f_wd = 0.0001
    f_opt = torch.optim.Adam(
        params=[f_shape, f_exp, f_pose, f_neck_pose, f_trans],
        lr=f_lr,
        weight_decay=f_wd
    )

    # checkpoints hack
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        f_opt.load_state_dict(ckpt['f_optimizer_state_dict'])
        f_shape = ckpt['f_shape'].to(device)
        f_exp = ckpt['f_exp'].to(device)
        f_pose = ckpt['f_pose'].to(device)
        f_neck_pose = ckpt['f_neck_pose'].to(device)
        f_trans = ckpt['f_trans'].to(device)
        args.epsilon = ckpt['epsilon']
        args.fake_epsilon = ckpt['fake_epsilon']
    # checkpoints hack

    vertice, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=f_neck_pose, transl=f_trans)
    vertice = torch.squeeze(vertice)
    vertice = vertice.cuda()

    vertice = vertice[:, [0, 2, 1]]
    vertice[:, 1] = -vertice[:, 1]
    vertice *= 8

    faces = flamelayer.faces
    faces = torch.tensor(faces.astype(np.int32))
    faces = torch.squeeze(faces)
    faces = faces.cuda()
    # Write to an .obj file
    # outmesh_dir = './output'
    # safe_mkdir(outmesh_dir)
    # outmesh_path = join(outmesh_dir, 'hello_flame.obj')
    # write_simple_obj(mesh_v=vertice.cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)

    # FLAME zone

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():

            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)

            vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=f_neck_pose, transl=f_trans)
            vertice_out = torch.squeeze(vertice_out)
            # outmesh_dir = './output/t_face_4_8'
            # safe_mkdir(outmesh_dir)
            outmesh_path = os.path.join(testsavedir, 'face.obj')

            write_simple_obj(mesh_v=vertice_out.cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)

            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(vertice, faces, render_poses, hwf, K, args.chunk_render, render_kwargs_test,
                                  gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)

            if images is None:
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            else:
                images_o = torch.tensor(images).to(device=device, dtype=torch.float)
                rgbs = torch.tensor(rgbs).to(device=device, dtype=torch.float)

                print('images_o',images_o.shape)
                print('rgbs', rgbs.shape)

                images_o=torch.movedim(images_o, 3, 1)
                rgbs = torch.movedim(rgbs, 3, 1)

                print('images_o_m', images_o.shape)
                print('rgbs_m', rgbs.shape)

                print('images_o_d', images_o.dtype)
                print('rgbs_d', rgbs.dtype)

                # images_o = images_o.double()
                # rgbs = rgbs.double()
                #
                # print('images_o_d', images_o.dtype)
                # print('rgbs_d', rgbs.dtype)

                psnr = PeakSignalNoiseRatio()
                img_psnr_1=psnr(rgbs, images_o)

                img_loss = img2mse(rgbs, images_o)
                img_psnr_2 = mse2psnr(img_loss)

                ssim = StructuralSimilarityIndexMeasure()
                img_ssim=ssim(rgbs, images_o)

                lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
                img_lpips=lpips(rgbs, images_o)

                print("img_loss", img_loss)
                print("img_psnr_1", img_psnr_1)
                print("img_psnr_2", img_psnr_2)
                print("img_ssim", img_ssim)
                print("img_lpips", img_lpips)

                outstats_path = os.path.join(testsavedir, 'stats.txt')
                with open(outstats_path, 'w') as fp:
                    for s,v in {"img_loss": img_loss, "img_psnr": img_psnr_2, "img_ssim": img_ssim, "img_lpips": img_lpips}.items():
                        fp.write('%s %f\n' % (s, v))
                # torch.save({"img_loss": img_loss, "img_psnr": img_psnr_2, "img_ssim": img_ssim, "img_lpips": img_lpips}, outstats_path)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/tensorboard_logs/' + expname + '/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    alpha1 = 1.
    alpha1_grad = 50000

    # max_eps=0.4
    # min_eps=0.1
    #
    # render_kwargs_test['epsilon']=max_eps
    # render_kwargs_train['epsilon']=max_eps

    # N_iters = 100000 + 1
    # N_iters = 200000 + 1
    N_iters = 50000 + 1
    # N_iters = 20000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####

        # FLAME zone
        f_opt.zero_grad()
        vertice, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=f_neck_pose, transl=f_trans)
        vertice = torch.squeeze(vertice)
        optimizer.zero_grad()
        # max_x = torch.max(vertice[:, 0])
        # max_y = torch.max(vertice[:, 1])
        # max_z = torch.max(vertice[:, 2])
        # min_x = torch.min(vertice[:, 0])
        # min_y = torch.min(vertice[:, 1])
        # min_z = torch.min(vertice[:, 2])
        #
        # print("Most extreme max point on x-axis:", vertice[vertice[:, 0] == max_x])
        # print("Most extreme min point on x-axis:", vertice[vertice[:, 0] == min_x])
        # print("Most extreme max point on y-axis:", vertice[vertice[:, 1] == max_y])
        # print("Most extreme min point on y-axis:", vertice[vertice[:, 1] == min_y])
        # print("Most extreme max point on z-axis:", vertice[vertice[:, 2] == max_z])
        # print("Most extreme min point on z-axis:", vertice[vertice[:, 2] == min_z])

        vertice = vertice[:, [0, 2, 1]]
        vertice[:, 1] = -vertice[:, 1]
        vertice *= 8

        # v=vertice.cpu().detach().numpy()
        # np.set_printoptions(threshold=sys.maxsize)

        # FLAME zone

        # render_kwargs_train['use_vert'] = True
        # rgb_v, disp_v, acc_v, extras_v = render(vertice, faces, H, W, K, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)
        #
        #
        # img_loss_v = img2mse(rgb_v, target_s)
        # trans_v = extras_v['raw'][..., -1]
        # loss_v = img_loss_v
        # psnr_v = mse2psnr(img_loss_v)
        #
        # if 'rgb0' in extras_v:
        #     img_loss0_v = img2mse(extras_v['rgb0'], target_s)
        #     loss_v = loss_v + img_loss0_v
        #     psnr0_v = mse2psnr(img_loss0_v)
        #
        # loss_v = loss_v + (torch.sum(f_shape ** 2) / 2) * 1e-4  # *1e-4
        # loss_v = loss_v + (torch.sum(f_exp ** 2) / 2) * 1e-4  # *1e-4
        # loss_v = loss_v + (torch.sum(f_pose ** 2) / 2) * 1e-4  # *1e-4
        # loss_v = loss_v + (torch.sum(f_trans ** 2) / 2) * 1e-4  # *1e-4

        render_kwargs_train['use_vert'] = False
        rgb_f, disp_f, acc_f, extras_f = render(vertice, faces, H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        img_loss_f = img2mse(rgb_f, target_s)
        trans_f = extras_f['raw'][..., -1]
        loss_f = img_loss_f
        psnr_f = mse2psnr(img_loss_f)

        if 'rgb0' in extras_f:
            img_loss0_f = img2mse(extras_f['rgb0'], target_s)
            loss_f = loss_f + img_loss0_f
            psnr0_f = mse2psnr(img_loss0_f)

        # loss_f = loss_f + (torch.sum(f_shape ** 2) / 2) * 1e-4  # *1e-4
        # loss_f = loss_f + (torch.sum(f_exp ** 2) / 2) * 1e-4  # *1e-4
        # loss_f = loss_f + (torch.sum(f_pose ** 2) / 2) * 1e-4  # *1e-4
        # loss_f = loss_f + (torch.sum(f_trans ** 2) / 2) * 1e-3  # *1e-4


        # loss = alpha1 * loss_v + (1. - alpha1) * loss_f
        # psnr = alpha1 * psnr_v + (1. - alpha1) * psnr_f

        loss=loss_f
        psnr=psnr_f

        # if i<alpha1_grad:
        #     alpha1 = 1 - (i / alpha1_grad)
        # else:
        #     alpha1 = 0
        #
        # render_kwargs_test['epsilon']=min_eps+alpha1*(max_eps-min_eps)
        # render_kwargs_train['epsilon']=min_eps+alpha1*(max_eps-min_eps)

        # loss = loss_f
        # psnr = psnr_f

        # print("loss: ",loss)
        loss.backward()
        # FLAME zone
        f_opt.step()
        # FLAME zone

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####
        torch.cuda.empty_cache()
        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if render_kwargs_train['network_fine'] is None:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f_optimizer_state_dict': f_opt.state_dict(),
                    'f_shape': f_shape,
                    'f_exp': f_exp,
                    'f_pose': f_pose,
                    'f_trans': f_trans,
                    'f_neck_pose': f_neck_pose,
                    'epsilon': render_kwargs_test['epsilon'],
                    'fake_epsilon': render_kwargs_test['fake_epsilon'],
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f_optimizer_state_dict': f_opt.state_dict(),
                    'f_shape': f_shape,
                    'f_exp': f_exp,
                    'f_pose': f_pose,
                    'f_trans': f_trans,
                    'f_neck_pose': f_neck_pose,
                    'epsilon': render_kwargs_test['epsilon'],
                    'fake_epsilon': render_kwargs_test['fake_epsilon'],
                }, path)
            print('Saved checkpoints at', path)

        # render_kwargs_test['use_vert'] = True
        # torch.cuda.empty_cache()
        # if i % args.i_testset == 0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_v_{:06d}'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #
        #     vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, transl=f_trans)
        #     vertice_out = torch.squeeze(vertice_out)
        #     outmesh_path = os.path.join(testsavedir, 'face.obj')
        #     write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)
        #
        #     print('test poses shape', poses[i_test].shape)
        #     with torch.no_grad():
        #         render_path(vertice, faces, torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk_render,
        #                     render_kwargs_test,
        #                     gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
        #     print('Saved test set')
        # torch.cuda.empty_cache()
        # if i % args.i_video == 0 and i > 0:
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         rgbs, disps = render_path(vertice, faces, render_poses, hwf, K, args.chunk_render, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_v_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        render_kwargs_test['use_vert'] = False
        torch.cuda.empty_cache()
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_f_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=f_neck_pose, transl=f_trans)
                vertice_out = torch.squeeze(vertice_out)
                outmesh_path = os.path.join(testsavedir, 'face.obj')
                write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)

                print('test poses shape', poses[i_test].shape)

                render_path(vertice, faces, torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk_render,
                            render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
            print('Saved test set')

        # if i % args.i_testset == 0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_f_{:06d}_rot0'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #     with torch.no_grad():
        #         vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=f_neck_pose, transl=f_trans)
        #         vertice_out = torch.squeeze(vertice_out)
        #
        #         outmesh_path = os.path.join(testsavedir, 'face.obj')
        #         write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)
        #
        #         vertice_out = vertice_out[:, [0, 2, 1]]
        #         vertice_out[:, 1] = -vertice_out[:, 1]
        #         vertice_out *= 8
        #
        #         print('test poses shape', poses[i_test].shape)
        #         triangles_org = vertice[faces.long() - 1, :]
        #         triangles_out = vertice_out[faces.long() - 1, :]
        #         render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out, triangles_org)
        #         render_path(vertice, faces, torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk_render,
        #                     render_kwargs_test,
        #                     gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
        #         render_kwargs_test['trans_mat'] = None
        #     print('Saved test set')

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_f_{:06d}_rot1'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            radian = np.pi / 180.0
            with torch.no_grad():
                f_pose_rot=f_pose.clone().detach()
                f_pose_rot[0,3]=10.0*radian

                vertice_out, _ = flamelayer(f_shape, f_exp, f_pose_rot, neck_pose=f_neck_pose, transl=f_trans)
                vertice_out = torch.squeeze(vertice_out)

                outmesh_path = os.path.join(testsavedir, 'face.obj')
                write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)

                vertice_out = vertice_out[:, [0, 2, 1]]
                vertice_out[:, 1] = -vertice_out[:, 1]
                vertice_out *= 8

                print('test poses shape', poses[i_test].shape)
                triangles_org = vertice[faces.long() - 1, :]
                triangles_out = vertice_out[faces.long() - 1, :]
                render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out, triangles_org)
                render_path(vertice_out, faces, torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk_render,
                            render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
                render_kwargs_test['trans_mat'] = None
            print('Saved test set')

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_f_{:06d}_rot2'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            radian = np.pi / 180.0
            with torch.no_grad():
                out_neck_pose=nn.Parameter(torch.zeros(1, 3).float().to(device))
                out_neck_pose[0, 1] = 20.0 * radian

                vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, neck_pose=out_neck_pose, transl=f_trans)
                vertice_out = torch.squeeze(vertice_out)

                outmesh_path = os.path.join(testsavedir, 'face.obj')
                write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)

                vertice_out = vertice_out[:, [0, 2, 1]]
                vertice_out[:, 1] = -vertice_out[:, 1]
                vertice_out *= 8

                print('test poses shape', poses[i_test].shape)
                triangles_org = vertice[faces.long() - 1, :]
                triangles_out = vertice_out[faces.long() - 1, :]
                render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out,triangles_org)
                render_path(vertice_out, faces, torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk_render,
                            render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir, render_factor=args.render_factor)
                render_kwargs_test['trans_mat'] = None
            print('Saved test set')

        torch.cuda.empty_cache()
        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(vertice, faces, render_poses, hwf, K, args.chunk_render,
                                          render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_f_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(vertice, faces,, render_poses, hwf, args.chunk_render, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
        torch.cuda.empty_cache()
        # if i % args.i_video == 0 and i > 0:
        #     with torch.no_grad():
        #
        #         if args.render_test:
        #             # render_test switches to test poses
        #             images_r = images[i_test]
        #         else:
        #             # Default is smoother render_poses path
        #             images_r = None
        #
        #         testsavedir = os.path.join(basedir, expname,
        #                                    'render_{}_{:06d}'.format('test' if args.render_test else 'path', i))
        #         os.makedirs(testsavedir, exist_ok=True)
        #
        #         vertice_out, _ = flamelayer(f_shape, f_exp, f_pose, transl=f_trans)
        #         vertice_out = torch.squeeze(vertice_out)
        #         # outmesh_dir = './output/t_face_4_8'
        #         # safe_mkdir(outmesh_dir)
        #         outmesh_path = os.path.join(testsavedir, 'face.obj')
        #
        #         write_simple_obj(mesh_v=vertice_out.cpu().numpy(), mesh_f=flamelayer.faces, filepath=outmesh_path)
        #
        #         print('test poses shape', render_poses.shape)
        #
        #         rgbs, _ = render_path(vertice, faces, render_poses, hwf, K, args.chunk_render, render_kwargs_test,
        #                               gt_imgs=images_r,
        #                               savedir=testsavedir, render_factor=args.render_factor)
        #         print('Done rendering', testsavedir)
        #         imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        torch.cuda.empty_cache()
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', data=loss.item(), step=global_step)
            tf.summary.scalar('psnr', data=psnr.item(), step=global_step)

            # tf.summary.scalar('loss_v', data=loss_v.item(), step=global_step)
            # tf.summary.scalar('psnr_v', data=psnr_v.item(), step=global_step)
            #
            # tf.summary.scalar('loss_f', data=loss_f.item(), step=global_step)
            # tf.summary.scalar('psnr_f', data=psnr_f.item(), step=global_step)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        #     print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
        #     print('iter time {:.05f}'.format(dt))
        #
        # with tf.compat.v2.summary.record_summaries_every_n_global_steps(args.i_print):
        #     tf.compat.v2.summary.scalar(name='loss', data=loss, step=tf.compat.v1.train.get_or_create_global_step())
        #     tf.compat.v2.summary.scalar(name='psnr', data=psnr, step=tf.compat.v1.train.get_or_create_global_step())
        #     tf.compat.v2.summary.histogram(name='tran', data=trans, step=tf.compat.v1.train.get_or_create_global_step())
        #     if args.N_importance > 0:
        #         tf.compat.v2.summary.scalar(name='psnr0', data=psnr0, step=tf.compat.v1.train.get_or_create_global_step())
        #
        # if i % args.i_img == 0:
        #
        #     # Log a rendered validation view to Tensorboard
        #     img_i = np.random.choice(i_val)
        #     target = images[img_i]
        #     pose = poses[img_i, :3, :4]
        #     with torch.no_grad():
        #         rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
        #                                         **render_kwargs_test)
        #
        #     psnr = mse2psnr(img2mse(rgb, target))
        #
        #     with tf.compat.v2.summary.record_summaries_every_n_global_steps(args.i_img):
        #
        #         tf.compat.v2.summary.image(name='rgb', data=to8b(rgb)[tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #         tf.compat.v2.summary.image(name='disp', data=disp[tf.newaxis, ..., tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #         tf.compat.v2.summary.image(name='acc', data=acc[tf.newaxis, ..., tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #
        #         tf.compat.v2.summary.scalar(name='psnr_holdout', data=psnr, step=tf.compat.v1.train.get_or_create_global_step())
        #         tf.compat.v2.summary.image(name='rgb_holdout', data=target[tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #
        #     if args.N_importance > 0:
        #         with tf.compat.v2.summary.record_summaries_every_n_global_steps(args.i_img):
        #             tf.compat.v2.summary.image(name='rgb0', data=to8b(extras['rgb0'])[tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #             tf.compat.v2.summary.image(name='disp0', data=extras['disp0'][tf.newaxis, ..., tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())
        #             tf.compat.v2.summary.image(name='z_std', data=extras['z_std'][tf.newaxis, ..., tf.newaxis], step=tf.compat.v1.train.get_or_create_global_step())

        torch.cuda.empty_cache()
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
