import numpy as np
import os
import imageio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flame_nerf.trainer import FlameTrainer
from copy import deepcopy

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flame_nerf.face_utils import (
    flame_based_alpha_calculator_f_relu,
    flame_based_alpha_calculator_3_face_version,
    recover_homogenous_affine_transformation,
    write_simple_obj, transform_pt
)
from flame_nerf.nerf_pytorch.nerf_utils import (
    render,
    render_path
)

from flame_nerf.nerf_pytorch.run_nerf_helpers import (
    img2mse,
    mse2psnr, to8b
)

from flame_nerf.mesh_utils import intersection_points_on_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrozenFlameTrainer(FlameTrainer):

    def __init__(
            self,
            enhanced_mode_freeze=None,
            **kwargs
    ):
        self.enhanced_mode_freeze = enhanced_mode_freeze

        self.remove_rays = False

        super().__init__(
            **kwargs
        )

    def _train_prepare(self, n_iters):
        self.N_iters = n_iters
        if self.enhanced_mode:
            if self.enhanced_mode_start_iter is None:
                self.enhanced_mode_start_iter = self.enhanced_mode_freeze
            if self.enhanced_mode_stop_iter is None:
                self.enhanced_mode_stop_iter = n_iters

    def core_optimization_loop(
            self,
            optimizer, render_kwargs_train,
            batch_rays, i, target_s,
    ):
        if self.global_step > self.enhanced_mode_freeze: 
            self.update_trans_epsilon()

        self.vertices = self.flame_vertices()

        rgb, disp, acc, extras = render(
            self.H, self.W, self.K,
            chunk=self.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train
        )

        optimizer.zero_grad()
        self.f_opt.zero_grad()

        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        psnr0 = None
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()
        if self.enhanced_mode_freeze is not None:
            if self.global_step <= self.enhanced_mode_freeze: 
                self.f_opt.step()
        else:
            self.f_opt.step()
        
        return trans, loss, psnr, psnr0

    def sample_main_points(
        self,
        near: float,
        far: float,
        perturb: float,
        N_rays: int,
        N_samples: int,
        viewdirs: torch.Tensor,
        network_fn,
        network_query_fn,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float,
        white_bkgd: bool,
        pytest: bool,
        lindisp: bool,
        **kwargs
    ):
        """
        Sample and run NERF model to predict rgb
        """

        pts, z_vals = self._sample_main_points(
            rays_o, rays_d, near, far, N_rays, N_samples, perturb, pytest, lindisp
        )

        # take coord vertices from mesh
        if self.vertices is None:
            self.vertices = self.flame_vertices()
        vertices = self.vertices
        if 'test_vertices' in kwargs:
            if kwargs['test_vertices'] is not None:
                vertices = kwargs['test_vertices']

        # calculate distance to mesh
        distances_f, idx_f = flame_based_alpha_calculator_3_face_version(
            pts, vertices, self.faces
        )

        # near_v -- near_value
        near_v, far_v = near[0, 0].item(), far[0, 0].item()

        # sredni dystans pomiedzy punktami.
        dist_between_points = ((far_v - near_v) / (N_samples - 1))

        (additional_z_vals, additional_pts,
         additional_distance_f, additional_idx_f) = self.create_additional_points(
            rays_o=rays_o,
            rays_d=rays_d,
            z_vals=z_vals,
            vertices=vertices,
            distance_between_pts_and_mesh=distances_f,
            distance_between_points=dist_between_points
        )

        # dla kazdefo reya dostenimy n najblizszych punktow/odleglosci do mesha
        # te kotre po prostu zostaną a bedziemy wywalac te n_the_farthest_samples
        selected_points_distance_f, selected_points_indexes = torch.topk(
            distances_f, self.N_samples - self.n_the_farthest_samples,
            dim=1, largest=False
        )

        # concat nearest(selected) samples with additional points
        pts, z_vals, distances_f, idx_f = self._concat_selected_points_with_additional_points(
            rays_o=rays_o,
            rays_d=rays_d,
            z_vals=z_vals,
            idx_f=idx_f,
            additional_idx_f=additional_idx_f,
            selected_points_indexes=selected_points_indexes,
            selected_points_distance_f=selected_points_distance_f,
            additional_z_vals=additional_z_vals,
            additional_distance_f=additional_distance_f
        )

        if self.f_trans is not None:
            pts += self.f_trans

        if self.remove_rays:
            mass_point = torch.quantile(vertices, 0.75, dim=0)
            ray_idxs, intersection_points = intersection_points_on_mesh(self.faces, vertices, rays_o, rays_d)
            mask_front = torch.where(intersection_points[:, 1] > mass_point[1].item(), False, True)
            mask_intersect = torch.zeros(pts.shape[0], dtype=torch.bool)
            mask_intersect[ray_idxs] = True
            self.mask = torch.logical_and(mask_front, mask_intersect)
            self.mask = self.mask.unsqueeze(-1)

        if kwargs['trans_mat'] is not None:
            trans_mat_organ = kwargs['trans_mat'][idx_f, :]
            pts = transform_pt(pts, trans_mat_organ)

        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, fake_weights, depth_map = self.raw2outputs(
            raw=raw, z_vals=z_vals, rays_d=rays_d, raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd,
            pytest=pytest, distances_f=distances_f,
        )

        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, fake_weights, raw

    def raw2outputs(
            self,
            raw: torch.Tensor,
            z_vals: torch.Tensor,
            rays_d: torch.Tensor,
            raw_noise_std=0.0,
            white_bkgd=False,
            pytest=False,
            **kwargs
    ):
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

        distances_f = kwargs["distances_f"]
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        m = torch.nn.ReLU()
        if self.enhanced_mode is False:
            alpha = flame_based_alpha_calculator_f_relu(distances_f, m, self.epsilon)
            fake_alpha = flame_based_alpha_calculator_f_relu(distances_f, m, self.fake_epsilon)
        else:              
            # calc alpha from nerf
            raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
            dists = z_vals[..., 1:] - z_vals[..., :-1]

            # [N_rays, N_samples]
            dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
            dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(raw[..., 3].shape) * raw_noise_std

                # Overwrite randomly sampled data if pytest
                if pytest:
                    np.random.seed(0)
                    noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                    noise = torch.Tensor(noise)

            if self.global_step <= self.enhanced_mode_freeze:
                alpha = flame_based_alpha_calculator_f_relu(distances_f, m, self.trans_epsilon)
            else:
                alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
                alpha = (distances_f <= self.trans_epsilon) * alpha

            fake_alpha = alpha

            if self.remove_rays:
                alpha = alpha * self.mask
                fake_alpha = fake_alpha * self.mask
                self.mask = None

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]

        fake_weights = fake_alpha * torch.cumprod(
            torch.cat([torch.ones((fake_alpha.shape[0], 1)), 1. - fake_alpha + 1e-10], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, fake_weights, depth_map

    def sample_points(
            self,
            z_vals: torch.Tensor,
            weights: torch.Tensor,
            perturb: float,
            pytest: bool,
            rays_d: torch.Tensor,
            rays_o: torch.Tensor,
            rgb_map,
            disp_map,
            acc_map,
            network_fn,
            network_fine,
            network_query_fn,
            viewdirs: torch.Tensor,
            raw_noise_std: float,
            white_bkgd: bool,
            **kwargs
    ):

        rgb_map_0, disp_map_0, acc_map_0 = None, None, None
        raw = None
        z_samples = None

        if self.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples, pts = self._sample_points(
                z_vals_mid=z_vals_mid,
                weights=weights,
                perturb=perturb,
                pytest=pytest,
                rays_o=rays_o,
                rays_d=rays_d,
            )

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            vertices = self.vertices
            if 'test_vertices' in kwargs:
                if kwargs['test_vertices'] is not None:
                    vertices = kwargs['test_vertices']

            #relu = torch.nn.ReLU()
            distances_f, idx_f = flame_based_alpha_calculator_3_face_version(
                pts, vertices, self.faces
            )

            if self.f_trans is not None:
                pts += self.f_trans

            if self.remove_rays:
                mass_point = torch.quantile(vertices, 0.75, dim=0)
                ray_idxs, intersection_points = intersection_points_on_mesh(self.faces, vertices, rays_o, rays_d)
                mask_front = torch.where(intersection_points[:, 1] > mass_point[1].item(), False, True)
                mask_intersect = torch.zeros(pts.shape[0], dtype=torch.bool)
                mask_intersect[ray_idxs] = True
                self.mask = torch.logical_and(mask_front, mask_intersect)
                self.mask = self.mask.unsqueeze(-1)

            if kwargs['trans_mat'] is not None:
                trans_mat_organ = kwargs['trans_mat'][idx_f, :]
                pts = transform_pt(pts, trans_mat_organ)

            run_fn = network_fn if network_fine is None else network_fine
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, _, _, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest,
                distances_f=distances_f,
                enhanced_mode=self.enhanced_mode
            )

        return rgb_map_0, disp_map_0, acc_map_0, rgb_map, disp_map, acc_map, raw, z_samples

    def rest_is_logging(
            self, i, render_poses, hwf, poses, i_test, images,
            render_kwargs_train,
            render_kwargs_test,
            optimizer,
            **kwargs
    ):
        if i % self.i_weights == 0:
            path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(i))
            if render_kwargs_train['network_fine'] is None:
                torch.save({
                    'global_step': self.global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f_optimizer_state_dict': self.f_opt.state_dict(),
                    'f_shape': self.f_shape,
                    'f_exp': self.f_exp,
                    'f_pose': self.f_pose,
                    'f_trans': self.f_trans,
                    'f_neck_pose': self.f_neck_pose,
                    'epsilon': self.epsilon,
                    'fake_epsilon': self.fake_epsilon,
                    'trans_epsilon': self.trans_epsilon,
                }, path)
            else:
                torch.save({
                    'global_step': self.global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f_optimizer_state_dict': self.f_opt.state_dict(),
                    'f_shape': self.f_shape,
                    'f_exp': self.f_exp,
                    'f_pose': self.f_pose,
                    'f_trans': self.f_trans,
                    'f_neck_pose': self.f_neck_pose,
                    'epsilon': self.epsilon,
                    'fake_epsilon': self.fake_epsilon,
                    'trans_epsilon': self.trans_epsilon,
                }, path)
            print('Saved checkpoints at', path)

        torch.cuda.empty_cache()
        if i % self.i_testset == 0 and i > 0:
            self.render_testset(i=i, render_poses=render_poses, hwf=hwf,
                poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        if i % self.i_testset == 0 and i > 0:
            self.render_rot1(i=i, render_poses=render_poses, hwf=hwf,
                poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        if i % self.i_testset == 0 and i > 0:
            self.render_rot2(i=i, render_poses=render_poses, hwf=hwf,
                poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        if i % self.i_video == 0 and i > 0:
            self.render_video(i=i, render_poses=render_poses, hwf=hwf,
            render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        
    def render_testset(self, i, render_poses, hwf,
            poses, i_test, images,
            render_kwargs_test
    ):
        if isinstance(i, str):
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{}'.format(i))
        else:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            vertice = self.flame_vertices()

            outmesh_path = os.path.join(testsavedir, 'face.obj')
            write_simple_obj(mesh_v=vertice.detach().cpu().numpy(), mesh_f=self.faces,
                                filepath=outmesh_path)

            print('test poses shape', poses[i_test].shape)

            rgbs, _ = render_path(
                poses[i_test], hwf, self.K, self.chunk_render,
                render_kwargs_test,
                gt_imgs=images[i_test],
                savedir=testsavedir,
                render_factor=self.render_factor,
            )

            images_o = torch.tensor(images[i_test]).to(device=device, dtype=torch.float)
            rgbs = torch.tensor(rgbs).to(device=device, dtype=torch.float)

            images_o = torch.movedim(images_o, 3, 1)
            rgbs = torch.movedim(rgbs, 3, 1)

            psnr_f = PeakSignalNoiseRatio()
            img_psnr_1 = psnr_f(rgbs, images_o)

            img_loss = img2mse(rgbs, images_o)

            ssim_f = StructuralSimilarityIndexMeasure()
            img_ssim = ssim_f(rgbs, images_o)

            lpips_f = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
            img_lpips = lpips_f(rgbs, images_o)

            print("img_loss", img_loss)
            print("img_psnr_1", img_psnr_1)
            print("img_ssim", img_ssim)
            print("img_lpips", img_lpips)

            if isinstance(i, int):
                self.log_on_tensorboard(
                    i,
                    {
                        'test': {
                            'loss': img_loss,
                            'psnr': img_psnr_1,
                            'img_ssim': img_ssim,
                            'img_lpips': img_lpips,
                            'trans_eps': self.trans_epsilon
                        }
                    }
                )

            outstats_path = os.path.join(testsavedir, 'stats.txt')
            with open(outstats_path, 'w') as fp:
                for s, v in {"img_loss": img_loss, "img_psnr": img_psnr_1, "img_ssim": img_ssim,
                                "img_lpips": img_lpips, 'trans_eps': self.trans_epsilon,
                                "min_exp": self.f_exp.min().item(), "max_exp": self.f_exp.max().item()}.items():
                    fp.write('%s %f\n' % (s, v))

        print('Saved test set')

    def render_rot1(self, i, render_poses, hwf,
            poses, i_test, images,
            render_kwargs_test
    ):
        if isinstance(i, str):
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{}_rot1'.format(i))
        else:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{:06d}_rot1'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        radian = np.pi / 180.0
        with torch.no_grad():
            f_pose_rot = self.f_pose.clone().detach()
            f_pose_rot[0, 3] = self.mul * radian

            vertice = self.flame_vertices_test(
                self.f_shape, self.f_exp, f_pose_rot, self.f_neck_pose, self.f_trans
            )

            outmesh_path = os.path.join(testsavedir, 'face.obj')
            write_simple_obj(mesh_v=vertice.detach().cpu().numpy(), mesh_f=self.faces,
                                filepath=outmesh_path)

            print('test poses shape', poses[i_test].shape)
            triangles_org = self.vertices[self.faces.long(), :]
            triangles_out = vertice[self.faces.long(), :]

            trans_mat = recover_homogenous_affine_transformation(triangles_out, triangles_org) # transform matrix

            render_kwargs_test['trans_mat'] = trans_mat
            render_kwargs_test['test_vertices'] = vertice
            
            self.remove_rays = True
            rgbs, disps = render_path(poses[i_test], hwf, self.K, self.chunk_render,
                                        render_kwargs_test,
                                        gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor)
            self.remove_rays = False

            render_kwargs_test['trans_mat'] = None
            render_kwargs_test['test_vertices'] = None
        print('Saved test set')

    def render_rot2(self, i, render_poses, hwf,
            poses, i_test, images,
            render_kwargs_test
    ):
        if isinstance(i, str):
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{}_rot2'.format(i))
        else:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{:06d}_rot2'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        radian = np.pi / 180.0
        with torch.no_grad():
            out_neck_pose = nn.Parameter(torch.zeros(1, 3).float().to(device))
            out_neck_pose[0, 1] = 20.0 * radian

            vertice = self.flame_vertices_test(
                self.f_shape, self.f_exp, self.f_pose, out_neck_pose, self.f_trans
            )

            outmesh_path = os.path.join(testsavedir, 'face.obj')
            write_simple_obj(mesh_v=vertice.detach().cpu().numpy(), mesh_f=self.faces,
                                filepath=outmesh_path)

            print('test poses shape', poses[i_test].shape)
            triangles_org = self.vertices[self.faces.long(), :]
            triangles_out = vertice[self.faces.long(), :]

            render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out, triangles_org)
            render_kwargs_test['test_vertices'] = vertice

            self.remove_rays = True
            rgbs, disps = render_path(
                torch.Tensor(poses[i_test]).to(device), hwf, self.K, self.chunk_render,
                render_kwargs_test,
                gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor
            )
            self.remove_rays = False

            render_kwargs_test['trans_mat'] = None
            render_kwargs_test['test_vertices'] = None

        print('Saved test set')
    
    def render_video(self, i, render_poses, hwf,
            render_kwargs_test
    ):
        # Turn on testing mode
        with torch.no_grad():
            self.remove_rays = True
            rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk_render,
                                        render_kwargs_test)
            self.remove_rays = False

        print('Done, saving', rgbs.shape, disps.shape)
        if isinstance(i, str):
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_f_{}_'.format(self.expname, i))
        else:
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_f_{:06d}_'.format(self.expname, i))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    def render_load(self):
        hwf, poses, i_test, i_val, i_train, images, render_poses = self.load_data()

        poses = torch.Tensor(poses).to(device)

        if self.render_test:
            render_poses = np.array(poses[i_test])
            render_poses = torch.Tensor(render_poses).to(self.device)

        hwf = self.cast_intrinsics_to_right_types(hwf=hwf)
        self.create_log_dir_and_copy_the_config_file()
        _, render_kwargs_train, render_kwargs_test = self.create_nerf_model()

        self._train_prepare(40001)

        self.vertices = self.flame_vertices()

        print(self.trans_epsilon)

        torch.cuda.empty_cache()
        for j, val in enumerate(torch.arange(-6, 6)):
            self.mul = val
            # i="pose0"
            # self.render_testset(i=i, render_poses=render_poses, hwf=hwf,
            #     poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

            i="remove_rays" + str(j)
            torch.cuda.empty_cache()
            self.render_rot1(i=i, render_poses=render_poses, hwf=hwf,
                poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)
            
            # torch.cuda.empty_cache()
            # self.render_rot2(i=i, render_poses=render_poses, hwf=hwf,
            #     poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        # torch.cuda.empty_cache()
        # self.render_video(i=i, render_poses=render_poses, hwf=hwf,
        # render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()