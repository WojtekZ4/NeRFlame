"""Trainer to train Flame-Nerf, based on Nerf"""

import numpy as np
import os
import imageio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flame_nerf.nerf_pytorch.trainer import Trainer
from flame_nerf.nerf_pytorch import NeRF
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

from flame_nerf.utils import load_obj_from_config
from FLAME import FLAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlameTrainer(Trainer):

    def __init__(
            self,
            epsilon: float,
            fake_epsilon: float,
            trans_the_smallest_epsilon: float,
            trans_the_biggest_epsilon: float,
            n_the_farthest_samples: int,
            n_central_samples: int,
            n_additional_samples: int,
            flame_config,
            chunk_render,
            enhanced_mode_start_iter=None,
            enhanced_mode_stop_iter=None,
            **kwargs
    ):
        """
        n_additional_samples - in fact it is n_additional_samples-1,
        and n_additional_samples should be even.
        """
        self.epsilon = epsilon
        self.fake_epsilon = fake_epsilon
        self.trans_the_smallest_epsilon = trans_the_smallest_epsilon
        self.trans_the_biggest_epsilon = trans_the_biggest_epsilon
        self.n_the_farthest_samples = n_the_farthest_samples
        self.n_central_samples = n_central_samples
        self.n_additional_samples = n_additional_samples
        self.enhanced_mode_start_iter = enhanced_mode_start_iter
        self.enhanced_mode_stop_iter = enhanced_mode_stop_iter

        self.enhanced_mode = True
        self.chunk_render = chunk_render

        super().__init__(
            **kwargs
        )

        # flame model
        flame_config = load_obj_from_config(cfg=flame_config)
        self.model_flame = FLAME(flame_config).to(self.device)

        self.f_shape = nn.Parameter(torch.zeros(1, 100).float().to(self.device) / 100)
        self.f_exp = nn.Parameter(torch.zeros(1, 50).float().to(self.device))
        self.f_pose = nn.Parameter(torch.zeros(1, 6).float().to(self.device))
        self.f_neck_pose = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.f_trans = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.vertices_mal = nn.Parameter(8 * torch.ones(1, 1).float().to(self.device))

        f_lr = 0.001
        f_wd = 0.0001
        self.f_opt = torch.optim.Adam(
            params=[
                self.f_shape,
                self.f_exp,
                self.f_pose,
                self.f_neck_pose,
                self.f_trans,
                # self.vertices_mal
            ],
            lr=f_lr,
            weight_decay=f_wd
        )

        self.faces = self.flame_faces()
        self.vertices = None

        self.trans_epsilon_modifier = 1

        if self.enhanced_mode:
            self.minimum_trans_epsilon = self.trans_the_biggest_epsilon/self.trans_the_smallest_epsilon
        self.trans_epsilon = self.trans_the_smallest_epsilon * self.trans_epsilon_modifier

    def update_trans_epsilon(self):
        if self.enhanced_mode:
            if self.global_step > self.enhanced_mode_start_iter:
                trans_eps_diff = (1 - self.minimum_trans_epsilon) * (
                        1 - (self.global_step - self.enhanced_mode_start_iter) /
                        (self.enhanced_mode_stop_iter - self.enhanced_mode_start_iter)
                )
                self.trans_epsilon_modifier = self.minimum_trans_epsilon + trans_eps_diff
                self.trans_epsilon = self.trans_the_smallest_epsilon * self.trans_epsilon_modifier

    def _train_prepare(self, n_iters):
        self.N_iters = n_iters
        if self.enhanced_mode:
            if self.enhanced_mode_start_iter is None:
                self.enhanced_mode_start_iter = 1
            if self.enhanced_mode_stop_iter is None:
                self.enhanced_mode_stop_iter = n_iters

    def flame_vertices_test(
            self, f_shape, f_exp, f_pose, f_neck_pose, f_trans
    ):
        with torch.no_grad():
            vertices, _ = self.model_flame(
                f_shape, f_exp, f_pose,
                neck_pose=f_neck_pose, transl=f_trans
            )
            vertices = torch.squeeze(vertices)
            # vertices = vertices.cuda()

            vertices = vertices[:, [0, 2, 1]]
            vertices[:, 1] = -vertices[:, 1]
            vertices *= self.vertices_mal

            return vertices

    def flame_vertices(self):
        """
        Return mesh vertices using FLAME model
        """
        vertices, _ = self.model_flame(
            self.f_shape, self.f_exp, self.f_pose,
            neck_pose=self.f_neck_pose, transl=self.f_trans
        )
        vertices = torch.squeeze(vertices)
        # vertices = vertices.cuda()

        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 1] = -vertices[:, 1]
        vertices *= self.vertices_mal

        if self.tensorboard_logging:
            self.log_on_tensorboard(
                self.global_step,
                {
                    'train': {
                        'vertices_mal': self.vertices_mal,
                    }
                }
            )

        return vertices

    def flame_faces(self):
        """
        Return faces (vertices) indexes using FLAME model.
        """
        faces = self.model_flame.faces
        faces = torch.tensor(faces.astype(np.int32))
        faces = torch.squeeze(faces)
        faces = faces.cuda()
        return faces

    def create_nerf_model(self):
        """
        Create Nerf model based on https://github.com/yenchenlin/nerf-pytorch"
        implementation. Add additional atributes used to rendering.
        """
        optimizer, render_kwargs_train, render_kwargs_test = self._create_nerf_model(
            model=NeRF
        )

        additional_render_kwargs_train = {
            'epsilon': self.epsilon,
            'fake_epsilon': self.fake_epsilon,
            'trans_the_smallest_epsilon': self.trans_the_smallest_epsilon,
            'trans_the_biggest_epsilon': self.trans_the_biggest_epsilon,
            'enhanced_mode': self.enhanced_mode,
            'trans_epsilon_modifier': self.trans_epsilon_modifier,
            'trans_epsilon': self.trans_epsilon,
            'n_the_farthest_samples': self.n_the_farthest_samples,
            'n_central_samples': self.n_central_samples,
            'n_additional_samples': self.n_additional_samples,
            'trans_mat': None
        }

        for i in additional_render_kwargs_train:
            render_kwargs_train[i] = additional_render_kwargs_train[i]
            render_kwargs_test[i] = additional_render_kwargs_train[i]

        return optimizer, render_kwargs_train, render_kwargs_test

    def selects_closest_points_to_mesh(self, z_vals, distance_between_pts_and_mesh):
        # take n_central_samples the closest vertices
        # to create n_additional_points arround them
        # wybierz n_central_samples  niemiejszych odleglosci  dla kazdego raya
        # dla kazdefo reya dostenimy 10 najblizszych punktow/odleglosci do mesha
        best_vert, best_indexes = torch.topk(
            distance_between_pts_and_mesh,
            self.n_central_samples, dim=1, largest=False
        )

        # wybierz z_vals dla centralnych punktow wokol ktorych bedziemy robic dodatkowe punkty
        z_values_to_closest_to_mesh = torch.gather(z_vals, 1, best_indexes)

        return z_values_to_closest_to_mesh

    def create_additional_points(
        self,
        rays_o, rays_d,
        z_vals,
        vertices,
        distance_between_pts_and_mesh,
        distance_between_points
    ):

        z_values_to_closest_to_mesh = self.selects_closest_points_to_mesh(
            z_vals=z_vals,
            distance_between_pts_and_mesh=distance_between_pts_and_mesh
        )

        n_rays = z_values_to_closest_to_mesh.shape[0]
        # bedziemy teraz budowac greeda wokol punktow centralnych
        # dla kazedgo raya, dla kazdego centrlanego punktu powieksz wektor  self.n_additional_samples - 1 razy
        expanded_distance_between_pts_and_mesh = z_values_to_closest_to_mesh.unsqueeze(2).expand(
            n_rays,
            z_values_to_closest_to_mesh.shape[1],
            self.n_additional_samples - 1
        )

        # stworz otoczke wokol punktow (z_vals) expanded_tensor
        # ignoruj ostatni punkt
        new_dist_between_points = torch.linspace(
            -distance_between_points * 0.5,
            distance_between_points * 0.5,
            self.n_additional_samples
        )[:-1]

        # stworz nowe z_vals dla additional points na podstawie centralnych z_vasl (expanded_tensor) i roznicy step_tensor
        additional_z_vals = expanded_distance_between_pts_and_mesh + new_dist_between_points
        additional_z_vals = additional_z_vals.reshape(n_rays, -1)

        # [N_rays, N_samples, 3]
        _additional_z_vals = additional_z_vals[..., :, None]  # poszerz [N_rays, n_central * n_additiona, 1]

        # stworz punkty additional
        additional_pts = rays_o[..., None, :] + rays_d[..., None, :] * _additional_z_vals

        # policz dystance do mesha tych dodatkowych punktow
        additional_distance_f, additional_idx_f = flame_based_alpha_calculator_3_face_version(
            additional_pts, vertices,
            self.faces
        )

        return additional_z_vals, additional_pts, additional_distance_f

    @staticmethod
    def _concat_selected_points_with_additional_points(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        selected_points_indexes: torch.Tensor,
        selected_points_distance_f: torch.Tensor,
        additional_z_vals: torch.Tensor,
        additional_distance_f: torch.Tensor,
    ):
        z_vals_without_farthest = torch.gather(z_vals, 1, selected_points_indexes)
        _z_vals = torch.hstack([z_vals_without_farthest, additional_z_vals])

        distances_f = torch.hstack([selected_points_distance_f, additional_distance_f])
        z_vals, z_vals_sorted_indexes = torch.sort(_z_vals, dim=1)
        distances_f = torch.gather(distances_f, 1, z_vals_sorted_indexes)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals, distances_f

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
        vertices = self.vertices

        # calculate distance to mesh
        distances_f, idx_f = flame_based_alpha_calculator_3_face_version(
            pts, vertices, self.faces
        )

        # near_v -- near_value
        near_v, far_v = near[0, 0].item(), far[0, 0].item()

        # sredni dystans pomiedzy punktami.
        dist_between_points = ((far_v - near_v) / (N_samples - 1))

        additional_z_vals, additional_pts, additional_distance_f = self.create_additional_points(
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
        pts, z_vals, distances_f = self._concat_selected_points_with_additional_points(
            rays_o=rays_o,
            rays_d=rays_d,
            z_vals=z_vals,
            selected_points_indexes=selected_points_indexes,
            selected_points_distance_f=selected_points_distance_f,
            additional_z_vals=additional_z_vals,
            additional_distance_f=additional_distance_f
        )

        if self.f_trans is not None:
            pts += self.f_trans

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
            trans_alpha = flame_based_alpha_calculator_f_relu(distances_f, m, self.trans_epsilon)

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

            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
            fake_alpha = alpha

            alpha = torch.minimum(alpha, trans_alpha)

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

            #relu = torch.nn.ReLU()
            distances_f, idx_f = flame_based_alpha_calculator_3_face_version(
                pts, self.vertices, self.faces
            )

            run_fn = network_fn if network_fine is None else network_fine
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, _, _, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest,
                distances_f=distances_f,
                enhanced_mode=self.enhanced_mode
            )

        return rgb_map_0, disp_map_0, acc_map_0, rgb_map, disp_map, acc_map, raw, z_samples

    def core_optimization_loop(
            self,
            optimizer, render_kwargs_train,
            batch_rays, i, target_s,
    ):
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
        self.f_opt.step()

        return trans, loss, psnr, psnr0

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
                    'epsilon': render_kwargs_test['epsilon'],
                    'fake_epsilon': render_kwargs_test['fake_epsilon'],
                    'trans_epsilon': render_kwargs_test['trans_epsilon'],
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
                    'epsilon': render_kwargs_test['epsilon'],
                    'fake_epsilon': render_kwargs_test['fake_epsilon'],
                    'trans_epsilon': render_kwargs_test['trans_epsilon'],
                }, path)
            print('Saved checkpoints at', path)

        torch.cuda.empty_cache()

        if i % self.i_testset == 0 and i > 0:
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

                self.log_on_tensorboard(
                    i,
                    {
                        'test': {
                            'loss': img_loss,
                            'psnr': img_psnr_1,
                            'img_ssim': img_ssim,
                            'img_lpips': img_lpips
                        }
                    }
                )

                outstats_path = os.path.join(testsavedir, 'stats.txt')
                with open(outstats_path, 'w') as fp:
                    for s, v in {"img_loss": img_loss, "img_psnr": img_psnr_1, "img_ssim": img_ssim,
                                 "img_lpips": img_lpips}.items():
                        fp.write('%s %f\n' % (s, v))

            print('Saved test set')

        if i % self.i_testset == 0 and i > 0:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{:06d}_rot1'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            radian = np.pi / 180.0
            with torch.no_grad():
                f_pose_rot = self.f_pose.clone().detach()
                f_pose_rot[0, 3] = 10.0 * radian

                vertice = self.flame_vertices_test(
                    self.f_shape, self.f_exp, f_pose_rot, self.f_neck_pose, self.f_trans
                )

                outmesh_path = os.path.join(testsavedir, 'face.obj')
                write_simple_obj(mesh_v=vertice.detach().cpu().numpy(), mesh_f=self.faces,
                                 filepath=outmesh_path)

                print('test poses shape', poses[i_test].shape)
                triangles_org = self.vertices[self.faces.long(), :]
                triangles_out = vertice[self.faces.long(), :]

                render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out, triangles_org)
                rgbs, disps = render_path(poses[i_test], hwf, self.K, self.chunk_render,
                                          render_kwargs_test,
                                          gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor)
                render_kwargs_test['trans_mat'] = None
            print('Saved test set')

        if i % self.i_testset == 0 and i > 0:
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
                rgbs, disps = render_path(
                    torch.Tensor(poses[i_test]).to(device), hwf, self.K, self.chunk_render,
                    render_kwargs_test,
                    gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor
                )
                render_kwargs_test['trans_mat'] = None
            print('Saved test set')

        torch.cuda.empty_cache()
        if i % self.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk_render,
                                          render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_f_{:06d}_'.format(self.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        torch.cuda.empty_cache()
