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
    mse2psnr, to8b,
    get_embedder, NeRF
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrozenFlameTrainer(FlameTrainer):

    def __init__(
            self,
            enhanced_mode_switch=None,
            eps_increase_multiplier=None,
            additional_samples=False,
            samples_multiplier=None,
            use_pretrained_nerf=False,
            pretrained_save_iter=5000,
            pretrained_nerf_path=None,
            **kwargs
    ):
        self.enhanced_mode_switch = enhanced_mode_switch
        self.eps_increase_multiplier = eps_increase_multiplier
        self.additional_samples = additional_samples
        self.samples_multiplier = samples_multiplier
        self.use_pretrained_nerf = use_pretrained_nerf
        self.pretrained_save_iter = pretrained_save_iter
        self.pretrained_nerf_path = pretrained_nerf_path

        self.remove_rays = False

        super().__init__(
            **kwargs
        )

        if self.use_pretrained_nerf:
            if self.pretrained_nerf_path is not None:
                self.load_pretrained_nerf()
            else:
                self.pretrained_nerf = None
                self.pretrained_nerf_fine = None

    def load_pretrained_nerf(self):
        _, input_ch = get_embedder(self.multires, self.i_embed)

        input_ch_views = 0
        if self.use_viewdirs:
            _, input_ch_views = get_embedder(self.multires_views, self.i_embed)
        output_ch = 5 if self.N_importance > 0 else 4
        skips = [4]
        ckpt = torch.load(self.pretrained_nerf_path)

        self.pretrained_nerf = NeRF(D=self.netdepth, W=self.netwidth,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=self.use_viewdirs).to(device)
        self.pretrained_nerf.load_state_dict(ckpt['network_fn_state_dict'])

        self.pretrained_nerf_fine = None
        if self.N_importance > 0:
            self.pretrained_nerf_fine = NeRF(D=self.netdepth_fine, W=self.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=self.use_viewdirs).to(device)
            self.pretrained_nerf_fine.load_state_dict(ckpt['network_fine_state_dict'])

        self.old_trans_epsilon = ckpt['trans_epsilon']
        for param in self.pretrained_nerf.parameters():
            param.requires_grad = False
        if self.pretrained_nerf_fine is not None:
            for param in self.pretrained_nerf_fine.parameters():
                param.requires_grad = False 

    def update_trans_epsilon(self):
        if self.enhanced_mode:
            if self.eps_increase_multiplier is None:
                if self.global_step > self.enhanced_mode_start_iter:
                    trans_eps_diff = (1 - self.minimum_trans_epsilon) * (
                            1 - (self.global_step - self.enhanced_mode_start_iter) /
                            (self.enhanced_mode_stop_iter - self.enhanced_mode_start_iter)
                    )
                    self.trans_epsilon_modifier = self.minimum_trans_epsilon + trans_eps_diff
                    self.trans_epsilon = self.trans_the_smallest_epsilon * self.trans_epsilon_modifier
            else:
                if self.enhanced_mode_switch_iter >= self.global_step > self.enhanced_mode_start_iter:
                    trans_eps_diff = (1 - self.minimum_trans_epsilon) * (
                            1 - (self.global_step - self.enhanced_mode_start_iter) /
                            (self.enhanced_mode_switch_iter - self.enhanced_mode_start_iter)
                    )
                    self.trans_epsilon_modifier = self.minimum_trans_epsilon + trans_eps_diff
                    self.trans_epsilon = self.trans_the_smallest_epsilon * self.trans_epsilon_modifier
                elif self.global_step > self.enhanced_mode_switch_iter:
                    trans_eps_mul = (self.eps_increase_multiplier - 1) / (self.enhanced_mode_stop_iter - self.enhanced_mode_switch_iter)
                    self.trans_epsilon = self.trans_the_biggest_epsilon * (1 + trans_eps_mul * (self.global_step - self.enhanced_mode_switch_iter))                   

    def _train_prepare(self, n_iters):
        self.N_iters = n_iters
        if self.enhanced_mode:
            if self.enhanced_mode_start_iter is None:
                self.enhanced_mode_start_iter = 1
            if self.enhanced_mode_stop_iter is None:
                self.enhanced_mode_stop_iter = n_iters
            if self.enhanced_mode_switch is not None:
                self.enhanced_mode_switch_iter = self.enhanced_mode_start_iter \
                    + self.enhanced_mode_switch * (self.enhanced_mode_stop_iter - self.enhanced_mode_start_iter)

    def update_sample_values(self, train_phase=True):
        if self.enhanced_mode:
            if train_phase:
                if self.additional_samples and self.global_step == self.enhanced_mode_switch_iter:
                    self.n_central_samples *= self.samples_multiplier
                    self.n_additional_samples *= self.samples_multiplier
            else:
                self.n_central_samples *= self.samples_multiplier
                self.n_additional_samples *= self.samples_multiplier

    def core_optimization_loop(
            self,
            optimizer, render_kwargs_train,
            batch_rays, i, target_s,
    ):
        if self.use_pretrained_nerf and i == self.pretrained_save_iter and self.pretrained_nerf_path is None:
            self.pretrained_nerf = deepcopy(render_kwargs_train["network_fn"])
            self.pretrained_nerf_fine = deepcopy(render_kwargs_train["network_fine"])
            self.old_trans_epsilon = self.trans_epsilon
            for param in self.pretrained_nerf.parameters():
                param.requires_grad = False
            if self.pretrained_nerf_fine is not None:
                for param in self.pretrained_nerf_fine.parameters():
                    param.requires_grad = False

        self.update_trans_epsilon()
        self.vertices = self.flame_vertices()
        self.update_sample_values()

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
        if self.enhanced_mode_switch is not None:
            if self.global_step <= self.enhanced_mode_switch_iter: 
                self.f_opt.step()
        else:
            self.f_opt.step()
        torch.clamp(self.f_exp, min=-1.9, max=1.9)
        
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

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                network_fn_tmp = render_kwargs_test["network_fn"]
                network_fine_tmp = render_kwargs_test["network_fine"]
                trans_epsilon_tmp = self.trans_epsilon
                render_kwargs_test["network_fn"] = self.pretrained_nerf
                render_kwargs_test["network_fine"] = self.pretrained_nerf_fine
                self.trans_epsilon = self.old_trans_epsilon

            rgbs, _ = render_path(
                poses[i_test], hwf, self.K, self.chunk_render,
                render_kwargs_test,
                gt_imgs=images[i_test],
                savedir=testsavedir,
                render_factor=self.render_factor,
            )

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                render_kwargs_test["network_fn"] = network_fn_tmp
                render_kwargs_test["network_fine"] = network_fine_tmp
                self.trans_epsilon = trans_epsilon_tmp

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

            trans_mat = recover_homogenous_affine_transformation(triangles_out, triangles_org) # transform matrix

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                network_fn_tmp = render_kwargs_test["network_fn"]
                network_fine_tmp = render_kwargs_test["network_fine"]
                trans_epsilon_tmp = self.trans_epsilon
                render_kwargs_test["network_fn"] = self.pretrained_nerf
                render_kwargs_test["network_fine"] = self.pretrained_nerf_fine
                self.trans_epsilon = self.old_trans_epsilon

            render_kwargs_test['trans_mat'] = trans_mat
            render_kwargs_test['test_vertices'] = vertice

            rgbs, disps = render_path(poses[i_test], hwf, self.K, self.chunk_render,
                                        render_kwargs_test,
                                        gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor)

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                render_kwargs_test["network_fn"] = network_fn_tmp
                render_kwargs_test["network_fine"] = network_fine_tmp
                self.trans_epsilon = trans_epsilon_tmp

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

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                network_fn_tmp = render_kwargs_test["network_fn"]
                network_fine_tmp = render_kwargs_test["network_fine"]
                trans_epsilon_tmp = self.trans_epsilon
                render_kwargs_test["network_fn"] = self.pretrained_nerf
                render_kwargs_test["network_fine"] = self.pretrained_nerf_fine
                self.trans_epsilon = self.old_trans_epsilon

            render_kwargs_test['trans_mat'] = recover_homogenous_affine_transformation(triangles_out, triangles_org)
            render_kwargs_test['test_vertices'] = vertice

            rgbs, disps = render_path(
                torch.Tensor(poses[i_test]).to(device), hwf, self.K, self.chunk_render,
                render_kwargs_test,
                gt_imgs=images[i_test], savedir=testsavedir, render_factor=self.render_factor
            )

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                render_kwargs_test["network_fn"] = network_fn_tmp
                render_kwargs_test["network_fine"] = network_fine_tmp
                self.trans_epsilon = trans_epsilon_tmp

            render_kwargs_test['trans_mat'] = None
            render_kwargs_test['test_vertices'] = None

        print('Saved test set')
    
    def render_video(self, i, render_poses, hwf,
            render_kwargs_test
    ):
        # Turn on testing mode
        with torch.no_grad():
            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                network_fn_tmp = render_kwargs_test["network_fn"]
                network_fine_tmp = render_kwargs_test["network_fine"]
                trans_epsilon_tmp = self.trans_epsilon
                render_kwargs_test["network_fn"] = self.pretrained_nerf
                render_kwargs_test["network_fine"] = self.pretrained_nerf_fine
                self.trans_epsilon = self.old_trans_epsilon

            rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk_render,
                                        render_kwargs_test)

            if self.use_pretrained_nerf and self.pretrained_nerf is not None:
                render_kwargs_test["network_fn"] = network_fn_tmp
                render_kwargs_test["network_fine"] = network_fine_tmp
                self.trans_epsilon = trans_epsilon_tmp

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

        self._train_prepare(50001)
        self.global_step=10000
        self.update_trans_epsilon()
        self.old_trans_epsilon = self.trans_epsilon
        self.update_sample_values(train_phase=False)
        print(self.old_trans_epsilon, self.n_central_samples, self.n_additional_samples)

        self.vertices = self.flame_vertices()

        self.remove_rays = True

        torch.cuda.empty_cache()
        i="load_nerf_test2_remove_rays_add"
        self.render_testset(i=i, render_poses=render_poses, hwf=hwf,
            poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        self.render_rot1(i=i, render_poses=render_poses, hwf=hwf,
            poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        torch.cuda.empty_cache()
        self.render_rot2(i=i, render_poses=render_poses, hwf=hwf,
            poses=poses, i_test=i_test, images=images, render_kwargs_test=render_kwargs_test)

        # torch.cuda.empty_cache()
        # self.render_video(i=i, render_poses=render_poses, hwf=hwf,
        # render_kwargs_test=render_kwargs_test)

        self.remove_rays = False

        torch.cuda.empty_cache()
        
