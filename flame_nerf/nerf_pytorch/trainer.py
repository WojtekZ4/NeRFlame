"""Trainer to train Nerf, based on https://github.com/yenchenlin/nerf-pytorch"""

from pathlib import Path

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from flame_nerf.nerf_pytorch.nerf_utils import *
from flame_nerf.nerf_pytorch.load_LINEMOD import load_LINEMOD_data
from flame_nerf.nerf_pytorch.load_blender import load_blender_data
from flame_nerf.nerf_pytorch.load_deepvoxels import load_dv_data
from flame_nerf.nerf_pytorch.load_llff import load_llff_data
from flame_nerf.nerf_pytorch.run_nerf_helpers import NeRF


class Trainer:

    def __init__(
        self,
        dataset_type,
        basedir,
        expname,
        no_batching,
        datadir,
        device="cpu",
        render_test = False,
        config_path=None,
        N_rand=32 * 32 * 4,
        render_only=False,
        chunk=1024 * 32,
        render_factor=0,
        multires=10,
        i_embed=0,
        multires_views=4,
        netchunk=1024 * 64,
        lrate=5e-4,
        lrate_decay=250,
        use_viewdirs=True,
        N_importance=0,
        netdepth=8,
        netwidth=256,
        netdepth_fine=8,
        netwidth_fine=256,
        ft_path=None,
        perturb=1.0,
        raw_noise_std=0.0,
        N_samples=64,
        lindisp=True,
        precrop_iters=0,
        precrop_frac=0.5,
        i_weights=10000,
        i_testset=100,
        i_video=5000,
        i_print=100,
        near=None,
        far=None,
        half_res=None,
        testskip=None,
        white_bkgd=None,
        tensorboard_logging: bool = True,
        input_dims_embed: int = 1
    ):
        self.start = None
        self.dataset_type = dataset_type
        self.render_test = render_test
        self.render_only = render_only
        self.basedir = basedir
        self.expname = expname
        self.config_path = config_path
        self.device = device
        self.chunk = chunk
        self.render_factor = render_factor
        self.N_rand = N_rand
        self.no_batching = no_batching
        self.use_batching = not self.no_batching
        self.datadir = datadir
        self.multires = multires
        self.i_embed = i_embed
        self.multires_views = multires_views
        self.netwidth_fine = netwidth_fine
        self.netchunk = netchunk
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.use_viewdirs = use_viewdirs
        self.N_importance = N_importance
        self.netdepth = netdepth
        self.netwidth = netwidth
        self.netdepth_fine = netdepth_fine
        self.netwidth_fine = netwidth_fine
        self.ft_path = ft_path
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.N_samples = N_samples
        self.lindisp = lindisp
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.i_weights = i_weights
        self.i_testset = i_testset
        self.i_video = i_video
        self.i_print = i_print
        self.tensorboard_logging = tensorboard_logging
        self.input_dims_embed = input_dims_embed

        self.K = None
        self.global_step = None
        self.W = None
        self.H = None
        self.c2w = None
        self.N_iters = None
        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd

        self.near = near
        self.far = far

        if ~self.render_only & tensorboard_logging:
                self.writer = SummaryWriter(
                    log_dir=f'{self.basedir}/metrics/{self.expname}'
                )

    def load_data(
        self,
        **kwargs
    ):
        """Load data and prepare poses."""
        if self.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(
                self.datadir, kwargs['factor'],
              recenter=True, bd_factor=.75,
              spherify=kwargs['spherify']
            )

            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, self.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if kwargs['llffhold'] > 0:
                print('Auto LLFF holdout,', kwargs['llffhold'])
                i_test = np.arange(images.shape[0])[::kwargs['llffhold']]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])

            print('DEFINING BOUNDS')
            if kwargs['no_ndc']:
                self.near = np.ndarray.min(bds) * .9
                self.far = np.ndarray.max(bds) * 1.

            else:
                self.near = 0.
                self.far = 1.
            print('NEAR FAR', self.near, self.far)

        elif self.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(
                self.datadir, self.half_res, self.testskip
            )
            print('Loaded blender', images.shape, render_poses.shape, hwf, self.datadir)
            i_train, i_val, i_test = i_split

            if self.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        elif self.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
                self.datadir, self.half_res,
                self.testskip
            )
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if self.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        elif self.dataset_type == 'deepvoxels':

            images, poses, render_poses, hwf, i_split = load_dv_data(
                scene=self.shape,
                basedir=self.basedir,
                testskip=self.testskip
            )

            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, self.datadir)
            i_train, i_val, i_test = i_split

            hemi_r = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
            self.near = hemi_r - 1.
            self.far = hemi_r + 1.

        else:
            raise ('Unknown dataset type', self.dataset_type, 'exiting')

        return hwf, poses, i_test, i_val, i_train, images, render_poses

    def cast_intrinsics_to_right_types(self, hwf):
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if self.K is None:
            self.K = np.array(
                [[focal, 0, 0.5 * W],
                 [0, focal, 0.5 * H],
                 [0, 0, 1]]
            )

        self.H = H
        self.W = W
        return hwf

    def create_log_dir_and_copy_the_config_file(self):
        basedir = self.basedir
        expname = self.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            dict = self.__dict__
            for arg in dict:
                file.write('{} = {}\n'.format(arg, dict[arg]))
        if self.config_path is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.config_path, 'r').read())

    def create_nerf_model(self):
        return self._create_nerf_model(model=NeRF)

    def _create_nerf_model(self, model):
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(self, model=model)
        self.global_step = start
        self.start = start

        bds_dict = {
            'near': self.near,
            'far': self.far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        return optimizer, render_kwargs_train, render_kwargs_test

    def render(self, render_test, images, i_test, render_poses, hwf, render_kwargs_test, **kwargs):
        with torch.no_grad():
            if render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                self.basedir,
                self.expname,
                'renderonly_{}_{:06d}'.format(
                    'test' if render_test else 'path', self.global_step
                )
            )

            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(
                render_poses, hwf, self.K, self.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=self.render_factor
            )
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

    def prepare_raybatch_tensor_if_batching_random_rays(
        self,
        poses,
        images,
        i_train
    ):
        i_batch = None
        rays_rgb = None

        if self.use_batching :
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
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

        # Move training data to GPU
        if self.use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if self.use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)

        return images, poses, rays_rgb, i_batch

    def rest_is_logging(
            self, i, render_poses, hwf, poses, i_test, images,
            loss, psnr, render_kwargs_train, render_kwargs_test, optimizer
    ):
        if i % self.i_weights == 0:
            path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(i))
            data = {
                'global_step': self.global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                data['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(data, path)
            print('Saved checkpoints at', path)

        if i % self.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:06d}_'.format(self.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % self.i_testset == 0 and i > 0:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, existargs_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                target_s = images[i_test]
                rgbs, _ = render_path(torch.Tensor(poses[i_test]).to(device), hwf, self.K, self.chunk,
                                      render_kwargs_test,
                                      gt_imgs=target_s, savedir=testsavedir)

                img_loss = img2mse(torch.Tensor(rgbs), torch.Tensor(target_s))
                loss = img_loss
                psnr = mse2psnr(img_loss)

                self.log_on_tensorboard(
                    i,
                    {
                        'test': {
                            'loss': loss,
                            'psnr': psnr
                        }
                    }
                )
            print('Saved test set')

        if i % self.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

    def sample_random_ray_batch(
        self,
        rays_rgb,
        i_batch,
        i_train,
        images,
        poses,
        i
    ):
        if self.use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + self.N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += self.N_rand
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
            self.c2w = torch.Tensor(pose)

            if self.N_rand is not None:
                rays_o, rays_d = get_rays(self.H, self.W, self.K, self.c2w)  # (H, W, 3), (H, W, 3)

                if i < self.precrop_iters:
                    dH = int(self.H // 2 * self.precrop_frac)
                    dW = int(self.W // 2 * self.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                            torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == self.start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {self.precrop_iters}")
                else:
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        return rays_rgb, i_batch, batch_rays, target_s

    def core_optimization_loop(
        self,
        optimizer, render_kwargs_train,
        batch_rays, i, target_s,
    ):
        rgb, disp, acc, extras = render(self.H, self.W, self.K,
            chunk=self.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train
        )

        optimizer.zero_grad()
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

        return trans, loss, psnr, psnr0

    def update_learning_rate(self, optimizer):
        decay_rate = 0.1
        decay_steps = self.lrate_decay * 1000
        new_lrate = self.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

    def _sample_points(
        self,
        z_vals_mid,
        weights,
        perturb,
        pytest,
        rays_d,
        rays_o,
        n_importance=None
    ):

        if n_importance is None:
            n_importance = self.N_importance
        z_vals_mid = z_vals_mid
        weights = weights
        perturb = perturb
        pytest = pytest

        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            n_importance,
            det=(perturb == 0.),
            pytest=pytest
        )

        z_samples = z_samples.detach()

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]  # [N_rays, N_importance, 3]
        return z_samples, pts

    def log_on_tensorboard(self, step, metrics):
        for i in metrics:
            for j in metrics[i]:
                self.writer.add_scalar(f"{i}/{j}", metrics[i][j], step)
        self.writer.flush()

    def _train_prepare(self, n_iters):
        self.N_iters = n_iters

    def train(self, N_iters = 200000 + 1):
        self._train_prepare(N_iters)

        hwf, poses, i_test, i_val, i_train, images, render_poses = self.load_data()

        if self.render_test:
            render_poses = np.array(poses[i_test])
            render_poses = torch.Tensor(render_poses).to(self.device)

        hwf = self.cast_intrinsics_to_right_types(hwf=hwf)
        self.create_log_dir_and_copy_the_config_file()
        optimizer, render_kwargs_train, render_kwargs_test = self.create_nerf_model()

        if self.render_only:
            self.render(self.render_test, images, i_test, render_poses, hwf, render_kwargs_test)
            return self.render_only

        images, poses, rays_rgb, i_batch = self.prepare_raybatch_tensor_if_batching_random_rays(
            poses, images, i_train
        )

        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        start = self.start + 1
        for i in trange(start, N_iters):
            rays_rgb, i_batch, batch_rays, target_s = self.sample_random_ray_batch(
                rays_rgb,
                i_batch,
                i_train,
                images,
                poses,
                i
            )

            trans, loss, psnr, psnr0 = self.core_optimization_loop(
                optimizer, render_kwargs_train,
                batch_rays, i, target_s,
            )

            if self.tensorboard_logging:
                self.log_on_tensorboard(
                    i,
                    {
                        'train': {
                            'loss': loss,
                            'psnr': psnr
                        }
                    }
                )

            self.update_learning_rate(optimizer)

            self.rest_is_logging(
                i,
                render_poses,
                hwf,
                poses,
                i_test,
                images,
                render_kwargs_train,
                render_kwargs_test,
                optimizer
            )

            self.global_step += 1

        self.writer.close()

    def _sample_main_points(
            self, rays_o, rays_d,
            near, far,
            N_rays,
            N_samples, perturb,
            pytest, lindisp
    ):
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

        # [N_rays, N_samples, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals

    def sample_main_points(
        self,
        near,
        far,
        perturb,
        N_rays,
        N_samples,
        viewdirs,
        network_fn,
        network_query_fn,
        rays_o,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest,
        lindisp,
        **kwargs
    ):

        rgb_map, disp_map, acc_map, depth_map = None, None, None, None
        weights = None
        z_vals = None

        if N_samples > 0:
            pts, z_vals = self._sample_main_points(
                rays_o, rays_d, near,
                far, N_rays, N_samples,
                perturb, pytest, lindisp
            )

            raw = network_query_fn(pts, viewdirs, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                pytest=pytest
            )
        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, weights, raw

    def sample_points(
        self,
        z_vals,
        weights,
        perturb,
        pytest,
        rays_d,
        rays_o,
        rgb_map,
        disp_map,
        acc_map,
        network_fn,
        network_fine,
        network_query_fn,
        viewdirs,
        raw_noise_std, white_bkgd, **kwargs
    ):
        rgb_map_0, disp_map_0, acc_map_0, raw = None, None, None, None
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
            run_fn = network_fn if network_fine is None else network_fine

            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, _, _ = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)

        return rgb_map_0, disp_map_0, acc_map_0, rgb_map, disp_map, acc_map, raw, z_samples

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, **kwargs):
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
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat(
            [torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def run_network(self, inputs, viewdirs, fn, embed_fn, embeddirs_fn):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = batchify(fn, self.netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
