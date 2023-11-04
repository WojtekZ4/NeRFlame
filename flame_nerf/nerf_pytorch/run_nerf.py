from flame_nerf.nerf_pytorch.nerf_utils import *
from flame_nerf.utils import load_obj_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    trainer_config,
):
    trainer = load_obj_from_config(cfg=trainer_config)
    trainer.train()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    trainer_config = {"kwargs": {
        'dataset_type': args.dataset_type,
        'render_test': args.render_test,
        'render_only': args.render_only,
        'basedir': args.basedir,
        'expname': args.expname,
        'config_path': args.config_path,
        'device': device,
        'render_factor': args.render_factor,
        'chunk': args.chunk,
        'N_rand': args.N_rand,
        'no_batching': args.no_batching,
        'half_res': args.half_res,
        'testskip': args.testskip,
        'white_bkgd': args.white_bkgd,
        'datadir': args.datadir,
        'multires': args.multires,
        'i_embed': args.i_embed,
        'multires_views': args.multires_views,
        'netchunk': args.netchunk,
        'lrate': args.lrate,
        'input_dims_embed': args.input_dims_embed,
        'lrate_decay': args.lrate_decay,
        'use_viewdirs': args.use_viewdirs,
        'N_importance': args.N_importance,
        'netdepth': args.netdepth,
        'netwidth': args.netwidth,
        'netdepth_fine': args.netdepth_fine,
        'netwidth_fine': args.netwidth_fine,
        'ft_path': args.ft_path,
        'perturb': args.perturb,
        'raw_noise_std': args.raw_noise_std,
        'N_samples': args.N_samples,
        'lindisp': args.lindisp,
        'precrop_iters': args.precrop_iters,
        'precrop_frac': args.precrop_frac,
        'i_weights': args.i_weights,
        'i_testset': args.i_testset,
        'i_video': args.i_video,
        'i_print': args.i_print
    }, "module": "flame_nerf.nerf_pytorch.Trainer"}

    train(
        trainer_config,
    )
