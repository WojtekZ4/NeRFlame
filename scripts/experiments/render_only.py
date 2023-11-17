"""Script for running FlameNerf."""
import click
import torch
import yaml
from flame_nerf.utils import load_obj_from_config

torch.set_default_tensor_type('torch.cuda.FloatTensor')


@click.command()
@click.option(
    "--hparams_path",
    help="Type of selected dataset",
    type=str,
    default="scripts/configs/faces_f1036_A.yaml"
)
@click.option(
    "--model",
    help="Selected model",
    type=str,
    default="face_f1036A_flame_nerf"
)
def main(
        hparams_path: str,
        model: str,
):
    """Main."""
    with open(hparams_path, "r") as fin:
        hparams = yaml.safe_load(fin)[model]

    torch.manual_seed(42)  # 0

    trainer = load_obj_from_config(cfg=hparams)

    trainer.render_load()

if __name__ == "__main__":
    main()