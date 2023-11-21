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
    hparams_paths =  ["scripts/configs/" + face + ".yaml" for face in \
                    ["faces_f1036_A", "faces_f1042_S", "faces_M1000_N", "faces_m1007_S", "faces_m1011_D", "faces_m1047_S"]]
    models =    [face + "_flame_nerf" for face in \
                ["face_f1036A", "face_f1042S", "face_M1000N", "face_m1007S", "face_m1011D", "face_m1047S"]]
    for hparams_path, model in zip(hparams_paths, models):
        with open(hparams_path, "r") as fin:
            hparams = yaml.safe_load(fin)[model]

        torch.manual_seed(42)  # 0

        trainer = load_obj_from_config(cfg=hparams)

        trainer.render_load()

if __name__ == "__main__":
    main()