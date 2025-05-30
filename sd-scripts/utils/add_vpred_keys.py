# Script to add `v_pred` and `ztsnr` keys to the state dict of a model

# Usage:
# Copy to kohya-ss/sd-scripts directory
# Run with:
# 
# source venv/bin/activate
# accelerate launch "./add_vpred_keys.py" ^
# --sd-model="/directory/modename.safetensors" ^
# --save-to="/directory/modelname-update.safetensors" ^
# --save-precision="fp16"

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from library import sai_model_spec, train_util
import library.model_util as model_util
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    # Add the keys to recognise v-pred SDXL models
    sd["v_pred"] = torch.zeros(1)
    sd["ztsnr"] = torch.zeros(1)

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, model, state_dict, dtype, metadata):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)


def add_vpred_keys(args):
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = torch.float16
    state_dict, metadata = load_state_dict(args.sd_model, save_dtype)

    logger.info(f"saving model to: {args.save_to}")
    save_to_file(save_to, state_dict, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="Precision in saving, fp16 if omitted / 保存時に精度を変更して保存する、省略時は精度と同じ",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help="Stable Diffusion model to load: ckpt or safetensors file  / 読み込むモデル、ckptまたはsafetensors。",
    )
    parser.add_argument(
        "--save_to", type=str, default=None, help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    add_vpred_keys(args)
