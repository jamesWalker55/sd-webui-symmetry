import os
from typing import Callable, Union

import gradio as gr
import numpy as np
import torch
from PIL import Image

import modules.scripts as scripts
from modules import shared
from modules.processing import StableDiffusionProcessing, decode_first_stage
from modules.scripts import PostprocessImageArgs, AlwaysVisible
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler
from modules.sd_samplers_kdiffusion import KDiffusionSampler


def latent_to_sample(sd_model, latent):
    assert (
        len(latent.shape) == 3
    ), f"Expected latent with 3 dimensions only, got shape: {latent.shape}"

    # Code extracted from:
    # modules/sd_samplers_common.py ; single_sample_to_image(...)
    return decode_first_stage(sd_model, latent.unsqueeze(0))[0]


def sample_to_latent(sd_model, img):
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(img))

def sample_to_image(sample):
    # Code extracted from:
    # modules/sd_samplers_common.py ; single_sample_to_image(...)
    sample = sample * 0.5 + 0.5
    sample = torch.clamp(sample, min=0.0, max=1.0)
    sample = 255.0 * np.moveaxis(sample.cpu().numpy(), 0, 2)
    sample = sample.astype(np.uint8)

    return Image.fromarray(sample)

def create_callback(original_callback):
    def callback(sampler, d: dict | torch.Tensor):
        # mirror(d)
        # mirror(d["denoised"])
        # _, _, _, width = latent.size()
        #
        # # Calculate the center column index
        # center_col = width // 2
        #
        # # Split the tensor into left and right parts
        # if width % 2 == 0:
        #     latent[:, :, :, center_col:] = torch.flip(latent[:, :, :, :center_col], dims=[3])
        # else:
        #     latent[:, :, :, center_col + 1:] = torch.flip(latent[:, :, :, :center_col], dims=[3])
        if isinstance(d, dict):
            latent = d['denoised']
        else:
            latent = d

        print(latent)
        print(latent.size())
        sample = latent_to_sample(shared.sd_model, latent[0])
        print(sample.size())
        new_latent = sample_to_latent(shared.sd_model, sample)
        print(new_latent.size())
        print(latent - new_latent)

        return original_callback(sampler, d)

    return callback


class Script(scripts.Script):
    original_callback_state: Union[Callable, None]
    original_show_progress_type: Union[str, None]

    def __init__(self):
        super().__init__()
        self.original_callback_state = None
        self.original_show_progress_type = None

    def title(self):
        return "Mirror"

    def show(self, is_img2img: bool):
        """This method MUST return AlwaysVisible for #process / #postprocess to be called"""
        return AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion("Symmetry", open=True, elem_id="symmetry-extension"):
            with gr.Row():
                is_enabled = gr.Checkbox(label="Script Enabled", value=True)

        return [is_enabled]

    def process(
        self,
        p: StableDiffusionProcessing,
        is_enabled: bool,
    ):
        print("Called #process")
        # if not is_enabled:
        #     return

        # # Override 'show_progress_type'
        # if self.original_show_progress_type is not None:
        #     raise RuntimeError("'show_progress_type' already overridden")
        #
        # self.original_show_progress_type = shared.opts.data["show_progress_type"]
        # shared.opts.data["show_progress_type"] = "Full"

        # Override callback state function
        if self.original_callback_state is not None:
            raise RuntimeError("Callback state already overridden")

        if p.sampler_name in ("DDIM", "PLMS", "UniPC"):
            self.original_callback_state = VanillaStableDiffusionSampler.update_step
            VanillaStableDiffusionSampler.update_step = create_callback(
                self.original_callback_state
            )
        else:
            self.original_callback_state = KDiffusionSampler.callback_state
            KDiffusionSampler.callback_state = create_callback(
                self.original_callback_state
            )

    def postprocess(
        self,
        p: StableDiffusionProcessing,
        pp: PostprocessImageArgs,
        is_enabled: bool,
    ):
        print("Called #postprocess")
        # if not is_enabled:
        #     return

        # # Override 'show_progress_type'
        # if self.original_show_progress_type is None:
        #     raise RuntimeError("'show_progress_type' not overridden")
        #
        # shared.opts.data["show_progress_type"] = self.original_show_progress_type
        # self.original_show_progress_type = None

        # Revert callback state function
        if self.original_callback_state is None:
            raise RuntimeError("Callback state not overridden")

        if p.sampler_name in ("DDIM", "PLMS", "UniPC"):
            VanillaStableDiffusionSampler.update_step = self.original_callback_state
        else:
            KDiffusionSampler.callback_state = self.original_callback_state

        self.original_callback_state = None
