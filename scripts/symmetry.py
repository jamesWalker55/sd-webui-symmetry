from typing import Callable, Union

import gradio as gr
import torch

import modules.scripts as scripts
from modules.processing import StableDiffusionProcessing
from modules.scripts import PostprocessImageArgs, AlwaysVisible
from modules.sd_samplers_compvis import VanillaStableDiffusionSampler
from modules.sd_samplers_kdiffusion import KDiffusionSampler


def create_callback(original_callback):
    def callback(sampler, d: dict | torch.Tensor):
        if isinstance(d, dict):
            latent = d['denoised']
        else:
            latent = d

        _, _, _, width = latent.size()

        # Calculate the center column index
        center_col = width // 2

        # Modify the tensor in-place
        if width % 2 == 0:
            latent[:, :, :, center_col:] = torch.flip(latent[:, :, :, :center_col], dims=[3])
        else:
            latent[:, :, :, center_col + 1:] = torch.flip(latent[:, :, :, :center_col], dims=[3])

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
        # Revert callback state function
        if self.original_callback_state is None:
            raise RuntimeError("Callback state not overridden")

        if p.sampler_name in ("DDIM", "PLMS", "UniPC"):
            VanillaStableDiffusionSampler.update_step = self.original_callback_state
        else:
            KDiffusionSampler.callback_state = self.original_callback_state

        self.original_callback_state = None
