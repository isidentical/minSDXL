import torch
import numpy as np

from diffusers import StableDiffusionXLPipeline
from diffusers.models.modeling_utils import ModelMixin

from sdxl_rewrite import UNet2DConditionModel


pipe_1 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe_1 = pipe_1.to("cuda")


class UnetRewriteModel(UNet2DConditionModel, ModelMixin):
    pass


with torch.device("cuda"):
    with torch.cuda.amp.autocast():
        unet_new = UnetRewriteModel().half()
        unet_new.load_state_dict(pipe_1.unet.state_dict())

unet_new = unet_new.eval()

pipe_2 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    unet=unet_new,
)
pipe_2 = pipe_2.to("cuda")


result_1 = pipe_1(
    "a cat",
    generator=torch.Generator().manual_seed(42),
    output_type="np",
    num_inference_steps=30,
)
image_1 = result_1.images[0]

result_2 = pipe_2(
    "a cat",
    generator=torch.Generator().manual_seed(42),
    output_type="np",
    num_inference_steps=30,
)
image_2 = result_2.images[0]

