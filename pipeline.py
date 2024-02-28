# -*- coding: utf-8 -*-

from diffusers import AutoPipelineForText2Image
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter,DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler
from diffusers.utils import load_image
from controlnet_aux.pidi import PidiNetDetector
import diffusers
import os
from typing import List
from PIL import Image

# -*- coding: utf-8 -*-
class Pipeline:
    __jobs = 0
    __adapter_sketch_path = ""
    __base_model_path = ""
    __pidi_net_path = ""

    def __init__(self):
        diffusers.logging.set_verbosity_debug()
        self.get_model_path_from_env()
        self.prepare()

    def get_jobs(self) -> int:
        return self.__jobs

    # Config and load model 
    def prepare(self):
        self.set_scheduler()
        self.load_model()

    # Get model from env
    def get_model_path_from_env(self):
        self.__base_model_path = os.environ.get("BaseModelPath", "0x4f1f/TurboVisionXL-v3.2")
        self.__pidi_net_path = os.environ.get("PidiNetPath", "lllyasviel/Annotators")
        self.__adapter_sketch_path = os.environ.get("AdapterSketchPath", "TencentARC/t2i-adapter-sketch-sdxl-1.0")

    # Set sampling method
    def set_scheduler(self):
        self.scheduler = DPMSolverSinglestepScheduler.from_pretrained(self.__base_model_path, subfolder="scheduler", use_karras_sigmas=True)

    # load model
    def load_model(self):
        self.__pipeline = AutoPipelineForText2Image.from_pretrained(self.__base_model_path, torch_dtype=torch.float16, variant="fp16",scheduler=self.scheduler).to("cuda")
        self.__adapter_sketch = T2IAdapter.from_pretrained(self.__adapter_sketch_path, torch_dtype=torch.float16, varient="fp16").to("cuda")
        self.__pipeline_adapter = StableDiffusionXLAdapterPipeline.from_pretrained(self.__base_model_path, torch_dtype=torch.float16, variant="fp16",adapter=self.__adapter_sketch,scheduler=self.scheduler).to("cuda")
        self.__pidinet = PidiNetDetector.from_pretrained(self.__pidi_net_path).to("cuda")
        

    # text to image
    def text2img(self, prompt: str, width: int = 512, height: int = 512, negative_prompt: str = "", num_inference_steps: int = 5, guidance_scale: float = 1.5) -> List[str]:
        self.__jobs += 1
        try:
            return self.__pipeline(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width,height=height).images
        except Exception as e:
            raise e
        finally:
            self.__jobs -= 1

    # text to image with sketch adapter
    def text2img_sketch(self, prompt: str, image: Image, width: int = 512, height: int = 512, negative_prompt: str = "", num_inference_steps: int = 5, guidance_scale: float = 1.5, adapter_conditioning_scale=0.9) -> List[str]:
        self.__jobs += 1
        try:
            input_image = load_image(image)
            input_image_sketch = self.__pidinet(
                input_image, detect_resolution=1024, image_resolution=1024, apply_filter=True
            )
            images = self.__pipeline_adapter(prompt=prompt, negative_prompt=negative_prompt,image=input_image_sketch,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,adapter_conditioning_scale=adapter_conditioning_scale,width=width,height=height).images
            images.append(input_image_sketch)
            return images
        except Exception as e:
            raise e
        finally:
            self.__jobs -= 1

pipe_inst = Pipeline()
