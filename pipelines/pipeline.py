# -*- coding: utf-8 -*-

from diffusers import StableDiffusionXLPipeline,StableDiffusionXLAdapterPipeline,T2IAdapter
import torch
from diffusers import EulerAncestralDiscreteScheduler,DPMSolverSinglestepScheduler
from pipelines.stablediffusion_xl_reference_pipeline import StableDiffusionXLReferencePipeline
from controlnet_aux.pidi import PidiNetDetector
import os
from PIL import Image
from typing import List

# -*- coding: utf-8 -*-
class Pipeline:
    __jobs = 0
    __adapter_sketch_path = ""
    __base_model_path = ""
    __pidi_net_path = ""

    def __init__(self):
        self.get_model_path_from_env()
        self.prepare()

    def get_jobs(self) -> int:
        return self.__jobs

    # Config and load model 
    def prepare(self):
        self.load_model()

    # Get model from env
    def get_model_path_from_env(self):
        self.__base_model_path = os.environ.get("BaseModelPath", "0x4f1f/TurboVisionXL-v3.2")
        self.__pidi_net_path = os.environ.get("PidiNetPath", "lllyasviel/Annotators")
        self.__adapter_sketch_path = os.environ.get("AdapterSketchPath", "TencentARC/t2i-adapter-sketch-sdxl-1.0")

    # load model
    def load_model(self):
        self.__pipeline = StableDiffusionXLPipeline.from_pretrained(self.__base_model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.__pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(self.__pipeline.scheduler.config)
        self.__adapter_sketch = T2IAdapter.from_pretrained(self.__adapter_sketch_path, torch_dtype=torch.float16, varient="fp16").to('cuda')
        # 使用同一个 pipeline.compontent 会有干扰
        self.__pipeline_adapter = StableDiffusionXLAdapterPipeline.from_pretrained(self.__base_model_path, torch_dtype=torch.float16, variant="fp16",adapter=self.__adapter_sketch).to("cuda")
        self.__pipeline_adapter.scheduler = EulerAncestralDiscreteScheduler.from_config(self.__pipeline_adapter.scheduler.config)
        self.__pipeline_reference = StableDiffusionXLReferencePipeline(
            vae=self.__pipeline.vae,
            text_encoder=self.__pipeline.text_encoder,
            tokenizer=self.__pipeline.tokenizer,
            unet=self.__pipeline.unet,
            scheduler=EulerAncestralDiscreteScheduler.from_config(self.__pipeline.scheduler.config),
            text_encoder_2=self.__pipeline.text_encoder_2,
            tokenizer_2=self.__pipeline.tokenizer_2,
        ).to("cuda")
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
            input_image_sketch = self.__pidinet(
                image, detect_resolution=1024, image_resolution=1024, apply_filter=True
            )
            input_image_sketch.save('sketch.png')
            images = self.__pipeline_adapter(prompt=prompt, negative_prompt=negative_prompt,image=input_image_sketch,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,adapter_conditioning_scale=adapter_conditioning_scale,width=width,height=height).images
            images.append(input_image_sketch)
            return images
        except Exception as e:
            raise e
        finally:
            self.__jobs -= 1

    def text2img_reference(self, prompt: str, image: Image, width: int = 512, height: int = 512, negative_prompt: str = "", num_inference_steps: int = 5, guidance_scale: float = 0,control_guidance_start: float = 0.6,control_guidance_end:float = 1) -> str:
        self.__jobs += 1
        try:
            return self.__pipeline_reference(ref_image=image,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            reference_attn=True,
            reference_adain=True).images[0]
        except Exception as e:
            raise e
        finally:
            self.__jobs -= 1

pipe_inst = Pipeline()
