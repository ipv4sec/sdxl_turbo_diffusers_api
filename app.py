# -*- coding: utf-8 -*-
import base64
import logging
from io import BytesIO

from PIL import Image
from flask import Flask, jsonify
from flask import request

from pipelines.pipeline import pipe_inst
from utils.image import image_to_base64

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


@app.route("/sdapi/v1/txt2img", methods=['POST'])
def sdapi_t2i():
    json_data = request.get_json(force=True, silent=True) or {}
    prompt = json_data.get('prompt', '')
    negative_prompt = json_data.get('negative_prompt', "")
    guidance_scale = float(json_data.get('cfg_scale', 1.5))
    adapter_conditioning_scale = float(json_data.get('adapter_conditioning_scale', 0.9))
    num_inference_steps = int(json_data.get('steps', 5))
    width = int(json_data.get('width', 512))
    height = int(json_data.get('height', 512))
    logger.info(f'get prompt: {prompt} --------> start generating……')
    response_data = {"images": []}
    if "alwayson_scripts" in json_data:
        if json_data['alwayson_scripts']['ControlNet']['args'][0]['module'] == "canny":
            # Canny
            input_image = Image.open(
                BytesIO(base64.b64decode(json_data['alwayson_scripts']['ControlNet']['args'][0]['input_image'])))
            image = pipe_inst.text2img_canny(prompt=prompt, image=input_image, negative_prompt=negative_prompt,
                                             guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                             width=width, height=height)
            response_data["images"].append(image_to_base64(image))
            return jsonify(response_data)
        if json_data['alwayson_scripts']['ControlNet']['args'][0]['module'] == "t2ia_sketch_pidi":
            # Sketch
            input_image = Image.open(
                BytesIO(base64.b64decode(json_data['alwayson_scripts']['ControlNet']['args'][0]['input_image'])))
            images = pipe_inst.text2img_sketch(prompt=prompt, negative_prompt=negative_prompt, image=input_image,
                                               num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                               adapter_conditioning_scale=adapter_conditioning_scale, width=width,
                                               height=height)
            response_data["images"].append(image_to_base64(images[0]))
            response_data["images"].append(image_to_base64(images[-1]))
            return jsonify(response_data)
        else:
            # Reference
            input_image = Image.open(
                BytesIO(base64.b64decode(json_data['alwayson_scripts']['ControlNet']['args'][0]['input_image'])))
            image = pipe_inst.text2img_reference(prompt=prompt, image=input_image, negative_prompt=negative_prompt,
                                                 guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                                 width=width, height=height)
            response_data["images"].append(image_to_base64(image))
            logger.info('--------> finished')
            return jsonify(response_data)
    else:
        images = pipe_inst.text2img(prompt=prompt, negative_prompt=negative_prompt,
                                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width,
                                    height=height)
        response_data["images"].append(image_to_base64(images[0]))
    return jsonify(response_data)


@app.route("/sdapi/v1/img2img", methods=['POST'])
def sdapi_i2i():
    json_data = request.get_json(force=True, silent=True) or {}
    prompt = json_data.get('prompt', '')
    negative_prompt = json_data.get('negative_prompt', "")
    guidance_scale = float(json_data.get('cfg_scale', 1.5))
    num_inference_steps = int(json_data.get('steps', 5))
    width = int(json_data.get('width', 512))
    height = int(json_data.get('height', 512))
    logger.info(f'get prompt: {prompt} --------> start generating……')
    response_data = {"images": []}
    if json_data.get("mask") is not None:
        # 局部重绘
        init_image = json_data['init_images'][0]
        init_image = init_image.replace("data:image/png;base64,", '')
        init_image = init_image.replace('data:image/webp;base64,', '')
        init_image = Image.open(
            BytesIO(base64.b64decode(init_image))).convert('RGB')
        mask_image = json_data['mask']
        mask_image = mask_image.replace('data:image/png;base64,', '')
        mask_image = mask_image.replace('data:image/webp;base64,', '')
        mask_image = Image.open(
            BytesIO(base64.b64decode(mask_image))).convert('RGB')
        image = pipe_inst.text2img_paint(prompt=prompt, init_image=init_image, mask_image=mask_image,
                                         negative_prompt=negative_prompt,
                                         guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                         width=width, height=height)
        response_data["images"].append(image_to_base64(image))
        return jsonify(response_data)
    else:
        images = pipe_inst.text2img(prompt=prompt, negative_prompt=negative_prompt,
                                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width,
                                    height=height)
        response_data["images"].append(image_to_base64(images[0]))
        return jsonify(response_data)


@app.route("/sdapi/v1/progress", methods=['get'])
def job_progress():
    response_data = {
        "progress": 0.0,
        "eta_relative": 0.0,
        "state": {
            "skipped": False,
            "interrupted": False,
            "job": "",
            "job_count": pipe_inst.get_jobs(),
            "job_timestamp": "20240122065448",
            "job_no": 1,
            "sampling_step": 0,
            "sampling_steps": 4
        },
        "current_image": None,
        "textinfo": None
    }
    return jsonify(response_data)


@app.route("/api/v1/text2img/controlnet/reference", methods=["post"])
def ti2_reference():
    json_data = request.get_json(force=True, silent=True) or {}
    prompt = json_data.get('prompt', '')
    negative_prompt = json_data.get('negative_prompt', "")
    guidance_scale = float(json_data.get('cfg_scale', 1.5))
    num_inference_steps = int(json_data.get('steps', 5))
    width = int(json_data.get('width', 512))
    height = int(json_data.get('height', 512))
    logger.info(f'get prompt: {prompt} --------> start generating……')
    response_data = {"images": []}
    input_image = Image.open(BytesIO(base64.b64decode(json_data.get('input_image'))))
    image = pipe_inst.text2img_reference(prompt=prompt, image=input_image, negative_prompt=negative_prompt,
                                         guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                         width=width, height=height)
    response_data["images"].append(image_to_base64(image))
    logger.info('--------> finished')
    return jsonify(response_data)
