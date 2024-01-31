# -*- coding: utf-8 -*-
import logging
from PIL import Image
from pipeline import pipe_inst
from utils.image import image_to_base64

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    app = Flask(__name__)

# 兼容 sd webui api
@app.route("/sdapi/v1/txt2img", methods=['POST'])
def sdapi_t2i():
    json_data = request.get_json(force=True, silent=True) or {}
    prompt = json_data.get('prompt', '')
    negative_prompt = json_data.get('negative_prompt', "")
    guidance_scale = float(json_data.get('cfg_scale', 1.5))
    num_inference_steps = int(json_data.get('steps', 5))
    width = int(json_data.get('width', 512))
    height = int(json_data.get('height', 512))
    logger.info(f'get prompt: {prompt} --------> start generating……')
    response_data = {"images": []}
    # 如果有 controlnet
    try:
        if ("alwayson_scripts" in json_data):
            input_image = Image.open(BytesIO(base64.b64decode(json_data['alwayson_scripts']['ControlNet']['args'][0]['input_image'])))
            images = pipe_inst.text2img_sketch(prompt=prompt, negative_prompt=negative_prompt,image=input_image,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,adapter_conditioning_scale=0.9,width=width,height=height)[0]
            response_data["images"].append(image_to_base64(images[0]))
            response_data["images"].append(image_to_base64(images[-1]))
        else:
            response_data["images"].append(image_to_base64(images[0]))
    except Exception as e:
        response_data['error_message'] = str(e)
    finally:
        logger.info(f'File generated successfully <-----------------')
        return jsonify(response_data)

@app.route("/sdapi/v1/progress", methods=['get'])
def job_progress():
    global task_status
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
