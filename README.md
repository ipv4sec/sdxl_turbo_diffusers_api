# sdxl_turbo_diffusers_api
A image generation service api, developed with diffusers + flask

# Get Started

1. You can load the model locally through the environment variable declaration, if you do not declare it, it will be downloaded from huggingface.co

```bash
export BaseModelPath="your/model/path"
export PidiNetPath="your/model/path"
export AdapterSketchPath="your/model/path"
```

2. Start the service through flask

```bash
pip install -r requirements.txt
```

```bash
python -m flask run
```

# Api doc

just like stable diffusion webui api, but only:

- /sdapi/v1/txt2img
- /sdapi/v1/progress
