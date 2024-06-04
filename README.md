# sdxl_turbo_diffusers_api

A image generation service api, developed with diffusers + flask

# Get Started

1. You can load the model locally through the environment variable declaration, if you do not declare it, it will be
   downloaded from huggingface.co

```bash
export BaseModelPath="your/model/path"
export PidiNetPath="your/model/path"
export AdapterSketchPath="your/model/path"
export LoraPath="your/model/path"
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

## /sdapi/v1/txt2img

Recommended parameters:

```json
{
  "prompt": "stylized by Daria Petrilli, Raw digital photo, 3D Rendering, Colorful Bird, it is Verdant, concept art, Stormy weather, Dark Academia, Light and shadow plays, film grain, Nikon d3300, 80mm, glitter texture, most beautiful artwork in the world, highly detailed, extremely content happy smile",
  "steps": 5,
  "cfg_scale": 1.5,
  "width": 1344,
  "height": 768
}
```

with sketch:

```json
{
  "prompt": "stylized by Daria Petrilli, Raw digital photo, 3D Rendering, Colorful Bird, it is Verdant, concept art, Stormy weather, Dark Academia, Light and shadow plays, film grain, Nikon d3300, 80mm, glitter texture, most beautiful artwork in the world, highly detailed, extremely content happy smile",
  "steps": 5,
  "cfg_scale": 4,
  "width": 1344,
  "height": 768
}
```

- /sdapi/v1/progress
- /api/v1/text2img/controlnet/reference
