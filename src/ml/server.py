import json
import base64
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from diffusers import DiffusionPipeline
from transparent_background import Remover
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO

app = FastAPI()


class ImageInput(BaseModel):
    image: str


class ImageOutput(BaseModel):
    background: str
    product: str
    x: int
    y: int


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def process_image(input_image, is_small, is_standing):
    canvas_size = (512, 512)
    canvas = Image.new("RGB", canvas_size, "white")

    if is_standing:
        center_height = 2 * canvas_size[1] // 3
    else:
        center_height = canvas_size[1] // 3

    paste_height = center_height - (input_image.height // 2)
    paste_width = (canvas_size[0] - input_image.width) // 2

    try:
        canvas.paste(input_image, (paste_width, paste_height))
    except:
        canvas = input_image.resize(512, 512)

    return canvas


def decode_base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


model_id = "microsoft/Phi-3-vision-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager')
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

examples = [
    {
        "prompt": "a desk lamp on the table",
        "placement": "standing",
        "size": "small"
    },
    {
        "prompt": "a chandelier in the dining room",
        "placement": "hanging",
        "size": "large"
    }
]

messages = [
    {"role": "user",
        "content": f"""\nPlease analyze the following image and provide specific details about the subject, its usual location, state and its size. 
        Follow these strict rules:
            1. Identify the subject in the image and its usual location.
            2. Determine subject's usual state: standing (on the ground) or hanging (from the ceiling).
                - Anything but ceiling is considered standing.
            3. Assess whether the subject is large or small.
                - For instance, a sofa is considered large, while a desk lamp is usually small.

        Answer in the following JSON format:
        {{
            "prompt": "subject and it's location",
            "placement": "standing/hanging",
            "size": "large/small"
        }}
        Examples: {examples}
        """}]

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id).to("cuda")

seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed)


@app.post("/process_image", response_model=ImageOutput)
def process_image_endpoint(input: ImageInput):

    image = decode_base64_to_image(input.image)

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = json.loads(response)

    img = resize_with_padding(image, (512, 512))
    remover = Remover(mode="base")
    fg_mask = remover.process(img, type="map")
    obj = img.crop(fg_mask.getbbox())
    img = process_image(obj, response["size"] == "small", response["placement"] == "standing")
    fg_mask = remover.process(img, type="map")

    mask = ImageOps.invert(fg_mask)

    negative_prompt = None
    cond_scale = 0.15

    mask_np = np.array(fg_mask)
    kernel = np.ones((20, 20), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
    mask_dilated = Image.fromarray(mask_dilated)

    with torch.autocast("cuda"):
        result = pipeline(
            prompt=response["prompt"],
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask,
            control_image=mask,
            num_images_per_prompt=1,
            generator=generator,
            num_inference_steps=20,
            guess_mode=False,
            controlnet_conditioning_scale=cond_scale,
        ).images[0]

        result2 = pipe(
            prompt="background",
            image=result,
            mask_image=mask_dilated,
            num_images_per_prompt=1,
            generator=generator,
            num_inference_steps=20,
        ).images[0]

    img = img.convert("RGBA")
    fg_mask = fg_mask.convert("L")

    img_np = np.array(img)
    mask_np = np.array(fg_mask)

    _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    b, g, r, a = cv2.split(img_np)
    alpha_channel = mask_binary
    rgba = cv2.merge((b, g, r, alpha_channel))
    result_img_pil = Image.fromarray(rgba, 'RGBA')

    bbox = fg_mask.getbbox()
    cropped_img_with_mask = result_img_pil.crop(bbox)

    background_b64 = encode_image_to_base64(result2)
    object_b64 = encode_image_to_base64(cropped_img_with_mask)

    return ImageOutput(
        background=background_b64,
        product=object_b64,
        x=bbox[0],
        y=bbox[1],
    )