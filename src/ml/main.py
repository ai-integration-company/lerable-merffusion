import json
import time
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from diffusers import DiffusionPipeline
from transparent_background import Remover
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)

    return ImageOps.expand(img, padding)


def process_image(input_image, is_small, is_standing):

    # Create a 512x512 white canvas
    canvas_size = (512, 512)
    canvas = Image.new("RGB", canvas_size, "white")

    if is_standing:
        # Paste the image around 1/3 height from the top of the canvas
        center_height = 2 * canvas_size[1] // 3
    else:
        # Paste the image around 2/3 height from the top of the canvas
        center_height = canvas_size[1] // 3

    paste_height = center_height - (input_image.height // 2)
    # Calculate the horizontal position (centered)
    paste_width = (canvas_size[0] - input_image.width) // 2

    try:
        # Paste the input image onto the canvas
        canvas.paste(input_image, (paste_width, paste_height))
    except:
        canvas = input_image.resize(512, 512)

    return canvas


def resize_image(input_image, target_width=None, target_height=None):

    # Get the original dimensions
    original_width, original_height = input_image.size

    # Calculate the new dimensions maintaining the aspect ratio
    if target_width and target_height:
        raise ValueError("Specify only one of target_width or target_height, not both.")
    elif target_width:
        # Calculate the new height maintaining the aspect ratio
        new_width = target_width
        new_height = int((new_width / original_width) * original_height)
    elif target_height:
        # Calculate the new width maintaining the aspect ratio
        new_height = target_height
        new_width = int((new_height / original_height) * original_width)
    else:
        raise ValueError("Specify either target_width or target_height.")

    resized_image = input_image.resize((new_width, new_height))

    return resized_image


model_id = "microsoft/Phi-3-vision-128k-instruct"

# use _attn_implementation='eager' to disable flash attention
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
    },
    {
        "prompt": "a sofa in the living room",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a ceiling fan in the bedroom",
        "placement": "hanging",
        "size": "large"
    },
    {
        "prompt": "a bookshelf in the bedroom",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a coffee table in the kitchen",
        "placement": "standing",
        "size": "small"
    },
    {
        "prompt": "a pendant light in the kitchen",
        "placement": "hanging",
        "size": "small"
    },
    {
        "prompt": "a floor lamp in the hall",
        "placement": "standing",
        "size": "small"
    },
    {
        "prompt": "a dining table in the kitchen",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a bed in the bedroom",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a ceiling-mounted smoke detector in the hall",
        "placement": "hanging",
        "size": "small"
    },
    {
        "prompt": "a kitchen island in the kitchen",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a bathtub in the bathroom",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a hanging plant in the living room",
        "placement": "hanging",
        "size": "small"
    },
    {
        "prompt": "a refrigerator in the kitchen",
        "placement": "standing",
        "size": "large"
    },
    {
        "prompt": "a microwave on the countertop",
        "placement": "standing",
        "size": "small"
    },
    {
        "prompt": "a ceiling light in the hallway",
        "placement": "hanging",
        "size": "small"
    },
    {
        "prompt": "a floor cushion in the playroom",
        "placement": "standing",
        "size": "small"
    },
    {
        "prompt": "an overhead projector in the classroom",
        "placement": "hanging",
        "size": "large"
    },
    {
        "prompt": "a rug on the floor",
        "placement": "standing",
        "size": "large"
    }
]
# Answer in the following format: subject, location
messages = [
    {"role": "user",
     "content":
     f"""<|image_1|>\nPlease analyze the following image and provide specific details about the subject, its usual location, state and its size. 
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

image = Image.open("lamp.jpg")

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda")

generation_args = {
    "max_new_tokens": 500,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
response = json.loads(response)
print(response)

model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
pipeline = pipeline.to("cuda")

img = Image.open("lamp.jpg")
img = resize_with_padding(img, (512, 512))


remover = Remover(mode="base")

fg_mask = remover.process(img, type="map")
obj = img.crop(fg_mask.getbbox())
img = process_image(obj, response["size"] == "small", response["placement"] == "standing")
fg_mask = remover.process(img, type="map")

seed = 42
mask = ImageOps.invert(fg_mask)
generator = torch.Generator(device="cuda").manual_seed(seed)

negative_prompt = None
cond_scale = 0.15

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

mask_np = np.array(fg_mask)
kernel = np.ones((20, 20), np.uint8)  # You can adjust the size of the kernel for more or less dilation
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
        mask_image=fg_mask,
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
