import sys
sys.path.append("../../")

import os
os.environ["CUDA_VISIBLE_DEVICE"] = "2"

import cv2
import gradio as gr
import numpy as np
import json
import argparse
from model.misc import get_device
from track_anything import TrackingAnything
from utils.download_util import load_file_from_url
import time

css = """
    .gradio-container {width: 85% !important}
    .gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
    button {border-radius: 8px !important;}
    .add_button {background-color: #4CAF50 !important;}
    .remove_button {background-color: #f44336 !important;}
    .mask_button_group {gap: 10px !important;}
    .video {height: 300px !important;}
    .image {height: 300px !important;}
    .video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
    .video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
    .margin_center {width: 50% !important; margin: auto !important;}
    .jc_center {justify-content: center !important;}
"""

def parse_augment():
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default=None)
        parser.add_argument('--sam_model_type', type=str, default="vit_h")
        parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")  
        parser.add_argument('--mask_save', default=False)
        args = parser.parse_args()
        
        if not args.device:
            args.device = str(get_device())

        return args

args = parse_augment()

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_fodler = os.path.join('..', '..', 'weights')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_fodler)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), checkpoint_fodler)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), checkpoint_fodler)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), checkpoint_fodler)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), checkpoint_fodler)

# initialize sam, cutie, propainter models
model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)

def get_frames_from_video(video_input, video_state):
    video_state["video_name"] = os.path.basename(video_input)
    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    video_state["origin_images"] = frames
    video_state["painted_images"] = frames.copy()
    video_state["masks"] = [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames)
    video_state["logits"] = [None] * len(frames)
    video_state["fps"] = fps
    return video_state, frames[0]

def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    print(points, labels)
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def collect_points(video_state, point_prompt, click_state, evt:gr.SelectData):
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])

    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    image = video_state["origin_images"][video_state["select_frame_number"]].copy()
    for point, labels in zip(prompt["input_point"], prompt["input_label"]):
        if labels == 1:
            cv2.circle(image, tuple(point), radius=7, color=(0,159,107), thickness=-1)
        else:
            cv2.circle(image, tuple(point), radius=7, color=(255,0,0), thickness=-1)

    print(prompt["input_point"], prompt["input_label"])
    return image, prompt

def vos_tracking_video(prompt, video_state):
    start = time.time()

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    mask, logit, painted_image = model.first_frame_click(
    image=video_state["origin_images"][video_state["select_frame_number"]],
    points=np.array(prompt["input_point"]),
    labels=np.array(prompt["input_label"]),
    multimask=prompt["multimask_output"],
)
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    model.cutie.clear_memory()
    following_frames = video_state["origin_images"][video_state["select_frame_number"]:]
    template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    model.cutie.clear_memory()

    video_state["masks"][video_state["select_frame_number"]:] = masks
    video_state["logits"][video_state["select_frame_number"]:] = logits
    video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=fps)

    end = time.time()

    text_output = f'Inference Time = {end-start:.3f}s'
    
    return video_output, video_state, text_output

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return template_frame, click_state

def generate_video_from_frames(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return output_path

with gr.Blocks(css=css) as iface:
    gr.HTML("<h1>Timing Segmentation and Tracking with SAM + Cutie</h1>")

    video_state = gr.State({"origin_images": None, "masks": None, "select_frame_number": 0})
    click_state = gr.State([[],[]])

    with gr.Column(scale=1):
        video_input = gr.Video(elem_classes="video")

        gr.Examples(
        examples = [os.path.join('/mnt/datascience1/temp/gradio', file) for file in os.listdir('/mnt/datascience1/temp/gradio') if file.endswith('.mp4')],
        inputs=[video_input],
    )

        template_frame = gr.Image(type="pil", interactive=True)
        point_prompt = gr.Radio(choices=["Positive", "Negative"], value="Positive", interactive=True)
        clear_button_click = gr.Button(value="Clear clicks")
        tracking_video_predict_button = gr.Button(value="Tracking")

        tracking_video_output = gr.Video(elem_classes="video")

        text_output = gr.Text(label="Masking Inference Time")

    prompt = gr.State({})
    video_input.change(fn=get_frames_from_video, inputs=[video_input, video_state], outputs=[video_state, template_frame])
    template_frame.select(fn=collect_points, inputs=[video_state, point_prompt, click_state], outputs=[template_frame, prompt])
    tracking_video_predict_button.click(fn=vos_tracking_video, inputs=[prompt, video_state], outputs=[tracking_video_output, video_state, text_output])
    clear_button_click.click(fn=clear_click, inputs=[video_state, click_state], outputs=[template_frame, click_state])

iface.launch(share=True)
