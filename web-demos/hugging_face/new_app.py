import sys
sys.path.append("../../")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import cv2
import torch
import shutil
import torchvision
import numpy as np
import gradio as gr
from PIL import Image
from inpainter.base_inpainter import ProInpainter
from sam2.build_sam import build_sam2_video_predictor


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

css = """
    .gradio-container {width: 85% !important}
    .gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
    button {border-radius: 8px !important;}
    .add_button {background-color: #009f6b !important;}
    .remove_button {background-color: #f44336 !important;}
    .mask_button_group {gap: 10px !important;}
    .video {height: 300px !important;}
    .image {height: 300px !important;}
    .video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
    .video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
    .margin_center {width: 50% !important; margin: auto !important;}
    .jc_center {justify-content: center !important;}
"""
GREEN_COLOR = (0, 159, 107)
FPS = 10                # TODO: get as a gr.State and infered from actual video

checkpoint_folder = os.path.join('..', '..', 'weights')

sam2_checkpoint = os.path.join(checkpoint_folder, 'sam2_hiera_large.pt')
model_cfg = "sam2_hiera_l.yaml"

propainter_checkpoint = os.path.join(checkpoint_folder, 'ProPainter.pth')
raft_checkpoint = os.path.join(checkpoint_folder, 'raft-things.pth')
flow_completion_checkpoint = os.path.join(checkpoint_folder, 'recurrent_flow_completion.pth')

device = torch.device("cuda:0")


with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:

    clicks_state = gr.State([[],[]])
    jpeg_dir = gr.State('')
    masking_model_inference_state = gr.State(None)
    buffer_masked_video, buffer_inpainted_video = gr.State(None), gr.State(None)            # Needed for some reason - otherwise bug when displaying videos at the end of pipeline
    masking_model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inpainting_model = ProInpainter(propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, device) 

    def clear_all():
        if os.path.exists('./result/video_jpeg/'):
            shutil.rmtree('./result/video_jpeg/')
        if os.path.exists('./result/masked_video/'):  
            shutil.rmtree('./result/masked_video/')
        if os.path.exists('./result/inpainted_video/'):  
            shutil.rmtree('./result/inpainted_video/')
        return None, None, None, None, '', [[],[]], None, None, None, ''


    def clear_video():
        return None, None, None, None, '', [[],[]], None, None, None, ''
        
    def get_first_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def initialize_masking_model(jpeg_dir):
        masking_model_inference_state = masking_model.init_state(video_path=jpeg_dir)
        return masking_model_inference_state

    def select_button_actions_wrapper(video_path):
        frame = get_first_frame(video_path)
        jpeg_dir = mp4_to_jpeg(video_path)
        masking_model_inference_state = initialize_masking_model(jpeg_dir) 
        return frame, jpeg_dir, masking_model_inference_state

    def handle_image_click(video_input, clicks_state, clicks_nature, masking_model_inference_state, evt: gr.SelectData):
        #saving click coordinates and nature in the gr.state
        clicks = evt.index if evt.index else []
        if clicks != None:
            clicks_state[0].append(clicks)
            clicks_state[1].extend([1 if clicks_nature == 'Positive' else 0])
        
        # masking first frame according to points selected so far (and initialize future video masking)
        frame_idx = 0       # for now only supports frame 0
        obj_id = 1          # for now only supports 1 mask

        points = np.array(clicks_state[0], dtype=np.float32)
        labels = np.array(clicks_state[1], np.int32)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, mask_logits = masking_model.add_new_points_or_box(inference_state=masking_model_inference_state,
                                                            frame_idx=frame_idx,
                                                            obj_id=obj_id,
                                                            points=points,
                                                            labels=labels,
                                                            )
        
        mask_logits = mask_logits[0]        # for now only supports 1 mask
        mask_logits = mask_logits.cpu().numpy().transpose(1,2,0)

        mask = mask_logits > 0.
        mask = np.repeat(mask, 4, axis=2)

        pixels_to_impact = (mask[:,:,:3] != [0,0,0]).all(axis=2)

        mask = np.uint8(mask * np.array(GREEN_COLOR + (0,)))
        mask[:, :, 3][pixels_to_impact] = 128

        mask = Image.fromarray(mask).convert('RGBA')
        image = Image.fromarray(get_first_frame(video_input)).convert('RGBA')

        image = Image.alpha_composite(image, mask).convert('RGB')
        # image = np.array(image.convert('RGB'))

        # also draw the points
        image = draw_points(image, clicks_state)

        return image, clicks_state

    def clear_clicks(video_input):
        clicks_state = [[],[]]
        image = get_first_frame(video_input)
        image = draw_points(image, clicks_state)
        return image, clicks_state, ''

    def draw_points(image, clicks_state):
        image = np.array(image.copy())
        for point, label in zip(*clicks_state):
            if label == 1:
                cv2.circle(image, tuple(point), radius=8, color=(255, 255, 255), thickness=2) #contour
                cv2.circle(image, tuple(point), radius=7, color=GREEN_COLOR, thickness=-1)
            else:
                cv2.circle(image, tuple(point), radius=8, color=(255, 255, 255), thickness=2) #contour
                cv2.circle(image, tuple(point), radius=7, color=(0,0,0), thickness=-1)
        return image

    def mp4_to_jpeg(video_input):
        jpeg_dir = f'./result/video_jpeg/{os.path.split(video_input)[-1][:-4]}/'
        os.makedirs(jpeg_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_input)
        fps = FPS #cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(jpeg_dir, f'{frame_count:05d}.jpeg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()

        return jpeg_dir  

    def mask_video(clicks_state, jpeg_dir, masking_model_inference_state):
        ann_frame_idx = 0       # for now only supports frame 0
        ann_obj_id = 1          # for now only supports 1 mask

        points = np.array(clicks_state[0], dtype=np.float32)
        labels = np.array(clicks_state[1], np.int32)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            masking_model.add_new_points_or_box(inference_state=masking_model_inference_state,
                                                frame_idx=ann_frame_idx,
                                                obj_id=ann_obj_id,
                                                points=points,
                                                labels=labels,
                                            )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in masking_model.propagate_in_video(masking_model_inference_state):
                video_segments[out_frame_idx] = {
                                                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                                                }

            masking_model.reset_state(masking_model_inference_state)

        return video_segments, masking_model_inference_state
    

    def masked_frames_to_video(video_segments, video_input, jpeg_dir):
        masked_video_dir = './result/masked_video/'
        os.makedirs(masked_video_dir, exist_ok=True)

        frames = sorted(os.listdir(jpeg_dir))
        width, height = Image.open(os.path.join(jpeg_dir, frames[0])).size

        video_path = os.path.join(masked_video_dir, os.path.split(video_input)[-1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out_video = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        for idx in range(len(frames)):
            current_frame = Image.open(os.path.join(jpeg_dir, frames[idx])).convert('RGBA')

            for _, mask in video_segments[idx].items():
                mask = np.repeat(mask.transpose(1,2,0), 4, axis=2)
                pixels_to_impact = (mask[:,:,:3] != [0,0,0]).all(axis=2)
                mask = np.uint8(mask * np.array(GREEN_COLOR + (0,)))
                mask[:, :, 3][pixels_to_impact] = 128
                mask = Image.fromarray(mask).resize((width, height)).convert('RGBA')
                current_frame = Image.alpha_composite(current_frame, mask)

            current_frame = current_frame.convert('RGB')
            current_frame = np.array(current_frame)

            out_video.write(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))

        out_video.release()

        return video_path
    
    def inpaint_video(jpeg_dir, video_segments):
        frames = sorted(os.listdir(jpeg_dir))

        frames_array = []
        masks_array = []
        for idx in range(len(frames)):
            frames_array.append(cv2.imread(os.path.join(jpeg_dir, frames[idx])))

            for _, mask in video_segments[idx].items():
                masks_array.append(mask.squeeze(0))

        frames_array = np.asarray(frames_array)
        masks_array = np.asarray(masks_array)

        inpainted_frames = inpainting_model.inpaint(frames_array, 
                                                masks_array, 
                                                ratio=1, 
                                                dilate_radius=8,
                                                raft_iter=20,
                                                subvideo_length=80, 
                                                neighbor_length=10, 
                                                ref_stride=10,
                                                )

        return inpainted_frames
    
    def inpainted_frames_to_video(frames, video_segments, video_input):
        inpainted_video_dir = './result/inpainted_video/'
        os.makedirs(inpainted_video_dir, exist_ok=True)

        video_path = os.path.join(inpainted_video_dir, os.path.split(video_input)[-1])

        frames_with_countours = []
        masks = [mask.transpose(1,2,0) for i in range(len(video_segments)) for _,mask in video_segments[i].items()]
        for frame, mask in zip(frames, masks):
            frame = frame[..., ::-1]             #GBR to RGB
            mask = np.uint8(mask)                #need 0-1 mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frames_with_countours.append(cv2.drawContours(frame.copy(), contours, -1, color=GREEN_COLOR, thickness=2))       

        frames_with_countours = torch.from_numpy(np.asarray(frames_with_countours))
        torchvision.io.write_video(video_path, frames_with_countours, fps=FPS, video_codec="libx264")

        return video_path

    
    def run_masking_inpainting(video_input, clicks_state, jpeg_dir, masking_model_inference_state):
        gr.Info('Inpainting the video...')
        video_segments, masking_model_inference_state = mask_video(clicks_state, jpeg_dir, masking_model_inference_state)
        masked_video = masked_frames_to_video(video_segments, video_input, jpeg_dir)
        inpainted_frames = inpaint_video(jpeg_dir, video_segments)
        inpainted_video = inpainted_frames_to_video(inpainted_frames, video_segments, video_input)
        return masked_video, inpainted_video, masking_model_inference_state, "Inpainting Complete."

    def display_videos(masked_video, inpainted_video):
        return masked_video, inpainted_video




    title = r"""<h1 align="center"><span style="color: green">**Surgery-ProPainter:**</span> Removing Medical Tools and Blood from Surgery Videos</h1>"""

    gr.Markdown(f"# {title}")
  
    with gr.Column():
        # input video
        gr.Markdown("## Upload a video")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):      
                video_input = gr.Video(elem_classes="video")
            with gr.Column(scale=2):
                example_folder = '/mnt/datascience1/temp/gradio'
                gr.Examples(examples=[os.path.join(example_folder, file) for file in os.listdir(example_folder) if file.endswith('.mp4')], inputs=[video_input])
                with gr.Row():
                    select_button = gr.Button("Select Video", elem_classes="add_button")
                    remove_button = gr.Button("Remove Video")
                    clear_button = gr.Button ("Clear All")


        # add masks
        gr.Markdown("## Select tools and/or blood")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                frame = gr.Image(type='pil', elem_classes="image")
            with gr.Column(scale=2):
                clicks_nature = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Point Nature", interactive=True)
                info_box = gr.Textbox(label='State')
                clear_clicks_button = gr.Button("Clear Cliks")
                validation_button_points = gr.Button("Validate & Run Inpainting", elem_classes="add_button")

        
        # Run masking + inpainting
        gr.Markdown("## Inpainted Video")
        with gr.Row():
            masked_video = gr.Video(elem_classes="video", label='masked video', scale=2)
            inpainted_video = gr.Video(elem_classes="video", label='inpainted video', scale=3)
            display_button = gr.Button("Display", elem_classes="add_button", scale=1)


    remove_button.click(clear_video, outputs=[video_input, masked_video, inpainted_video, frame, jpeg_dir, clicks_state, masking_model_inference_state, buffer_masked_video, buffer_inpainted_video, info_box])
    clear_button.click(clear_all, outputs=[video_input, masked_video, inpainted_video, frame, jpeg_dir, clicks_state, masking_model_inference_state, buffer_masked_video, buffer_inpainted_video, info_box])
    select_button.click(select_button_actions_wrapper, inputs=video_input, outputs=[frame, jpeg_dir, masking_model_inference_state])

    clear_clicks_button.click(clear_clicks, inputs=video_input, outputs=[frame, clicks_state, info_box])

    frame.select(handle_image_click, inputs=[video_input, clicks_state, clicks_nature, masking_model_inference_state], outputs=[frame, clicks_state])

    validation_button_points.click(run_masking_inpainting, inputs=[video_input, clicks_state, jpeg_dir, masking_model_inference_state], outputs=[buffer_masked_video, buffer_inpainted_video, masking_model_inference_state, info_box])
    display_button.click(display_videos, inputs=[buffer_masked_video, buffer_inpainted_video], outputs=[masked_video, inpainted_video])
    


demo.launch(share=True)