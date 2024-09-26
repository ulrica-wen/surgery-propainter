import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from .mask_painter import mask_painter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class BaseSegmenter:
    def __init__(self, SAM_checkpoint, model_type, device='cuda:0', fine_tuned_weights="../../fine_tuned_sam2_3000.torch"):
        """
        Initialize BaseSegmenter with SAM or SAM2 models, optionally loading fine-tuned weights.
        
        Parameters:
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        fine_tuned_weights: Path to fine-tuned model weights (optional)
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        # self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        self.model_cfg = "sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

        self.model = build_sam2( self.model_cfg, SAM_checkpoint, device="cuda")
        self.model.to(device=self.device)
        
        # Optionally load fine-tuned weights if provided
        if fine_tuned_weights:
            print(f"Loading fine-tuned weights from {fine_tuned_weights}")
            self.model.load_state_dict(torch.load(fine_tuned_weights, map_location=self.device))
        
        self.predictor = SAM2ImagePredictor(self.model)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        """
        Set image and embed it for segmentation tasks. Embedding avoids re-encoding the same image multiple times.
        """
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        """
        Reset the image embedding to allow for new images to be processed.
        """
        self.predictor.reset_predictor()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        Predict masks based on input prompts.
        
        Parameters:
        prompts: dictionary with 3 keys: 'point_coords', 'point_labels', 'mask_input'
        mode: prediction mode: 'point' (using points), 'mask' (using masks), or 'both' (using both points and masks)
        multimask: If True, return multiple masks, otherwise return a single mask
        
        Returns:
        masks, scores, logits
        """
        assert self.embedded, 'Prediction called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                                           point_labels=prompts['point_labels'], 
                                                           multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'], 
                                                           multimask_output=multimask)
        elif mode == 'both':   # both points and masks
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                                           point_labels=prompts['point_labels'], 
                                                           mask_input=prompts['mask_input'], 
                                                           multimask_output=multimask)
        else:
            raise NotImplementedError("Mode not implemented!")
        
        # Return masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits


if __name__ == "__main__":
    # load and show an image
    image = cv2.imread('/hhd3/gaoshang/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # numpy array (h, w, 3)

    # Initialize BaseSegmenter with fine-tuned weights (optional)
    SAM_checkpoint = "/mnt/datascience3/Boya/bouyang/segment-anything-2/checkpoints/sam2_hiera_small.pt"
    model_type = 'vit_h'
    device = "cuda:4"
    fine_tuned_weights = '/path/to/fine_tuned_weights.pth'  # Replace with actual path
    base_segmenter = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device, fine_tuned_weights=fine_tuned_weights)
    
    # Image embedding (once embedded, multiple prompts can be applied)
    base_segmenter.set_image(image)
    
    # Example 1: Using point only
    mode = 'point'
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 1]), 
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=False)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)

    # Example 2: Using both points and masks
    mode = 'both'
    mask_input = logits[np.argmax(scores), :, :]
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 0]), 
        'mask_input': mask_input[None, :, :]
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/hhd3/gaoshang/truck_both.jpg', painted_image)

    # Example 3: Using mask only
    mode = 'mask'
    mask_input = logits[np.argmax(scores), :, :]
    prompts = {'mask_input': mask_input[None, :, :]}
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/hhd3/gaoshang/truck_mask.jpg', painted_image)
