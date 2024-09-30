import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import sys
sys.path.append('../')

import os

from tracker.config import CONFIG
from tracker.model.cutie import CUTIE
from tracker.inference.inference_core import InferenceCore
from tracker.utils.mask_mapper import MaskMapper

from tools.painter import mask_painter

from sam2.build_sam import build_sam2_video_predictor

sam2_configs = {'sam2_hiera_large.pt': 'sam2_hiera_l.yaml',}        # add config name for new models


class ControlerAndTracker:
    def __init__(self, sam_checkpoint, model_type, device):

        self.device = device
        self.model_cfg = sam2_configs[os.path.basename(sam_checkpoint)]
        self.sam_controler = build_sam2_video_predictor(self.model_cfg, sam_checkpoint, device=device)
    
    def init_state(self, video_path):
        self.inference_state = self.sam_controler.init_state(video_path=video_path)

    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, mask_color=3):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        _, out_obj_ids, out_mask_logits = self.sam_controler.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        import code; code.interact(local=locals())
            
        
        assert len(points)==len(labels)
        
        painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        
        return mask, logit, painted_image





class BaseTracker:
    def __init__(self, cutie_checkpoint, device) -> None:
        """
        device: model device
        cutie_checkpoint: checkpoint of XMem model
        """
        config = OmegaConf.create(CONFIG)

        # initialise XMem
        network = CUTIE(config).to(device).eval()
        model_weights = torch.load(cutie_checkpoint, map_location=device)
        network.load_weights(model_weights)

        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        self.device = device
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def image_to_torch(self, frame: np.ndarray, device: str = 'cuda'):
            # frame: H*W*3 numpy array
            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
            return frame
    
    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
        else:
            mask = None
            labels = None

        # prepare inputs
        frame_tensor = self.image_to_torch(frame, self.device)
        
        # track one frame
        probs = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs+1):
            if np.max(final_mask==obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()