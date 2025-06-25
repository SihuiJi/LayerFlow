import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset

import cv2
import json
from PIL import Image
from pathlib import Path
def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item
    
class SFTDataset_layer_f3(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3, is_train_data=True, seg_only=False, train_all_tasks=False, add_mask=False,train_2_tasks=False,bg2fg=False,fg2bg=False,load_tensors=False,image_to_video=False,caption_column="prompts.txt",video_column="videos.txt"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset_layer_f3, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.is_train_data = is_train_data
        self.seg_only = seg_only
        self.train_all_tasks = train_all_tasks
        self.train_2_tasks = train_2_tasks
        self.bg2fg = bg2fg
        self.fg2bg = fg2bg
        self.add_mask = add_mask
        self.data_dir = data_dir
        if self.is_train_data:
            self.load_tensors = load_tensors
        else:
            self.load_tensors = False
        self.image_to_video = image_to_video
        self.caption_column = caption_column
        self.video_column = video_column
        if self.load_tensors:
            self.data_dir = Path(data_dir)
        if self.load_tensors:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
            if len(self.video_paths) != len(self.prompts):
                raise ValueError(
                    f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
                )
        else:
            if self.is_train_data:
                if 'YoutubeVOS' in data_dir:
                    self.json_path = os.path.join(data_dir, 'meta_test.json')
                else:
                    self.json_path = os.path.join(data_dir, 'meta.json')
            else:
                if 'YoutubeVOS' in data_dir:
                    self.json_path = os.path.join(data_dir, 'meta_test.json')
                elif 'DAVIS' in data_dir and self.seg_only:
                    self.json_path = os.path.join(data_dir, 'meta_seg_test.json')
                        
            self.json_data = json.load(open(self.json_path, 'r'))['videos']
            self.vid_list = list(self.json_data.keys())
            self.metas = []
            
            for vid in self.vid_list:
                vid_data = self.json_data[vid]
                max_id = vid_data['max_id']
                max_vid = vid_data["objects"][max_id]
                
                vid_frames = sorted(max_vid["frames"])
                fg_prompt = vid_data['fg_prompt']
                bg_prompt = vid_data['bg_prompt']
                bl_prompt = vid_data['bl_prompt']
                if not self.is_train_data:
                    if self.train_all_tasks or self.train_2_tasks:
                        is_seg = vid_data['is_seg']
                    elif self.bg2fg:
                        is_seg = 2
                    elif self.fg2bg:
                        is_seg = 1
                    else:
                        if self.seg_only:
                            is_seg = "True"
                        elif 'is_seg' in vid_data:
                            is_seg = vid_data['is_seg']
                        else:
                            is_seg = "False"
                        

                meta = {}
                meta['video'] = vid
                meta['fg_prompt'] = fg_prompt
                meta['frames'] = vid_frames
                meta['bg_prompt'] = bg_prompt
                meta['bl_prompt'] = bl_prompt
                if not self.is_train_data:
                    meta['is_seg'] = is_seg
                self.metas.append(meta)
    
    def image_process_simple(self, rgba):
        rgb = rgba[..., :3].astype(np.float32) / 127.5 - 1
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        alpha = (alpha > 0).astype(np.float32)
        blend_rgb = (rgb + 1) * alpha - alpha
        blend_rgb = torch.from_numpy(blend_rgb).movedim(-1, 0).contiguous()
        return blend_rgb
    
    def resize_and_normalize(self, tensor_frms):
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        return tensor_frms
    
    def _load_dataset_from_local_path(self):
        if not self.data_dir.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_dir.joinpath(self.caption_column)
        video_path = self.data_dir.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_dir` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_dir` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_dir.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_dir=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths
    
    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_dir=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds
    
    def __getitem__(self, index):
        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._load_preprocessed_latents_and_embeds(self.video_paths[index])

            item =  {
                "txt": prompt_embeds,
                "mp4": video_latents,
                "load_tensors": True
            }
        else:
            decord.bridge.set_bridge("torch")

            meta = self.metas[index]
            video, frames, fg_prompt, bl_prompt, bg_prompt = meta['video'], meta['frames'], meta['fg_prompt'], meta['bl_prompt'], meta['bg_prompt']
            if not self.is_train_data:
                is_seg = meta['is_seg']
            
            frames = frames[:64]
            vid_len = len(frames[:64])
            clip_length = min(vid_len, self.max_num_frames - 1)
            start_idx   = random.randint(0, vid_len - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.max_num_frames, dtype=int)
            sample_indx = list(batch_index)
            sample_indx.sort()
            fg_rgbs, bl_rgbs, bg_rgbs = [], [], []
            if self.add_mask:
                fg_mask_rgbs = []
            for j in range(self.max_num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                if 'YoutubeVOS' in self.data_dir:
                    img_path = os.path.join(self.data_dir, 'JPEGImages', video, frame_name + '.jpg')
                    bg_path = os.path.join(self.data_dir, 'impainting', video, frame_name + '.jpg')
                    fg_mask_path = img_path.replace('JPEGImages','mask').replace('.jpg', '.png')
                    fg_mask = np.array(Image.open(fg_mask_path).convert('L'))
                    if self.add_mask:
                        fg_mask_convert = np.array(Image.open(fg_mask_path).convert('RGB')) 
                elif 'DAVIS' in self.data_dir:
                    img_path = os.path.join(self.data_dir, video, 'bl', frame_name + '.png')
                    bg_path = os.path.join(self.data_dir, video, 'bg', frame_name + '.png') 
                    fg_mask_path = os.path.join(self.data_dir, video, 'mask', frame_name + '.png') 
                    fg_mask = np.array(Image.open(fg_mask_path).convert('L'))
                    if self.add_mask:
                        fg_mask_convert = np.array(Image.open(fg_mask_path).convert('RGB')) 
                
                img = np.array(Image.open(img_path))
                h, w = img.shape[:2]
                bg = np.array(Image.open(bg_path))
                
                fg_mask = cv2.resize(fg_mask, (w,h))
                if self.add_mask:
                    fg_mask_img = cv2.resize(fg_mask_convert, (w,h))
                    fg_mask_img = fg_mask_img.astype(np.float32) / 127.5 - 1
                    fg_mask_rgb = torch.from_numpy(fg_mask_img).movedim(-1, 0).contiguous()
                    fg_mask_rgbs.append(fg_mask_rgb)
                    
                fg_img = np.concatenate([img, fg_mask[..., None]], axis=-1)
                fg_rgb = self.image_process_simple(fg_img)
                bl_mask = np.ones_like(fg_mask)*255
                
                bl_img = np.concatenate([img, bl_mask[..., None]], axis=-1)
                blend_rgb = self.image_process_simple(bl_img)
                bg_mask = np.ones_like(bg)*255
                bg_img = np.concatenate([bg, bg_mask], axis=-1)
                bg_rgb = self.image_process_simple(bg_img)
                
                fg_rgbs.append(fg_rgb)
                bl_rgbs.append(blend_rgb)
                bg_rgbs.append(bg_rgb)

            fg_rgbs = self.resize_and_normalize(torch.stack(fg_rgbs, dim=0))
            bl_rgbs = self.resize_and_normalize(torch.stack(bl_rgbs, dim=0))
            bg_rgbs = self.resize_and_normalize(torch.stack(bg_rgbs, dim=0))
            if self.add_mask:
                fg_mask_rgbs = self.resize_and_normalize(torch.stack(fg_mask_rgbs, dim=0))
                rgbs = torch.cat([fg_rgbs, fg_mask_rgbs, bg_rgbs, bl_rgbs], dim=0)
            else:
                rgbs = torch.cat([fg_rgbs, bg_rgbs, bl_rgbs], dim=0)
            
            item = {
                "mp4": rgbs,
                "txt": ['0, '+fg_prompt,'1, '+bg_prompt,'2, '+bl_prompt],
                "load_tensors": False
            }
        if not self.is_train_data:
            item['is_seg'] = is_seg
        item["num_frames"] = self.max_num_frames
        item["scale"] = 1
        item["fps"] = self.fps
            
        return item

    def __len__(self):
        if self.load_tensors:
            return len(self.video_paths)
        else:
            return len(self.vid_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

class SFTDataset_layer_f3_stage1(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3, is_train_data=True, seg_only=False, train_all_tasks=False, add_mask=False,train_2_tasks=False,bg2fg=False,fg2bg=False,load_tensors=False,image_to_video=False,caption_column="prompts.txt",video_column="videos.txt"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset_layer_f3_stage1, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.is_train_data = is_train_data
        self.seg_only = seg_only
        self.train_all_tasks = train_all_tasks
        self.train_2_tasks = train_2_tasks
        self.bg2fg = bg2fg
        self.fg2bg = fg2bg
        self.add_mask = add_mask
        self.data_dir = data_dir
        if self.is_train_data:
            self.load_tensors = load_tensors
        else:
            self.load_tensors = False
        self.image_to_video = image_to_video
        self.caption_column = caption_column
        self.video_column = video_column
        if self.load_tensors:
            self.data_dir = Path(data_dir)
        if self.load_tensors:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
            if len(self.video_paths) != len(self.prompts):
                raise ValueError(
                    f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
                )
        else:
            if 'YoutubeVOS' in data_dir:
                self.json_path = os.path.join(data_dir, 'meta_test.json')
            elif 'DAVIS' in data_dir and not self.is_train_data:
                if self.seg_only:
                    self.json_path = os.path.join(data_dir, 'meta_seg_test.json')
                else:
                    self.json_path = os.path.join(data_dir, 'meta_fg2bg_test.json')

            self.json_data = json.load(open(self.json_path, 'r'))['videos']
            self.vid_list = list(self.json_data.keys())
            self.metas = []
            
            for vid in self.vid_list:
                vid_data = self.json_data[vid]
                max_id = vid_data['max_id']
                max_vid = vid_data["objects"][max_id]
                vid_frames = sorted(max_vid["frames"])
                fg_prompt = vid_data['fg_prompt']
                bl_prompt = vid_data['bl_prompt']
                bg_prompt = vid_data['bg_prompt']
                if not self.is_train_data:
                    if self.train_all_tasks or self.train_2_tasks:
                        is_seg = vid_data['is_seg']
                    elif self.bg2fg:
                        is_seg = 2
                    elif self.fg2bg:
                        is_seg = 1
                    else:
                        if self.seg_only:
                            is_seg = "True"
                        elif 'is_seg' in vid_data:
                            is_seg = vid_data['is_seg']
                        else:
                            is_seg = "False"
                        
                meta = {}
                meta['video'] = vid
                meta['fg_prompt'] = fg_prompt
                meta['frames'] = vid_frames
                meta['bg_prompt'] = bg_prompt
                meta['bl_prompt'] = bl_prompt
                if not self.is_train_data:
                    meta['is_seg'] = is_seg
                self.metas.append(meta)
    
    def image_process_simple(self, rgba):
        rgb = rgba[..., :3].astype(np.float32) / 127.5 - 1
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        blend_rgb = (rgb + 1) * alpha - alpha
        blend_rgb = torch.from_numpy(blend_rgb).movedim(-1, 0).contiguous()
        return blend_rgb
    
    def resize_and_normalize(self, tensor_frms):
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        return tensor_frms

    def _load_dataset_from_local_path(self):
        if not self.data_dir.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_dir.joinpath(self.caption_column)
        video_path = self.data_dir.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_dir` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_dir` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_dir.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_dir=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths
    
    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_dir=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds
    
    def __getitem__(self, index):
        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._load_preprocessed_latents_and_embeds(self.video_paths[index])

            item =  {
                "txt": prompt_embeds,
                "mp4": video_latents,
                "load_tensors": True
            }
        else:
            decord.bridge.set_bridge("torch")

            meta = self.metas[index]
            video, frames, fg_prompt, bl_prompt, bg_prompt = meta['video'], meta['frames'], meta['fg_prompt'], meta['bl_prompt'], meta['bg_prompt']
            if 'shutterstock' in self.data_dir or 'MOSE' in self.data_dir:
                obj_id = meta['obj_id']
            if not self.is_train_data:
                is_seg = meta['is_seg']
            frames = frames[:64]
            vid_len = len(frames[:64])
            
            frame_indx = random.randint(0, vid_len-1)
            frame_name = frames[frame_indx]
            
            if 'YoutubeVOS' in self.data_dir:
                img_path = os.path.join(self.data_dir, 'JPEGImages', video, frame_name + '.jpg')
                bg_path = os.path.join(self.data_dir, 'impainting', video, frame_name + '.jpg')
                fg_mask_path = img_path.replace('JPEGImages','mask').replace('.jpg', '.png')
            elif 'DAVIS' in self.data_dir:
                img_path = os.path.join(self.data_dir, video, 'bl', frame_name + '.png')
                bg_path = os.path.join(self.data_dir, video, 'bg', frame_name + '.png') 
                fg_mask_path = os.path.join(self.data_dir, video, 'mask', frame_name + '.png') 
            
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            bg = np.array(Image.open(bg_path))

            fg_mask = np.array(Image.open(fg_mask_path).convert('L'))
            fg_mask = cv2.resize(fg_mask, (w,h))
            if self.add_mask:
                fg_mask_convert = np.array(Image.open(fg_mask_path).convert('RGB')) 
                fg_mask_img = cv2.resize(fg_mask_convert, (w,h))
                fg_mask_img = fg_mask_img.astype(np.float32) / 127.5 - 1
                fg_mask_rgb = torch.from_numpy(fg_mask_img).movedim(-1, 0).contiguous()
                
            fg_img = np.concatenate([img, fg_mask[..., None]], axis=-1)
            fg_rgb = self.image_process_simple(fg_img)
            bl_mask = np.ones_like(fg_mask)*255
            
            bl_img = np.concatenate([img, bl_mask[..., None]], axis=-1)
            blend_rgb = self.image_process_simple(bl_img)
            bg_mask = np.ones_like(bg)*255
            bg_img = np.concatenate([bg, bg_mask], axis=-1)
            bg_rgb = self.image_process_simple(bg_img)
                            
            fg_rgbs = [fg_rgb]*self.max_num_frames
            bl_rgbs = [blend_rgb]*self.max_num_frames
            bg_rgbs = [bg_rgb]*self.max_num_frames
            fg_rgbs = self.resize_and_normalize(torch.stack(fg_rgbs, dim=0))
            bl_rgbs = self.resize_and_normalize(torch.stack(bl_rgbs, dim=0))
            bg_rgbs = self.resize_and_normalize(torch.stack(bg_rgbs, dim=0))
            if self.add_mask:
                fg_mask_rgbs = [fg_mask_rgb]*self.max_num_frames
                fg_mask_rgbs = self.resize_and_normalize(torch.stack(fg_mask_rgbs, dim=0))
                rgbs = torch.cat([fg_rgbs, fg_mask_rgbs, bg_rgbs, bl_rgbs], dim=0)
            else:
                rgbs = torch.cat([fg_rgbs, bg_rgbs, bl_rgbs], dim=0)
            
            item = {
                "mp4": rgbs,
                "txt": ['0, '+fg_prompt,'1, '+bg_prompt,'2, '+bl_prompt],
                "load_tensors": False
            }
        if not self.is_train_data:
            item['is_seg'] = is_seg
        item["num_frames"] = self.max_num_frames
        item["scale"] = 1
        item["fps"] = self.fps
        
        return item

    def __len__(self):
        if self.load_tensors:
            return len(self.video_paths)
        else:
            return len(self.vid_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

class SFTDataset_layer_f3_stage2(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3, is_train_data=True, seg_only=False, prompt_seg=False, multi_text=True, two_text=False, train_all_tasks=False, add_mask=False,train_2_tasks=False,bg2fg=False,fg2bg=False,load_tensors=False,image_to_video=False,caption_column="prompts.txt",video_column="videos.txt",test_index=0):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset_layer_f3_stage2, self).__init__()
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.is_train_data = is_train_data
        self.seg_only = seg_only
        self.prompt_seg = prompt_seg
        self.multi_text = multi_text
        self.two_text = two_text
        self.train_all_tasks = train_all_tasks
        self.train_2_tasks = train_2_tasks
        self.bg2fg = bg2fg
        self.fg2bg = fg2bg
        self.add_mask = add_mask
        self.data_dir = data_dir
        self.test_index = test_index
        if self.is_train_data:
            self.load_tensors = load_tensors
        else:
            self.load_tensors = False
        self.image_to_video = image_to_video
        self.caption_column = caption_column
        self.video_column = video_column
        if not self.is_train_data:
            self.scale = 0
        else:
            if 'copy_and_paste_rgba' in self.data_dir:
                self.scale = 1
            elif 'YoutubeVOS' in self.data_dir or 'DAVIS' in self.data_dir:
                self.scale = 0
        if self.load_tensors:
            self.data_dir = Path(data_dir)
        if self.load_tensors:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
            if len(self.video_paths) != len(self.prompts):
                raise ValueError(
                    f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
                )
        else:
            if 'YoutubeVOS' in data_dir:
                if self.is_train_data:
                    self.json_path = os.path.join(data_dir, 'meta_test.json')
                else:
                    if not seg_only:
                        self.json_path = os.path.join(data_dir, 'meta_test.json')
                self.json_data = json.load(open(self.json_path, 'r'))['videos']
            elif 'DAVIS' in data_dir:
                if self.fg2bg:
                    self.json_path = os.path.join(data_dir, f'meta_fg2bg_test.json')
                elif self.seg_only:
                    self.json_path = os.path.join(data_dir, 'meta_seg_test.json')
                self.json_data = json.load(open(self.json_path, 'r'))['videos']
            else:
                self.json_path = os.path.join(data_dir, 'meta.json')
                self.json_data = json.load(open(self.json_path, 'r'))
            print(self.json_path)
            self.vid_list = list(self.json_data.keys())

            self.metas = []
            
            if 'YoutubeVOS' in data_dir or 'DAVIS' in self.data_dir:
                for vid in self.vid_list:
                    vid_data = self.json_data[vid]
                    max_id = vid_data['max_id']
                    max_vid = vid_data["objects"][max_id]
                    vid_frames = sorted(max_vid["frames"])
                    fg_prompt = vid_data['fg_prompt']
                    bl_prompt = vid_data['bl_prompt']
                    bg_prompt = vid_data['bg_prompt']
                    if not self.is_train_data:
                        if self.train_all_tasks or self.train_2_tasks:
                            is_seg = vid_data['is_seg']
                        elif self.bg2fg:
                            is_seg = 2
                        elif self.fg2bg:
                            is_seg = 1
                        else:
                            if self.seg_only:
                                is_seg = "True"
                            elif 'is_seg' in vid_data:
                                is_seg = vid_data['is_seg']
                            else:
                                is_seg = "False"
                    
                    meta = {}
                    meta['video'] = vid
                    meta['fg_prompt'] = fg_prompt
                    meta['frames'] = vid_frames
                    meta['bg_prompt'] = bg_prompt
                    meta['bl_prompt'] = bl_prompt
                    if not self.is_train_data:
                        meta['is_seg'] = is_seg
                    if 'shutterstock/video21024_fg' in self.data_dir and not 'shutterstock/video21024_fg_tag' in self.data_dir or 'MOSE' in self.data_dir:
                        meta['obj_id'] = max_id
                    
                    self.metas.append(meta)
            elif 'copy_and_paste_rgba' in data_dir:
                for img in self.vid_list:
                    image_data = self.json_data[img]
                    
                    meta = {}
                    meta['video'] = img
                    meta['fg_prompt'] = image_data['prompt_fg']
                    meta['bg_prompt'] = image_data['prompt_bg']
                    meta['bl_prompt'] = image_data['prompt']
                    meta['bl_images'] = [image_data['images']]
                    meta['bg_images'] = [image_data['images_bg']]
                    meta['fg_rgba_images'] = [image_data['images_fg']]
                    self.metas.append(meta)
    
    def image_process_simple(self, rgba):
        rgb = rgba[..., :3].astype(np.float32) / 127.5 - 1
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        blend_rgb = (rgb + 1) * alpha - alpha
        blend_rgb = torch.from_numpy(blend_rgb).movedim(-1, 0).contiguous()
        return blend_rgb
    
    def resize_and_normalize(self, tensor_frms):
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        return tensor_frms
    
    def _load_dataset_from_local_path(self):
        if not self.data_dir.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_dir.joinpath(self.caption_column)
        video_path = self.data_dir.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_dir` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_dir` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_dir.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_dir=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths
    
    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_dir=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds
    
    def __getitem__(self, index):
        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._load_preprocessed_latents_and_embeds(self.video_paths[index])

            item =  {
                "txt": prompt_embeds,
                "mp4": video_latents,
                "load_tensors": True
            }
        else:
            decord.bridge.set_bridge("torch")
            meta = self.metas[index]
            
            if not self.is_train_data:
                is_seg = meta['is_seg']
            if 'copy_and_paste_rgba' in self.data_dir:
                video, bl_images, bg_images, fg_rgba_images, fg_prompt, bl_prompt, bg_prompt = meta['video'], meta['bl_images'], meta['bg_images'], meta['fg_rgba_images'], meta['fg_prompt'], meta['bl_prompt'], meta['bg_prompt']
                img_path = os.path.join(self.data_dir,bl_images[0])
                img = np.array(Image.open(img_path).convert("RGB"))
                fg_rgba_path = os.path.join(self.data_dir,fg_rgba_images[0])
                fg_img = np.array(Image.open(fg_rgba_path).convert("RGBA"))
                if self.add_mask:
                    fg_mask = fg_img[..., 3]
                    fg_mask_convert = Image.fromarray(fg_mask).convert('RGB')
                    fg_mask_convert = np.array(fg_mask_convert).astype(np.float32) / 127.5 - 1
                    fg_mask_rgb = torch.from_numpy(fg_mask_convert).movedim(-1, 0).contiguous()
                    fg_mask_rgbs = [fg_mask_rgb]*self.max_num_frames
                    
                fg_rgb = self.image_process_simple(fg_img)
                fg_rgbs = [fg_rgb]*self.max_num_frames
                
                bg_img_path = os.path.join(self.data_dir,bg_images[0])
                bg_img = np.array(Image.open(bg_img_path).convert("RGB"))
                bl_mask = np.ones_like(img[...,-1])*255
                bl_img = np.concatenate([img, bl_mask[..., None]], axis=-1)
                blend_rgb = self.image_process_simple(bl_img)
                
                bg_mask = np.ones_like(bg_img[...,-1])*255
                bg_img = np.concatenate([bg_img, bg_mask[..., None]], axis=-1)
                bg_rgb = self.image_process_simple(bg_img)
                bl_rgbs = [blend_rgb]*self.max_num_frames
                bg_rgbs = [bg_rgb]*self.max_num_frames
            elif 'YoutubeVOS' in self.data_dir or 'DAVIS' in self.data_dir:
                video, frames, fg_prompt, bl_prompt, bg_prompt = meta['video'], meta['frames'], meta['fg_prompt'], meta['bl_prompt'], meta['bg_prompt']
                frames = frames[:64]
                vid_len = len(frames[:64])
                clip_length = min(vid_len, self.max_num_frames - 1)
                start_idx   = random.randint(0, vid_len - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.max_num_frames, dtype=int)
                sample_indx = list(batch_index)
                sample_indx.sort()
                fg_rgbs, bl_rgbs, bg_rgbs = [], [], []
                if self.add_mask:
                    fg_mask_rgbs = []
                for j in range(self.max_num_frames):
                    frame_indx = sample_indx[j]
                    frame_name = frames[frame_indx]
                    if 'YoutubeVOS' in self.data_dir:
                        img_path = os.path.join(self.data_dir, 'JPEGImages', video, frame_name + '.jpg')
                        bg_path = os.path.join(self.data_dir, 'impainting', video, frame_name + '.jpg')
                        fg_mask_path = img_path.replace('JPEGImages','mask')
                    elif 'DAVIS' in self.data_dir:
                        if not self.is_train_data:
                            video = video.split("*")[0]
                        img_path = os.path.join(self.data_dir, video, 'bl', frame_name + '.png')
                        bg_path = img_path
                        fg_mask_path = os.path.join(self.data_dir, video, 'mask', frame_name + '.png') 
                    else:
                        img_path = os.path.join(self.data_dir, video, 'com', frame_name + '.png')
                        bg_path = os.path.join(self.data_dir, video, 'bgr', frame_name + '.png')
                        fg_mask_path = os.path.join(self.data_dir, video, 'pha', frame_name + '.png')
                    
                    img = np.array(Image.open(img_path))
                    h, w = img.shape[:2]
                    bg = np.array(Image.open(bg_path))

                    fg_mask = np.array(Image.open(fg_mask_path).convert('L'))
                    fg_mask = cv2.resize(fg_mask, (w,h))
                    if self.add_mask:
                        fg_mask_convert = np.array(Image.open(fg_mask_path).convert('RGB')) 
                        fg_mask_img = cv2.resize(fg_mask_convert, (w,h))
                        fg_mask_img = fg_mask_img.astype(np.float32) / 127.5 - 1
                        fg_mask_rgb = torch.from_numpy(fg_mask_img).movedim(-1, 0).contiguous()
                        fg_mask_rgbs.append(fg_mask_rgb)
                    
                    fg_img = np.concatenate([img, fg_mask[..., None]], axis=-1)
                    fg_rgb = self.image_process_simple(fg_img)
                    bl_mask = np.ones_like(fg_mask)*255
                    
                    bl_img = np.concatenate([img, bl_mask[..., None]], axis=-1)
                    blend_rgb = self.image_process_simple(bl_img)
                    bg_mask = np.ones_like(bg)*255
                    bg_img = np.concatenate([bg, bg_mask], axis=-1)
                    bg_rgb = self.image_process_simple(bg_img)
                    
                    fg_rgbs.append(fg_rgb)
                    bl_rgbs.append(blend_rgb)
                    bg_rgbs.append(bg_rgb)

            fg_rgbs = self.resize_and_normalize(torch.stack(fg_rgbs, dim=0))
            bl_rgbs = self.resize_and_normalize(torch.stack(bl_rgbs, dim=0))
            bg_rgbs = self.resize_and_normalize(torch.stack(bg_rgbs, dim=0))
            if self.add_mask:
                fg_mask_rgbs = self.resize_and_normalize(torch.stack(fg_mask_rgbs, dim=0))
                rgbs = torch.cat([fg_rgbs, fg_mask_rgbs, bg_rgbs, bl_rgbs], dim=0)
            else:
                rgbs = torch.cat([fg_rgbs, bg_rgbs, bl_rgbs], dim=0)
            
            item = {
                "mp4": rgbs,
                "load_tensors": False
            }
            if self.multi_text:
                item['txt'] = ['0, '+fg_prompt,'1, '+bg_prompt,'2, '+'segmentation' if self.prompt_seg else '2, '+bl_prompt]
            elif self.two_text:
                item['txt'] = ['0, '+fg_prompt,'1, '+bg_prompt]
            else:
                item['txt'] = ['segmentation' if self.prompt_seg else bl_prompt]
            
        if not self.is_train_data:
            item['is_seg'] = is_seg
        item["num_frames"] = self.max_num_frames
        item["scale"] = self.scale
        item["fps"] = self.fps
        
        return item

    def __len__(self):
        if self.load_tensors:
            return len(self.video_paths)
        else:
            return len(self.vid_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
    
    
if __name__ == '__main__':
    import argparse, os
    from yaml import safe_load
    from torch.utils.data import Dataset, DataLoader
    import torch
    import cv2
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xx.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        yaml_args = safe_load(f)
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()
    data_params = args.data['params']
    dataset = SFTDataset_layer_f3("../data/xx", data_params['video_size'], \
        data_params['fps'],data_params['max_num_frames'], data_params['skip_frms_num'], \
        is_train_data=True, seg_only=False,\
        train_all_tasks=False, add_mask=True,train_2_tasks=False,bg2fg=False,fg2bg=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    test_dataset_folder = 'test_dataset/test_dataset_tag'
    os.makedirs(test_dataset_folder,exist_ok=True)
    from tqdm import tqdm as tdqm
    for i, data in tdqm(enumerate(dataloader), total=len(dataloader)):
        if i==2:
            break

        mp4 = data['mp4'][0]
        mp4 = ((mp4 * 0.5 + 0.5)*255).cpu().detach()
        mp4 = mp4.cpu().detach()
        
        frames = []
        for t in range(mp4.size(0)):

            frame_data = mp4[t].numpy().transpose(1, 2, 0)
            frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)
            frame = Image.fromarray(frame_data)
            frame.save(f'{test_dataset_folder}/{i}_{t}.png')
            frames.append(frame)

        frames[0].save(f'{test_dataset_folder}/{i}.gif', save_all=True, append_images=frames[1:], duration=1000/data_params['fps'], loop=0)

        
        