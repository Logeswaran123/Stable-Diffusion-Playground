## Animate mode
Animate mode can generate "2D" or "3D" videos from input prompts. Also, it can perform Video-to-Video conversion of a "Video Input" based on input prompts.

### Run command
Clone the repo, and run the cmds from Stable-Diffusion-Playground dir.
```python
pip install -r requirements.txt
python setup.py
mkdir models
mkdir pretrained
cd animation_mode
python setup.py
cd ..
```
```python
python run.py --mode animate --save
```
<br />
Animate mode uses configurations specified in ./animation_mode/config.py. Specify the configurations for video generation in this file.

### Configurations
| Argument         | Description                     | Choices                   |
| ---------------- |:-------------------------------:|:-------------------------:|
| FPS              | Frame rate of the output video  | Integer number            |
| width            | width of the frame              | Integer number            |
| height           | height of the frame             | Integer number            |
| max_frames       | Number of frames in the video   | Integer number            |
| seed             | Seed value for frame generation | Integer number            |
| seed_behavior    | Seed mode                       | "iter", "fixed"           |
| animation_mode   | Mode of animation               | "2D", "3D", "Video Input" |
| guidance_scale   | Indicates how much output should be linked to prompt | Float number.<br /> Allowed: guidance_scale > 1. |
| num_inference_steps | Number of denoising steps | Integer number |
| diffusion_cadence | number of frames to generate between frames | Integer number.<br /> Allowed: > 1 for "2D", "3D" animation_mode |
| border | Border mode used in image transformation | "wrap", "replicate" |
| angle | Angle of rotation in degrees | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - integer |
| zoom | Amount of zoom | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| translation_x | Amount translation along X-axis | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| translation_y | Amount translation along Y-axis | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| translation_z | Amount translation along Z-axis | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| rotation_3d_x | Amount rotation parallel to X-axis.<br /> Used only for "3D" animation_mode | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| rotation_3d_y | Amount rotation parallel to Y-axis.<br /> Used only for "3D" animation_mode | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| rotation_3d_z | Amount rotation parallel to Z-axis.<br /> Used only for "3D" animation_mode | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float |
| strength_schedule | Indicates how much to transform the current frame from previous frame | String. Format: "frame_id:(value)".<br /> frame_id - integer, value - float.<br /> Allowed: values in range (0, 1] |
| color_coherence | Match the color of generated frames to first frame | "None", "Match Frame 0 RGB", "Match Frame 0 HSV", "Match Frame 0 LAB" |
| smooth | Smoothen image | "None", "Smooth", "Smooth_more" |
| use_depth_warping | Warp image by depth prediction.<br /> Used only for "3D" animation_mode | Bool.<br /> Allowed: True, False  |
| midas_weight | If <1.0, loads AdaBins model, else loads midas model.<br /> Used only for "3D" animation_mode | Float number.<br /> Allowed: midas_weight > 0.0 |
| near_plane | Used in image transformation. Used in py3d_tools.FoVPerspectiveCameras.<br /> Used only for "3D" animation_mode | Integer number |
| far_plane | Used in image transformation. Used in py3d_tools.FoVPerspectiveCameras.<br /> Used only for "3D" animation_mode | Integer number |
| fov | Used in image transformation. Used in py3d_tools.FoVPerspectiveCameras.<br /> Used only for "3D" animation_mode | Integer number |
| padding_mode | Padding mode in image transformation.<br /> Used in torch.nn.functional.grid_sample.<br /> Used only for "3D" animation_mode | "zeros", "border", "reflection" |
| sampling_mode | Sampling mode in image transformation.<br /> Used in torch.nn.functional.grid_sample.<br /> Used only for "3D" animation_mode | "bilinear", "nearest", "bicubic" |
| save_depth_maps | Save the predicted depth maps | Bool.<br /> Allowed: True, False |
| video_init_path | Path to video file.<br /> Used only for "Video Input" animation_mode | String |
| video_same_size | Indicates if output video should be same size as input video.<br /> Used only for "Video Input" animation_mode | Bool.<br /> Allowed: True, False |
| extract_nth_frame | Extract every nth frame from video.<br /> Used only for "Video Input" animation_mode | Integer number |
| animation_prompts | Dictionary with key as frame id and value as prompt | Format: {"frame_id":promt}.<br /> frame_id - Integer number, promt - String |

<br />
<b>Note:</b><br />
angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, strength_schedule can take a series of values.<br />
It should be in format: "frame_id:(value),frame_id:(value),..."<br /><br />
For example, angle="0:(0),10:(30),20:(-30)". This means that starting from frame 0 till frame 9, the frames will have no angle change. Then, from frame 10 till frame 19, the frames will rotate clock-wise by 30 degrees. Then, from frame 20 till end of video or max_frames, the frames will rotate anti-clockwise by 30 degrees.

---

animation_prompts can take a series of prompts.<br />
It should be in format: {"frame_id":prompt, "frame_id":prompt, ...}<br /><br />
For example, animation_prompts = {"0":"White clouds in blue sky, realistic, 8k!!!", "100":"Aeroplane in blue sky, realistic!!"}. This means from frame 0 till frame 99, the generated frames will be based on prompt "White clouds in blue sky, realistic, 8k!!!". Then, from frame 100 till end of video or max_frames, the frames will be based on prompt "Aeroplane in blue sky, realistic!!".

---
