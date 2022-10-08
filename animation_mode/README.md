## Animate mode
Animate mode can generate "2D" or "3D" videos from input prompts. Also, it can perform Video-to-Video conversion of a "Video Input" based on input prompts.

### Run command
Clone the repo, and run the cmd from Stable-Diffusion-Playground dir.
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
| guidance_scale   | Indicates how much output should be linked to prompt | Float number. Allowed: guidance_scale > 1. |
| num_inference_steps | Number of denoising steps | Integer number |
| diffusion_cadence | number of frames to generate between frames | Integer number. Allowed: > 1 for "2D", "3D" animation_mode |
| border | Border mode used in image transformation | "warp", "replicate" |
| angle | Angle of rotation in degrees | Integer number |
| zoom | Amount of zoom | Float number |
| translation_x | Amount translation along X-axis | Float number |
| translation_y | Amount translation along Y-axis | Float number |
| translation_z | Amount translation along Z-axis | Float number |
| rotation_3d_x | Amount rotation parallel to X-axis | Float number. Used only for "3D" animation_mode |
| rotation_3d_y | Amount rotation parallel to Y-axis | Float number. Used only for "3D" animation_mode |
| rotation_3d_z | Amount rotation parallel to Z-axis | Float number. Used only for "3D" animation_mode |
| strength_schedule | Indicates how much to transform the current frame from previous frame | Float number. Allowed: values in range (0, 1] |
| color_coherence | Match the color of generated frames to first frame | "None", "Match Frame 0 RGB", "Match Frame 0 HSV", "Match Frame 0 LAB" |
| smooth | Smoothen image | "None", "Smooth", "Smooth_more" |
| use_depth_warping | | |
| midas_weight | | |
| near_plane | | |
| far_plane | | |
| fov | | |
| padding_mode | | |
| sampling_mode | | |
| save_depth_maps | | |
| video_init_path | Path to video file | String. Used for "Video Input" animation_mode |
| video_same_size | Indicates if output video should be same size as input video | Bool. Allowed: True, False. Used for "Video Input" animation_mode |
| extract_nth_frame | Extract every nth frame from video | Integer number |
| animation_prompts | Dictionary with key as frame id and value as prompt | Format: {frame_id:promt}.<br /> frame_id - Integer number, promt - String |
