
import gradio as gr
import os
import sys
import torch
import gc
import datetime
from PIL import Image
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradioApp")

# Add the submodule to path so we can import from generate_bnb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lingbot-world-base-cam-nf4"))

# Now we can import the pipeline class
# Note: We rely on the generate_bnb.py structure we inspected
try:
    from generate_bnb import WanI2V_BNB, save_video
except ImportError:
    # Fallback if running from a different CWD
    sys.path.insert(0, "./lingbot-world-base-cam-nf4")
    from generate_bnb import WanI2V_BNB, save_video

# Global Pipeline Variable
PIPELINE = None
CKPT_DIR = "lingbot-world-base-cam"

def get_action_paths():
    """Scans the examples directory."""
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    if not os.path.exists(examples_dir):
        return []
    paths = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
    return [os.path.join("examples", p).replace("\\", "/") for p in sorted(paths)]

def load_model_if_needed():
    """Loads the model into global variable if not present."""
    global PIPELINE
    if PIPELINE is None:
        logger.info("[App] Initializing Model Pipeline...")
        # Check if checkpoints exist
        if not os.path.exists(CKPT_DIR):
             raise gr.Error(f"Checkpoint directory {CKPT_DIR} not found.")
             
        PIPELINE = WanI2V_BNB(
            checkpoint_dir=CKPT_DIR,
            t5_cpu=True # Default to True to save VRAM on 4090
        )
        logger.info("[App] Model Pipeline Loaded Successfully.")
    return PIPELINE

def generate_video(image_path, prompt, action_path, custom_action_path, resolution, frames, steps, progress=gr.Progress()):
    """
    Direct generation using loaded pipeline.
    """
    global PIPELINE
    
    # Validation
    if image_path is None:
        raise gr.Error("Please upload an input image.")
        
    action_path_final = custom_action_path.strip() if custom_action_path and custom_action_path.strip() else action_path
    if action_path_final == "None":
        action_path_final = None

    # Determine size
    if resolution == "480P":
        size_str = "480*832"
    else:
        size_str = "720*1280"
    h, w = map(int, size_str.split('*'))
    max_area = h * w
    
    # Load Image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise gr.Error(f"Failed to load image: {e}")

    # Load Model (Lazy Loading)
    progress(0, desc="Loading Model (One-time)...")
    try:
        pipeline = load_model_if_needed()
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.mp4"

    # Run Generation
    progress(0.1, desc="Generating Video frames...")
    try:
        # We invoke generate directly. 
        # Note: generate() in generate_bnb.py does not take a callback for progress, 
        # so the progress bar will jump from 0.1 to 1.0 when done.
        # But stdout will still show the tqdm bar in the console.
        
        video_tensor = pipeline.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path_final,
            max_area=max_area,
            frame_num=int(frames),
            sampling_steps=int(steps),
            guide_scale=5.0
        )
        
        # Save
        save_video(video_tensor, output_filename)
        
        # Cleanup a bit (optional, mainly just empty cache)
        torch.cuda.empty_cache()
        
        return output_filename

    except Exception as e:
        logger.error(f"Generation Error: {e}")
        raise gr.Error(f"Generation failed: {e}")


# ---------------- UI Setup ----------------
device_info = f"Device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Device: CPU (Slow)"

with gr.Blocks(title="LingBot-World NF4 (v1.0.0)") as demo:
    gr.Markdown(
        f"""
        # LingBot-World NF4 (Persistent Server) v1.0.0
        ## Based on the [Official LingBot-World Repository](https://github.com/Robbyant/lingbot-world)
        
        ### Windows Port by [Zast](https://zast57.com/) | [My GitHub](https://github.com/zast57)
        **{device_info}**
        """
    )
    gr.Markdown("Generate video from image with camera control. **Model stays loaded in RAM for faster subsequent runs.**")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Settings")
            image_input = gr.Image(type="filepath", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value="The video presents a soaring journey through a fantasy jungle.", lines=3, info="Describe the scene and the movement. Be descriptive!")
            
            with gr.Row():
                # Tooltip for Action Path explaining presets
                action_info = (
                    "Select a camera movement preset:\n"
                    "• examples/00: Forward Motion (Racing/Chase)\n"
                    "• examples/01: Panoramic Orbit\n"
                    "• examples/02: Complex Tracking (Pan + Tilt + Forward)"
                )
                action_pd = gr.Dropdown(
                    choices=["None"] + get_action_paths(), 
                    label="Action Path (Preset)", 
                    value="examples/00" if "examples/00" in get_action_paths() else "None", 
                    allow_custom_value=True,
                    info=action_info
                )
                custom_action_input = gr.Textbox(label="Custom Action Path", info="Path to a folder containing poses.npy and intrinsics.npy (Optional)")
            
            with gr.Row():
                resolution_input = gr.Dropdown(choices=["480P", "720P"], label="Resolution", value="480P", info="Higher resolution = slower generation & more VRAM.")
                frames_input = gr.Slider(minimum=1, maximum=161, step=4, value=49, label="Frames", info="Duration (16fps). 49 frames ≈ 3s.")
                steps_input = gr.Slider(minimum=10, maximum=100, step=1, value=40, label="Sampling Steps", info="Higher = better quality but slower.")

            generate_btn = gr.Button("Generate Video", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Output")
            video_output = gr.Video(label="Generated Video")
            
    generate_btn.click(
        fn=generate_video,
        inputs=[image_input, prompt_input, action_pd, custom_action_input, resolution_input, frames_input, steps_input],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False)
