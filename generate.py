import torch
from diffusers import FluxPipeline

def run_flux_fast():
    print("Loading FLUX.1-schnell model...")
    
    # 4-step ultra-fast model
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    
    # Enable CPU offload to save GPU memory
    pipe.enable_model_cpu_offload()

    prompt = (
        "Cinematic macro shot of an antique mechanical wristwatch resting on wet moss, "
        "intricate gears exposed, water droplets reflecting soft morning light, 8k, photorealistic"
    )

    print("Generating image in 4 steps...")
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

    filename = "cinematic_output.png"
    image.save(filename)
    print(f"Done! Image saved as {filename}")

if __name__ == "__main__":
    run_flux_fast()
