# Simple demo showing how to use the model using torch.hub

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():

    # Load the hero model from torch.hub
    model = torch.hub.load(
        'nianticlabs/mvsanywhere',
        'mvsanywhere',
        trust_repo=True,
        pretrained=True,  # Load pre-trained weights
        dot_model=False,  # Use metadata-based cost volume
    )

    # Load images
    cur_image = Image.open('demo_assets/cur_image.png').convert('RGB')
    src_image = [
        Image.open(f'demo_assets/src_image_{i}.png').convert('RGB')
        for i in range(1, 8)
    ]

    # Load camera poses and intrinsics
    cameras = np.load('demo_assets/cameras.npy', allow_pickle=True).item()
    cur_pose = cameras['cur_pose']
    src_pose = cameras['src_pose']
    cur_intrinsics = cameras['cur_intrinsics']
    src_intrinsics = cameras['src_intrinsics']

    # Run the model
    with torch.inference_mode():
        output = model.preprocess_and_run(
            cur_image=cur_image,
            src_image=src_image,
            cur_pose=cur_pose,
            src_pose=src_pose,
            cur_intrinsics=cur_intrinsics,
            src_intrinsics=src_intrinsics,
            depth_range=None,  # Estimate depth range automatically
            num_refinement_steps=1,  # Do 1 refinement step (two steps in total)
            target_image_size=[480, 640],  # Resize the input images to this size
            device='cuda'
        )

    # Save the depth viz as png
    plt.imsave('simple_demo_invdepth.png', 1.0 / output['depth'].cpu().numpy(), cmap='jet')


if __name__ == '__main__':
    main()