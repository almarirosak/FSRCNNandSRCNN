import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def visualize_images(original, upsampled, output):

    plt.figure(figsize=(12, 4))


    depth = original.shape[1]
    mid_slice = min(depth // 2, original.shape[1] - 1)
    plt.subplot(1, 3, 1)
    plt.imshow(original[0, mid_slice, :, :].cpu().permute(1, 2, 0).numpy(), cmap="gray")
    plt.title('Original Image')
    plt.axis('off')


    depth_upsampled = upsampled.shape[1]
    mid_slice_upsampled = min(depth_upsampled // 2, depth_upsampled - 1)
    plt.subplot(1, 3, 2)
    plt.imshow(upsampled[0, mid_slice_upsampled, :, :].cpu().permute(1, 2, 0).numpy(), cmap="gray")
    plt.title('Upsampled Image')
    plt.axis('off')

    depth_output = output.shape[1]
    mid_slice_output = min(depth_output // 2, depth_output - 1)
    plt.subplot(1, 3, 3)
    plt.imshow(output[0, mid_slice_output, :, :].cpu().permute(1, 2, 0).numpy(), cmap="gray")
    plt.title('Model Output')
    plt.axis('off')

    plt.show()


def calculate_metrics(output, target):

    output_np = output.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()


    min_dim = min(output_np.shape[-2:])
    if min_dim < 7:
        win_size = min_dim
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = 7

    ssim_value = ssim(output_np, target_np, data_range=target_np.max() - target_np.min(), win_size=win_size)
    mse_value = np.mean((output_np - target_np) ** 2)
    psnr_value = psnr(target_np, output_np, data_range=target_np.max() - target_np.min()) if mse_value > 0 else float('inf')

    return ssim_value, psnr_value, mse_value