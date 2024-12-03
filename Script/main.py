import torch
from torch.utils.data import DataLoader, random_split
from dataset import VolumeDataset
from SRCNN import SRCNN_3D
from FSRCNN import FSRCNN_3D
import argparse
import time
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from utils import calculate_metrics, visualize_images

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_type):

    writer = SummaryWriter(log_dir=f'runs/{model_type}')
    checkpoint_dir = f'models/{model_type}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        for low_res_inputs, high_res_targets in tqdm(train_loader, desc=f"{model_type} Epoch {epoch+1}/{num_epochs}", leave=False):
            low_res_inputs = low_res_inputs.to(device)
            high_res_targets = high_res_targets.to(device)

            if model_type == 'SRCNN':
                inputs_upsampled = F.interpolate(low_res_inputs, size=high_res_targets.shape[2:], mode='trilinear', align_corners=False)
                outputs = model(inputs_upsampled)
            elif model_type == 'FSRCNN':
                outputs = model(low_res_inputs)
                outputs = F.interpolate(outputs, size=high_res_targets.shape[2:], mode='trilinear', align_corners=False)

            loss = criterion(outputs, high_res_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), epoch)

        model.eval()
        ssim_total, psnr_total, mse_total = 0, 0, 0
        selected_images = random.sample(range(len(val_loader.dataset)), k=2)

        with torch.no_grad():
            for idx in selected_images:
                low_res_input, high_res_target = val_loader.dataset[idx]
                low_res_input = low_res_input.unsqueeze(0).to(device)
                high_res_target = high_res_target.unsqueeze(0).to(device)

                if model_type == 'SRCNN':
                    inputs_upsampled = F.interpolate(low_res_input, size=high_res_target.shape[2:], mode='trilinear', align_corners=False)
                    outputs = model(inputs_upsampled)
                elif model_type == 'FSRCNN':
                    outputs = model(low_res_input)
                    outputs = F.interpolate(outputs, size=high_res_target.shape[2:], mode='trilinear', align_corners=False)

                ssim_val, psnr_val, mse_val = calculate_metrics(outputs, high_res_target)
                ssim_total += ssim_val
                psnr_total += psnr_val
                mse_total += mse_val

                if model_type == 'SRCNN':
                    if (epoch + 1) % 10 == 0:
                        visualize_images(high_res_target[0], inputs_upsampled[0], outputs[0])
                elif model_type == 'FSRCNN':
                    if (epoch + 1) % 10 == 0:
                        visualize_images(high_res_target[0], low_res_input[0], outputs[0])



        avg_ssim = ssim_total / len(selected_images)
        avg_psnr = psnr_total / len(selected_images)
        avg_mse = mse_total / len(selected_images)
        writer.add_scalar('SSIM/val', avg_ssim, epoch)
        writer.add_scalar('PSNR/val', avg_psnr, epoch)
        writer.add_scalar('MSE/val', avg_mse, epoch)

        print(f'{model_type} Validation - Epoch [{epoch+1}/{num_epochs}], SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, MSE: {avg_mse:.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{model_type}_epoch_{epoch+1}.pth'))

    writer.close()

def main():
    parser = argparse.ArgumentParser(description="3D Super-Resolution Training")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model', type=str, choices=['SRCNN', 'FSRCNN'], required=True, help='Model to train: SRCNN or FSRCNN')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = VolumeDataset(root_dir=args.root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SRCNN3D() if args.model == 'SRCNN' else FSRCNN_3D()
    train_model(model, train_loader, val_loader, torch.nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=args.learning_rate), device, args.epochs, args.model)

if __name__ == '__main__':
    main()
