from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

import matplotlib.pyplot as plt
import torch
import random


class Grad_CAM_implementaiton:

    def __init__(self, model, target_layers, test_dataset, mean, std, device):
        self.model = model
        self.mean = mean.item()
        self.std = std.item()
        self.device = device
        self.test_normal_images = random.sample([(image, label) for image, label in test_dataset if label == 0], 3)
        self.test_pneumonia_images = random.sample([(image, label) for image, label in test_dataset if label == 1], 3)
        self.target_layers = target_layers #model.features[3]

    def plot_pneumonia(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        self.model.eval()

        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
            )

        for i, (image, label) in enumerate(self.test_pneumonia_images):

            # Add batch dimension
            input_tensor = image.unsqueeze(0).to(self.device)

            # Prediction
            with torch.no_grad():
                logit = self.model(input_tensor)
                prob = torch.sigmoid(logit).item()
                pred = int(prob >= 0.5)

            # Grad-CAM for pneumonia class
            targets = [BinaryClassifierOutputTarget(1)]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            # ---- Undo normalization for visualization ----
            img = image.squeeze().cpu().numpy()

            img = img * self.std + self.mean
            img = img.clip(0, 1)

            # ---- Original ----
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title(
                f"Original\nTrue={label}, Pred={pred}, P={prob:.2f}"
            )
            axes[i, 0].axis("off")

            # ---- Grad-CAM ----
            im = axes[i, 1].imshow(
                grayscale_cam,
                cmap="jet",
                vmin=0,
                vmax=1
            )
            axes[i, 1].set_title("Grad-CAM")
            axes[i, 1].axis("off")

            # ---- Overlay ----
            axes[i, 2].imshow(img, cmap="gray")
            axes[i, 2].imshow(
                grayscale_cam,
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1
            )
            axes[i, 2].set_title("Grad-CAM overlay")
            axes[i, 2].axis("off")


        # One colorbar for all CAM images
        cbar = fig.colorbar(
            im,
            ax=axes[:, 1:].ravel().tolist(),
            fraction=0.02,
            pad=0.02
        )

        cbar.set_label("Grad-CAM activation")

        plt.show()

    def plot_normal(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        self.model.eval()

        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
            )

        for i, (image, label) in enumerate(self.test_normal_images):

            # Add batch dimension
            input_tensor = image.unsqueeze(0).to(self.device)

            # Prediction
            with torch.no_grad():
                logit = self.model(input_tensor)
                prob = torch.sigmoid(logit).item()
                pred = int(prob >= 0.5)

            # Grad-CAM for pneumonia class
            targets = [BinaryClassifierOutputTarget(0)]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            # ---- Undo normalization for visualization ----
            img = image.squeeze().cpu().numpy()

            img = img * self.std + self.mean
            img = img.clip(0, 1)

            # ---- Original ----
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title(
                f"Original\nTrue={label}, Pred={pred}, P={prob:.2f}"
            )
            axes[i, 0].axis("off")

            # ---- Grad-CAM ----
            im = axes[i, 1].imshow(
                grayscale_cam,
                cmap="jet",
                vmin=0,
                vmax=1
            )
            axes[i, 1].set_title("Grad-CAM")
            axes[i, 1].axis("off")

            # ---- Overlay ----
            axes[i, 2].imshow(img, cmap="gray")
            axes[i, 2].imshow(
                grayscale_cam,
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1
            )
            axes[i, 2].set_title("Grad-CAM overlay")
            axes[i, 2].axis("off")


        # One colorbar for all CAM images
        cbar = fig.colorbar(
            im,
            ax=axes[:, 1:].ravel().tolist(),
            fraction=0.02,
            pad=0.02
        )

        cbar.set_label("Grad-CAM activation")

        plt.show()


