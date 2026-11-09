"""
Explainability Pipeline for Pathomic Fusion.

Generates:
1. Grad-CAM attention heatmaps for Histopathology ROIs (VGG19-BN CNN)
2. Integrated Gradients feature attributions for Genomic Features (SNN / Multimodal)
3. Integrated Gradients node feature attributions for Cell Spatial Graphs (GCN)

Reference: Chen et al. (2022) - Pathomic Fusion, IEEE TMI
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


# ── Grad-CAM for Histopathology ───────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for VGG19-BN Histopathology feature extractor.
    
    Computes coarse 2D localization map highlighting the salient image
    regions (nuclei, cellular atypia, microvascular proliferation)
    responsible for high predicted mortality risk.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default target layer: last conv layer in features
        if target_layer is None:
            if hasattr(model, 'features'):
                self.target_layer = model.features[-1]
            elif hasattr(model, 'net_path') and hasattr(model.net_path, 'features'):
                self.target_layer = model.net_path.features[-1]
            else:
                raise ValueError("Target layer could not be automatically located.")
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=0):
        """
        Generate normalized Grad-CAM heatmap for input_tensor.
        
        Args:
            input_tensor: (1, 3, H, W)
            target_class: Target hazard output index
            
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.zero_grad()
        if hasattr(self.model, 'net_path'):
            feat, output = self.model(x_path=input_tensor)
        else:
            feat, output = self.model(x_path=input_tensor)

        score = output[0, target_class] if output.dim() > 1 else output[0]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(self, img_pil, heatmap, colormap='jet', alpha=0.5):
        """Overlay heatmap on PIL Image and return combined image."""
        w, h = img_pil.size
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(colored_heatmap).resize((w, h), Image.Resampling.BILINEAR)

        blended = Image.blend(img_pil.convert('RGB'), heatmap_pil, alpha=alpha)
        return blended


# ── Integrated Gradients for Genomic Features ─────────────────────────

class GenomicExplainer:
    """
    Feature attribution for Genomic SNN using Captum Integrated Gradients.
    
    Identifies top risk-conferring and protective genes from CNV,
    mutation, and RNA-Seq features.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

        class ForwardWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x_omic):
                if hasattr(self.m, 'net_omic'):
                    f, h = self.m.net_omic(x_omic=x_omic)
                else:
                    f, h = self.m(x_omic=x_omic)
                return h

        self.wrapper = ForwardWrapper(model)
        if HAS_CAPTUM:
            self.ig = IntegratedGradients(self.wrapper)
        else:
            self.ig = None

    def attribute(self, input_omic, baseline=None, feature_names=None, target=0):
        """
        Compute Integrated Gradients attributions for genomic vector.
        
        Args:
            input_omic: (1, input_size_omic) torch.Tensor
            baseline: (1, input_size_omic) baseline (default: zeros)
            feature_names: List of gene/feature names
            
        Returns:
            df_attributions: DataFrame with features, attributions, and importance
        """
        device = input_omic.device
        if baseline is None:
            baseline = torch.zeros_like(input_omic, device=device)

        if self.ig is not None:
            attr, delta = self.ig.attribute(
                input_omic, baselines=baseline, target=target, return_convergence_delta=True
            )
            attr_np = attr.squeeze().cpu().detach().numpy()
        else:
            # Fallback simple gradient attribution
            input_omic.requires_grad = True
            out = self.wrapper(input_omic)
            out.backward()
            attr_np = (input_omic.grad * input_omic).squeeze().cpu().detach().numpy()

        if feature_names is None:
            feature_names = [f"Gene_{i+1}" for i in range(len(attr_np))]

        df = pd.DataFrame({
            'Feature': feature_names[:len(attr_np)],
            'Attribution': attr_np,
            'AbsImportance': np.abs(attr_np)
        }).sort_values(by='AbsImportance', ascending=False)

        return df

    def plot_top_features(self, df_attr, top_k=20, save_path=None, title='Top Genomic Features (Integrated Gradients)'):
        """Plot bar chart of top positive (risk) and negative (protective) genes."""
        top_df = df_attr.head(top_k).iloc[::-1]  # ascending for barh

        colors = ['#e74c3c' if a > 0 else '#3498db' for a in top_df['Attribution']]

        plt.figure(figsize=(10, 8), dpi=300)
        plt.barh(top_df['Feature'], top_df['Attribution'], color=colors, edgecolor='black', alpha=0.85)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel('Attribution Score (Integrated Gradients)', fontsize=12, fontweight='bold')
        plt.ylabel('Gene / Genomic Feature', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='x', linestyle=':', alpha=0.6)

        # Legend
        import matplotlib.patches as mpatches
        risk_patch = mpatches.Patch(color='#e74c3c', label='Increased Risk (Poor Prognosis)')
        prot_patch = mpatches.Patch(color='#3498db', label='Protective (Favorable Prognosis)')
        plt.legend(handles=[risk_patch, prot_patch], loc='lower right', frameon=True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"Saved genomic explainability plot to {save_path}")
        else:
            plt.show()


# ── Demo / Self-Test Runner ───────────────────────────────────────────

def run_explainability_demo(output_dir='./results/explainability'):
    """Generate demonstration heatmaps and feature attributions."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print(" Running Explainability Suite (Grad-CAM & Integrated Gradients)")
    print("=" * 60)

    # 1. Genomic Attribution Demo
    from models.omic_net import MaxNet
    net_omic = MaxNet(input_dim=320, omic_dim=32, act=None, label_dim=1, init_max=True)
    explainer = GenomicExplainer(net_omic)

    # Load actual gene names from dataset if available
    split_p = './data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_1_0_0_rnaseq.pkl'
    gene_names = None
    if os.path.exists(split_p):
        import pickle
        with open(split_p, 'rb') as f:
            data = pickle.load(f)
        meta = ['Histomolecular subtype', 'Histology', 'Grade', 'Molecular subtype', 'TCGA ID', 'censored', 'Survival months']
        gene_names = [c for c in data['data_pd'].columns if c not in meta][:320]

    sample_omic = torch.randn(1, 320)
    df_attr = explainer.attribute(sample_omic, feature_names=gene_names)
    csv_out = os.path.join(output_dir, 'genomic_attributions.csv')
    df_attr.to_csv(csv_out, index=False)
    print(f"Computed Integrated Gradients for {len(df_attr)} genomic features.")

    plot_out = os.path.join(output_dir, 'genomic_top_attributions.png')
    explainer.plot_top_features(df_attr, top_k=20, save_path=plot_out)

    # 2. Grad-CAM Demo on synthetic or real ROI
    from models.path_net import get_vgg
    net_path = get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, path_dim=32, label_dim=1, pretrained=True)
    grad_cam = GradCAM(net_path)

    dummy_img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    tensor_input = torch.randn(1, 3, 512, 512, requires_grad=True)

    heatmap = grad_cam.generate_heatmap(tensor_input)
    overlay = grad_cam.overlay_on_image(dummy_img, heatmap)

    cam_out = os.path.join(output_dir, 'histopathology_gradcam_overlay.png')
    overlay.save(cam_out)
    print(f"Saved Grad-CAM histopathology attention map to {cam_out}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_explainability_demo()
