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

        plt.figure(figsize=(10, 8), dpi=200)
        plt.barh(top_df['Feature'], top_df['Attribution'], color=colors, edgecolor='black', alpha=0.85)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel('Signed importance (Integrated Gradients)', fontsize=12, fontweight='bold')
        plt.ylabel('Gene / Genomic Feature', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='x', linestyle=':', alpha=0.6)

        # Legend
        import matplotlib.patches as mpatches
        risk_patch = mpatches.Patch(color='#e74c3c', label='Higher value → higher predicted risk')
        prot_patch = mpatches.Patch(color='#3498db', label='Higher value → lower predicted risk')
        plt.legend(handles=[risk_patch, prot_patch], loc='lower right', frameon=True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"Saved genomic explainability plot to {save_path}")
        else:
            plt.show()


# ── Explain the trained models ────────────────────────────────────────

def explain_genomic_model(exp_name='surv_15_rnaseq', model_name='omic',
                          task='surv', output_dir='./results/explainability',
                          checkpoints_dir='./checkpoints/TCGA_GBMLGG'):
    """
    Integrated Gradients for the trained genomic SNN.

    For every cross-validation fold, the trained network of that fold is
    explained on its own *test* patients (one genomic vector per patient).
    The baseline is the all-zero vector, i.e. the training-set mean of the
    standardised features. Attributions are averaged over patients and folds.

    Returns:
        DataFrame with the mean signed attribution and mean absolute
        attribution of every genomic feature, or None when no trained
        checkpoint is available.
    """
    import glob
    import pickle
    from config import parse_args
    from experiments import experiment_argv
    from models.networks import define_net
    from train import load_splits
    from datasets.data_loader import first_row_per_patient, valid_indices

    if not HAS_CAPTUM:
        print("captum is not installed; skipping genomic attribution.")
        return None

    opt = parse_args(experiment_argv(task, 'omic') + ['--gpu_ids', '-1'])
    opt.model_name = model_name
    split_file, cv_splits = load_splits(opt)
    with open(split_file, 'rb') as f:
        data_pd = pickle.load(f)['data_pd']
    meta = ['Histomolecular subtype', 'Histology', 'Grade', 'Molecular subtype',
            'TCGA ID', 'censored', 'Survival months']
    feature_names = [c for c in data_pd.columns if c not in meta]
    feature_names = feature_names[:opt.input_size_omic]

    ckpt_dir = os.path.join(checkpoints_dir, exp_name, model_name)
    absolute, slope_num, slope_den, n_patients, n_models = [], 0.0, 0.0, 0, 0
    for k in sorted(cv_splits):
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, f"{model_name}_{k}_s*.pt")))
        if not ckpts:
            continue
        test = cv_splits[k]['test']
        idx = first_row_per_patient(test['x_patname'], valid_indices(test, task))
        x = torch.tensor(np.asarray(test['x_omic'])[idx], dtype=torch.float32)
        for ckpt in ckpts:
            model = define_net(opt)
            model.load_state_dict(torch.load(
                ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
            explainer = GenomicExplainer(model)
            # survival: single risk output; grade: attribute the Grade IV class
            target = 0 if task == 'surv' else 2
            attr = explainer.ig.attribute(
                x, baselines=torch.zeros_like(x), target=target, n_steps=50
            ).detach().numpy()
            absolute.append(np.abs(attr).mean(axis=0))
            # direction of effect: slope of attribution against feature value
            slope_num = slope_num + (attr * x.numpy()).sum(axis=0)
            slope_den = slope_den + (x.numpy() ** 2).sum(axis=0)
            n_models += 1
        n_patients += len(idx)

    if not absolute:
        print(f"No trained checkpoints found in {ckpt_dir}; nothing to explain.")
        return None

    importance = np.mean(absolute, axis=0)
    direction = np.sign(slope_num / np.maximum(slope_den, 1e-12))
    df = pd.DataFrame({
        'Feature': feature_names,
        # signed importance: + when a higher feature value raises the
        # predicted risk, - when it lowers it
        'Attribution': importance * direction,
        'AbsImportance': importance,
    }).sort_values('AbsImportance', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'genomic_attributions.csv'), index=False)
    print(f"Integrated Gradients: {n_models} trained models, "
          f"{n_patients} test-patient evaluations, {len(df)} features.")
    GenomicExplainer(model).plot_top_features(
        df, top_k=20,
        save_path=os.path.join(output_dir, 'genomic_top_attributions.png'),
        title='Top genomic features — trained SNN, test patients of all folds\n'
              '(Integrated Gradients: mean |attribution| to predicted risk)')
    return df


def explain_histology_model(output_dir='./results/explainability',
                            dataroot='./data/TCGA_GBMLGG',
                            exp_name='surv_15_rnaseq', model_name='path',
                            checkpoints_dir='./checkpoints/TCGA_GBMLGG'):
    """
    Grad-CAM for a trained histology CNN on a real ROI.

    Requires both the ROI images and a histology CNN trained end-to-end on
    them (`--mode path --use_vgg_features 0`). Returns None otherwise: a
    heat-map from an untrained network or a synthetic image carries no
    information, so none is produced.
    """
    import glob
    roi_dir = os.path.join(dataroot, 'all_st_patches_512')
    images = sorted(glob.glob(os.path.join(roi_dir, '*.png')))
    ckpts = sorted(glob.glob(os.path.join(
        checkpoints_dir, exp_name, model_name, f"{model_name}_1_s*.pt")))
    state = (torch.load(ckpts[0], map_location='cpu', weights_only=False)
             if ckpts else None)
    trained_cnn = state is not None and any(
        key.startswith('features.') for key in state['model_state_dict'])
    if not images or not trained_cnn:
        print("Grad-CAM skipped: it needs the ROI images under "
              f"{roi_dir} and a histology CNN trained end-to-end on them.")
        return None

    from torchvision import transforms
    from models.path_net import get_vgg
    net = get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True,
                  path_dim=32, label_dim=1, pretrained=False)
    net.load_state_dict(state['model_state_dict'], strict=False)
    img = Image.open(images[0]).convert('RGB')
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(img).unsqueeze(0).requires_grad_(True)
    grad_cam = GradCAM(net)
    overlay = grad_cam.overlay_on_image(img, grad_cam.generate_heatmap(tensor))
    out = os.path.join(output_dir, 'histopathology_gradcam_overlay.png')
    os.makedirs(output_dir, exist_ok=True)
    overlay.save(out)
    print(f"Saved Grad-CAM overlay for {os.path.basename(images[0])} to {out}")
    return out


def run_explainability(output_dir='./results/explainability'):
    """Explain the trained models (genomic attributions, Grad-CAM)."""
    print("\n" + "=" * 60)
    print(" Explainability (Integrated Gradients & Grad-CAM)")
    print("=" * 60)
    explain_genomic_model(output_dir=output_dir)
    explain_histology_model(output_dir=output_dir)
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_explainability()
