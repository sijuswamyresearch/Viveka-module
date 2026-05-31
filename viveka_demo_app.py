import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies for advanced metrics
PYIQA_AVAILABLE = False
PYWT_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    import torch
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

# ====================== PAGE CONFIGURATION ======================
st.set_page_config(
    page_title="Viveka Post-Processing Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
    }
    .improvement-positive {
        color: #2e7d32;
        font-weight: bold;
    }
    .improvement-negative {
        color: #c62828;
        font-weight: bold;
    }
    .quality-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .quality-good {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .info-note {
        background: #d4edda;
        border: 1px solid #2196F3;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ====================== HELPER FUNCTIONS ======================
def normalize_display(image):
    """Normalize ANY image to 0-1 range for consistent display."""
    image = np.squeeze(image).astype(np.float32)
    return np.clip(image, 0, 1)

def ensure_same_shape(img1, img2):
    """Ensure two images have the same shape by resizing to the smaller dimensions."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        min_h = min(h1, h2)
        min_w = min(w1, w2)
        if h1 != min_h or w1 != min_w:
            img1 = cv2.resize(img1, (min_w, min_h))
        if h2 != min_h or w2 != min_w:
            img2 = cv2.resize(img2, (min_w, min_h))
    
    return img1, img2

def standardize_image(image):
    """Standardize image to 0-1 range."""
    image = image.astype(np.float64)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0, 1).astype(np.float32)

def to_uint8(image):
    """Convert to uint8."""
    image = standardize_image(image)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

def to_tensor_rgb(image):
    """Convert to torch tensor (RGB)."""
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def load_image_from_uploaded(uploaded_file):
    """Load image from uploaded file and convert to grayscale numpy array."""
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array

def estimate_input_quality(image):
    """
    Estimate the quality of the denoised input image.
    Returns a quality score and category.
    Higher sharpness + edge density = better quality = less refinement needed.
    """
    img_uint8 = to_uint8(image)
    
    # Edge sharpness (95th percentile of gradient)
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sharpness = np.percentile(grad_mag, 95)
    
    # Edge density
    threshold = np.percentile(grad_mag, 85)
    edge_density = np.mean(grad_mag > threshold)
    
    # Local contrast (std of 7x7 patches)
    local_std = np.std(cv2.GaussianBlur(img_uint8.astype(np.float32), (7, 7), 1.0))
    
    # Quality score (0-100, higher = better quality)
    quality_score = min(100, (sharpness * 0.5 + edge_density * 30 + local_std * 0.2))
    
    if quality_score > 60:
        category = "Excellent"
        recommendation = "Input is already high quality. Use gentle refinement or skip Viveka."
    elif quality_score > 40:
        category = "Good"
        recommendation = "Input is good quality. Balanced refinement recommended."
    elif quality_score > 20:
        category = "Fair"
        recommendation = "Input needs moderate enhancement. Balanced or strong refinement."
    else:
        category = "Poor"
        recommendation = "Input is over-smoothed. Strong refinement recommended."
    
    return {
        'score': quality_score,
        'category': category,
        'recommendation': recommendation,
        'sharpness': sharpness,
        'edge_density': edge_density,
        'local_std': local_std
    }


# ====================== VIVEKA REFINER (Quality-Aware, Pre/Post-CLAHE Output) ======================
class VivekaRefiner:
    def __init__(self, guide_thresh=1.5, input_thresh=3.0, spins=2,
                 use_adaptive_gains=True, use_pathology_preservation=True,
                 use_uncertainty_guidance=True,
                 use_edge_protection=True,
                 use_dct_detail=True,
                 refinement_strength='balanced',
                 smooth_weight=0.1, fidelity_weight=1.0, clinical_weight=0.05,
                 auto_quality_adapt=True):
        self.Tg = guide_thresh
        self.Ti = input_thresh
        self.spins = spins
        self.use_adaptive_gains = use_adaptive_gains
        self.use_pathology_preservation = use_pathology_preservation
        self.use_uncertainty_guidance = use_uncertainty_guidance
        self.use_edge_protection = use_edge_protection
        self.use_dct_detail = use_dct_detail
        self.refinement_strength = refinement_strength
        self.smooth_weight = smooth_weight
        self.fidelity_weight = fidelity_weight
        self.clinical_weight = clinical_weight
        self.auto_quality_adapt = auto_quality_adapt
        self.effective_Tg = guide_thresh
        self.effective_Ti = input_thresh

        if refinement_strength == 'gentle':
            self.base_clip_limit = 0.8
        elif refinement_strength == 'strong':
            self.base_clip_limit = 1.5
        else:
            self.base_clip_limit = 1.2
        
        self.clahe = cv2.createCLAHE(clipLimit=self.base_clip_limit, tileGridSize=(8,8))

        self.default_bone_gain = 1.0
        self.default_lung_gain = 0.3
        self.default_bg_gain = 0.0

    def anscombe_forward(self, img):
        return 2.0 * np.sqrt(np.maximum(img + (3.0/8.0), 0))

    def anscombe_inverse(self, img):
        return np.maximum(0, (img / 2.0)**2 - (3.0/8.0))

    def adapt_to_input_quality(self, denoised_image):
        """
        Adapt refinement parameters based on input image quality.
        High quality inputs → gentler refinement
        Low quality inputs → stronger refinement
        """
        if not self.auto_quality_adapt:
            self.effective_Tg = self.Tg
            self.effective_Ti = self.Ti
            self.clahe = cv2.createCLAHE(clipLimit=self.base_clip_limit, tileGridSize=(8,8))
            return
        
        quality_info = estimate_input_quality(denoised_image)
        quality_score = quality_info['score']
        
        if quality_score > 60:  # Excellent quality
            self.effective_Tg = self.Tg * 1.8
            self.effective_Ti = self.Ti * 1.8
            clip_limit = max(0.5, self.base_clip_limit * 0.6)
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        elif quality_score > 40:  # Good quality
            self.effective_Tg = self.Tg * 1.3
            self.effective_Ti = self.Ti * 1.3
            clip_limit = max(0.7, self.base_clip_limit * 0.85)
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        elif quality_score > 20:  # Fair quality
            self.effective_Tg = self.Tg
            self.effective_Ti = self.Ti
            self.clahe = cv2.createCLAHE(clipLimit=self.base_clip_limit, tileGridSize=(8,8))
        else:  # Poor quality
            self.effective_Tg = self.Tg * 0.7
            self.effective_Ti = self.Ti * 0.7
            clip_limit = min(2.0, self.base_clip_limit * 1.3)
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        
        return quality_info

    def compute_adaptive_gains(self, image, bone_mask, lung_mask, model_type='X-GAN',
                               baseline_quality=None, baseline_piqe=None):
        bone_pixels = image[bone_mask > 0.5]
        if len(bone_pixels) > 0:
            bone_density = np.percentile(bone_pixels, 95)
            bone_gain = 0.8 + 0.4 * bone_density
            bone_gain = np.clip(bone_gain, 0.8, 1.2)
        else:
            bone_gain = self.default_bone_gain

        lung_pixels = image[lung_mask > 0.5]
        if len(lung_pixels) > 0:
            lung_texture = np.std(lung_pixels)
            base_model = model_type.replace('+Viveka', '')

            if base_model == 'CGAN':
                gain_multiplier = 0.5
                max_gain = 0.35
                min_gain = 0.15
            elif base_model in ['RIDNet', 'CBDNet', 'X-ReCNN']:
                gain_multiplier = 1.5
                max_gain = 0.55
                min_gain = 0.25
            else:
                if self.refinement_strength == 'gentle':
                    gain_multiplier = 0.3
                    max_gain = 0.30
                    min_gain = 0.12
                elif self.refinement_strength == 'strong':
                    gain_multiplier = 0.8
                    max_gain = 0.45
                    min_gain = 0.25
                else:
                    if baseline_piqe is not None and baseline_piqe < 40:
                        gain_multiplier = 0.3
                    else:
                        gain_multiplier = 0.5
                    max_gain = 0.40
                    min_gain = 0.18

            lung_gain = min_gain + gain_multiplier * 1.5 * lung_texture
            lung_gain = np.clip(lung_gain, min_gain, max_gain)
        else:
            lung_gain = self.default_lung_gain

        return bone_gain, lung_gain, self.default_bg_gain

    def compute_uncertainty_map(self, gan_output, noisy_input):
        gan_output, noisy_input = ensure_same_shape(gan_output, noisy_input)

        kernel_size = 7
        local_mean = cv2.GaussianBlur(gan_output, (kernel_size, kernel_size), 1.0)
        local_var = cv2.GaussianBlur(gan_output**2, (kernel_size, kernel_size), 1.0) - local_mean**2
        local_var = np.maximum(local_var, 0)
        global_var = np.var(gan_output)
        if global_var > 1e-10:
            uncertainty = np.clip(local_var / global_var, 0, 1)
        else:
            uncertainty = np.zeros_like(gan_output)
        diff_from_noisy = np.abs(gan_output - noisy_input)
        uncertainty = 0.5 * uncertainty + 0.5 * np.clip(diff_from_noisy, 0, 1)
        return uncertainty

    def pathology_preservation_loss(self, refined, original):
        refined, original = ensure_same_shape(refined, original)
        refined_uint8 = (np.clip(refined, 0, 1) * 255).astype(np.uint8)
        original_uint8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
        sobel_refined = cv2.Sobel(refined_uint8, cv2.CV_64F, 1, 1, ksize=3)
        sobel_original = cv2.Sobel(original_uint8, cv2.CV_64F, 1, 1, ksize=3)
        edge_diff = np.mean(np.abs(sobel_refined - sobel_original)) / 255.0
        laplacian_refined = cv2.Laplacian(refined_uint8, cv2.CV_64F)
        laplacian_original = cv2.Laplacian(original_uint8, cv2.CV_64F)
        highfreq_diff = np.mean(np.abs(laplacian_refined - laplacian_original)) / 255.0
        return (edge_diff + highfreq_diff) / 2.0

    def run_oracle_dct(self, gan_vst, noisy_vst):
        if not self.use_dct_detail:
            return gan_vst

        gan_vst, noisy_vst = ensure_same_shape(gan_vst, noisy_vst)
        h, w = gan_vst.shape
        
        if h < 8 or w < 8:
            return gan_vst
        
        output = np.zeros_like(gan_vst)
        weights = np.zeros_like(gan_vst)
        P = 8
        stride = 4

        for y in range(0, h - P + 1, stride):
            for x in range(0, w - P + 1, stride):
                patch_g = gan_vst[y:y+P, x:x+P].copy()
                patch_n = noisy_vst[y:y+P, x:x+P].copy()
                dct_g = cv2.dct(patch_g.astype(np.float32))
                dct_n = cv2.dct(patch_n.astype(np.float32))
                # Use effective thresholds (quality-adapted)
                mask = np.logical_or(np.abs(dct_g) > self.effective_Tg, 
                                    np.abs(dct_n) > self.effective_Ti)
                dct_filtered = dct_n * mask.astype(np.float32)
                output[y:y+P, x:x+P] += cv2.idct(dct_filtered)
                weights[y:y+P, x:x+P] += 1.0

        valid_mask = weights > 0
        output[valid_mask] /= weights[valid_mask]
        output[~valid_mask] = gan_vst[~valid_mask]

        return output.astype(np.float32)

    def refine(self, gan_prediction, original_noisy_input, model_type='X-GAN',
               baseline_quality=None, baseline_piqe=None):
        """
        Refine the denoised image.
        Returns:
            final_pre_clahe: Refined image BEFORE CLAHE (for metric computation)
            final_display: Refined image AFTER CLAHE (for visual display)
            quality_info: Input quality assessment
        """
        manas_clean = np.clip(np.squeeze(gan_prediction).astype(np.float32), 0, 1)
        noisy_input = np.clip(np.squeeze(original_noisy_input).astype(np.float32), 0, 1)
        
        manas_clean, noisy_input = ensure_same_shape(manas_clean, noisy_input)
        
        # Auto-adapt parameters based on input quality
        quality_info = self.adapt_to_input_quality(manas_clean)

        gan_scaled = manas_clean * 255.0
        noisy_scaled = noisy_input * 255.0

        gan_scaled, noisy_scaled = ensure_same_shape(gan_scaled, noisy_scaled)

        gan_vst = self.anscombe_forward(gan_scaled)
        noisy_vst = self.anscombe_forward(noisy_scaled)

        shifts = [(0,0), (4,4)]
        acc = np.zeros_like(gan_vst, dtype=np.float32)
        for dy, dx in shifts:
            rolled_gan = np.roll(gan_vst, (dy,dx), (0,1))
            rolled_noisy = np.roll(noisy_vst, (dy,dx), (0,1))
            res = self.run_oracle_dct(rolled_gan, rolled_noisy)
            acc += np.roll(res, (-dy,-dx), (0,1))
        viveka_sharp = self.anscombe_inverse(acc / len(shifts)) / 255.0
        viveka_sharp = np.clip(viveka_sharp, 0, 1).astype(np.float32)

        if viveka_sharp.shape != manas_clean.shape:
            viveka_sharp = cv2.resize(viveka_sharp, (manas_clean.shape[1], manas_clean.shape[0]))

        manas_uint8 = (manas_clean * 255).astype(np.uint8)
        edges = cv2.Canny(manas_uint8, 30, 100)
        bone_mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1).astype(np.float32) / 255.0
        bone_mask = cv2.GaussianBlur(bone_mask, (0,0), 2.0)

        if bone_mask.shape != manas_clean.shape:
            bone_mask = cv2.resize(bone_mask, (manas_clean.shape[1], manas_clean.shape[0]))

        bg_mask = (manas_clean < 0.05).astype(np.float32)
        bg_mask = cv2.dilate(bg_mask, np.ones((3,3), np.uint8))

        if bg_mask.shape != manas_clean.shape:
            bg_mask = cv2.resize(bg_mask, (manas_clean.shape[1], manas_clean.shape[0]))

        lung_mask = 1.0 - np.maximum(bone_mask, bg_mask)

        if self.use_adaptive_gains:
            bone_gain, lung_gain, bg_gain = self.compute_adaptive_gains(
                manas_clean, bone_mask, lung_mask, model_type, baseline_quality, baseline_piqe
            )
        else:
            bone_gain, lung_gain, bg_gain = self.default_bone_gain, self.default_lung_gain, self.default_bg_gain

        if self.use_uncertainty_guidance:
            uncertainty = self.compute_uncertainty_map(manas_clean, noisy_input)
            uncertainty_weight = 1.0 + uncertainty
        else:
            uncertainty_weight = np.ones_like(manas_clean)

        if self.use_edge_protection:
            manas_uint8_edge = (manas_clean * 255).astype(np.uint8)
            sobel_x = cv2.Sobel(manas_uint8_edge, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(manas_uint8_edge, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x**2 + sobel_y**2) / 255.0
            protection_map = np.exp(-20.0 * grad_mag)
        else:
            protection_map = np.ones_like(manas_clean)

        weight_map = np.ones_like(manas_clean) * lung_gain * uncertainty_weight * protection_map
        weight_map = weight_map * (1.0 - bg_mask)
        weight_map = np.maximum(weight_map, bone_mask * bone_gain)

        detail_layer = viveka_sharp - manas_clean
        final_image = manas_clean + (detail_layer * weight_map)

        if self.use_pathology_preservation:
            pathology_loss = self.pathology_preservation_loss(final_image, manas_clean)
            if pathology_loss > 0.1:
                correction_factor = 1.0 - (pathology_loss - 0.1) * 2.0
                correction_factor = np.clip(correction_factor, 0.5, 1.0)
                final_image = manas_clean + (detail_layer * weight_map * correction_factor)

        # ===== PRE-CLAHE OUTPUT (for metric computation) =====
        final_pre_clahe = np.clip(final_image, 0, 1).astype(np.float32)
        
        # ===== POST-CLAHE OUTPUT (for visual display) =====
        final_uint8 = (final_pre_clahe * 255).astype(np.uint8)
        final_display = self.clahe.apply(final_uint8).astype(np.float32) / 255.0
        
        return final_pre_clahe, final_display, quality_info


# ====================== EVALUATION METRICS ======================

def compute_niqe(image, niqe_metric=None):
    """Compute NIQE - lower is better."""
    image_std = standardize_image(image)
    if niqe_metric is not None and PYIQA_AVAILABLE:
        try:
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = niqe_metric(tensor.to(niqe_metric.device))
            return score.item()
        except Exception:
            pass
    return 50.0

def compute_brisque(image, brisque_metric=None):
    """Compute BRISQUE - lower is better."""
    image_std = standardize_image(image)
    if brisque_metric is not None and PYIQA_AVAILABLE:
        try:
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = brisque_metric(tensor.to(brisque_metric.device))
            return score.item()
        except Exception:
            pass
    return 50.0

def compute_piqe(image, piqe_metric=None):
    """Compute PIQE - lower is better."""
    image_std = standardize_image(image)
    if piqe_metric is not None and PYIQA_AVAILABLE:
        try:
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = piqe_metric(tensor.to(piqe_metric.device))
            return score.item()
        except Exception:
            pass
    return 50.0

def compute_lassim(image, levels=4):
    """Compute Laplacian Pyramid SSIM - higher is better."""
    image_std = standardize_image(image)
    
    if image_std.shape[0] < 16 or image_std.shape[1] < 16:
        return 0.5
    
    pyramid = []
    current = image_std.copy()
    
    for level in range(levels):
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        if current.shape[0] >= 4 and current.shape[1] >= 4:
            downsampled = cv2.resize(blurred, (current.shape[1] // 2, current.shape[0] // 2))
            upsampled = cv2.resize(blurred, (current.shape[1], current.shape[0]))
        else:
            upsampled = blurred
            downsampled = current
        laplacian = current - upsampled
        pyramid.append(laplacian)
        if downsampled.shape[0] >= 4 and downsampled.shape[1] >= 4:
            current = downsampled
        else:
            break
    pyramid.append(current)
    
    lassim_score = 0
    total_weight = 0
    
    for i in range(len(pyramid) - 1):
        img1 = pyramid[i]
        img2 = cv2.resize(pyramid[i+1], (img1.shape[1], img1.shape[0]))
        
        try:
            ssim_val = ssim(img1, img2, data_range=max(1e-10, img1.max() - img1.min()))
        except Exception:
            ssim_val = 0.5
        
        weight = 0.5 ** i
        lassim_score += ssim_val * weight
        total_weight += weight
    
    if total_weight > 0:
        lassim_score = lassim_score / total_weight
    else:
        lassim_score = 0.5
    
    return float(np.clip(lassim_score, 0, 1))

def compute_haarpsimed(image):
    """Compute HaarPSI-based Medical Image Quality Metric - higher is better."""
    if not PYWT_AVAILABLE:
        return 0.5
    
    image_std = standardize_image(image)
    
    if image_std.shape[0] < 8 or image_std.shape[1] < 8:
        return 0.5
    
    try:
        coeffs = pywt.wavedec2(image_std, 'haar', level=min(3, int(np.log2(min(image_std.shape)))-1))
        
        scores = []
        weights = [0.5, 0.3, 0.2]
        
        for level in range(1, min(4, len(coeffs))):
            cA = coeffs[0]
            cH, cV, cD = coeffs[level]
            
            energy_H = np.sum(cH ** 2)
            energy_V = np.sum(cV ** 2)
            energy_D = np.sum(cD ** 2)
            
            structural_score = (energy_H + energy_V + energy_D) / (np.sum(cA ** 2) + 1e-7)
            scores.append(structural_score * weights[level-1])
        
        if len(scores) > 0:
            haarpsi_score = np.sum(scores)
            alpha = 5.8
            C = 5
            haarpesimed = 1.0 / (1.0 + alpha * np.exp(-C * haarpsi_score))
        else:
            haarpesimed = 0.5
        
        return float(np.clip(haarpesimed, 0, 1))
    except Exception:
        return 0.5

def compute_edge_density(image):
    """Compute edge density - higher indicates more edge content."""
    image_uint8 = to_uint8(image)
    sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold = np.percentile(edge_mag, 85)
    return float(np.mean(edge_mag > threshold))

def compute_mean_gradient(image):
    """Compute mean gradient magnitude - higher indicates sharper edges."""
    image_uint8 = to_uint8(image)
    sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

def compute_sharpness(image):
    """Compute sharpness using 95th percentile of gradient magnitude."""
    image_uint8 = to_uint8(image)
    sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.percentile(grad_mag, 95))

def compute_edge_preservation_index(ref_image, test_image):
    """EPI - correlation of edge magnitudes between reference and test."""
    ref_image, test_image = ensure_same_shape(ref_image, test_image)
    ref_uint8 = to_uint8(ref_image)
    test_uint8 = to_uint8(test_image)
    
    sobel_ref = cv2.Sobel(ref_uint8, cv2.CV_64F, 1, 1, ksize=3)
    sobel_test = cv2.Sobel(test_uint8, cv2.CV_64F, 1, 1, ksize=3)
    
    flat_ref = np.abs(sobel_ref).flatten()
    flat_test = np.abs(sobel_test).flatten()
    
    var_ref = np.var(flat_ref)
    var_test = np.var(flat_test)
    if var_ref < 1e-10 or var_test < 1e-10:
        return 0.0
    corr = np.corrcoef(flat_ref, flat_test)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def compute_psnr_relative(img1, img2):
    """PSNR for relative comparison between two images."""
    img1, img2 = ensure_same_shape(img1, img2)
    img1 = standardize_image(img1)
    img2 = standardize_image(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))

def compute_ssim_relative(img1, img2):
    """SSIM for relative comparison between two images."""
    img1, img2 = ensure_same_shape(img1, img2)
    img1 = standardize_image(img1)
    img2 = standardize_image(img2)
    data_range = max(1e-10, max(img1.max(), img2.max()) - min(img1.min(), img2.min()))
    return float(ssim(img1, img2, data_range=data_range))


# ====================== CLINICAL QUALITY METRICS ======================

def compute_cnr(image, noisy_input=None, bone_mask=None, background_mask=None):
    """ROI-Based Contrast-to-Noise Ratio. Higher = better tissue differentiation."""
    image_std = standardize_image(image)
    
    if bone_mask is None or background_mask is None:
        mask_source = standardize_image(noisy_input) if noisy_input is not None else image_std
        mask_source, image_std = ensure_same_shape(mask_source, image_std)
        mask_uint8 = (mask_source * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bone_mask = cv2.dilate(edges, kernel, iterations=1).astype(np.float32) / 255.0
        bone_mask = cv2.GaussianBlur(bone_mask, (5, 5), 2.0)
        bone_mask = cv2.resize(bone_mask, (image_std.shape[1], image_std.shape[0]))
        background_mask = (mask_source < 0.05).astype(np.float32)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background_mask = cv2.dilate(background_mask, kernel_bg, iterations=1)
        background_mask = cv2.resize(background_mask, (image_std.shape[1], image_std.shape[0]))
    
    bone_pixels = image_std[bone_mask > 0.3]
    bg_pixels = image_std[background_mask > 0.3]
    
    if len(bone_pixels) < 10 or len(bg_pixels) < 10:
        return 0.0
    
    mu_bone = np.mean(bone_pixels)
    mu_bg = np.mean(bg_pixels)
    sigma_bg = np.std(bg_pixels)
    
    if sigma_bg < 1e-7:
        return 0.0
    
    return float(np.abs(mu_bone - mu_bg) / sigma_bg)


def compute_fwhm_validated(image, noisy_input, num_edges=5, search_radius=30,
                           edge_contrast_threshold=0.12, check_ringing=True):
    """Edge Spread Function FWHM. Lower = sharper edges."""
    image_std = standardize_image(image)
    noisy_std = standardize_image(noisy_input)
    image_std, noisy_std = ensure_same_shape(image_std, noisy_std)
    
    image_uint8 = (image_std * 255).astype(np.uint8)
    noisy_uint8 = (noisy_std * 255).astype(np.uint8)
    
    noisy_edges = cv2.Canny(noisy_uint8, 50, 150)
    image_edges = cv2.Canny(image_uint8, 50, 150)
    valid_edges = noisy_edges & image_edges
    edge_y, edge_x = np.where(valid_edges > 0)
    
    if len(edge_x) < 10:
        return float('inf'), 0
    
    fwhm_values = []
    h, w = image_std.shape
    np.random.seed(42)
    
    max_attempts = num_edges * 10
    attempts = 0
    
    while len(fwhm_values) < num_edges and attempts < max_attempts:
        attempts += 1
        idx = np.random.randint(0, len(edge_x))
        ex, ey = edge_x[idx], edge_y[idx]
        
        y_start, y_end = max(0, ey-8), min(h, ey+9)
        x_start, x_end = max(0, ex-8), min(w, ex+9)
        local_noisy = noisy_std[y_start:y_end, x_start:x_end]
        
        if local_noisy.max() - local_noisy.min() < edge_contrast_threshold:
            continue
        
        profile_y_start, profile_y_end = max(0, ey-search_radius), min(h, ey+search_radius)
        if profile_y_end - profile_y_start < 10:
            continue
        
        profile = image_std[profile_y_start:profile_y_end, ex].astype(np.float64)
        if len(profile) < 10:
            continue
        
        prof_min, prof_max = profile.min(), profile.max()
        if prof_max - prof_min < 0.05:
            continue
        
        profile_norm = (profile - prof_min) / (prof_max - prof_min + 1e-10)
        above_half = profile_norm > 0.5
        
        if np.all(above_half) or not np.any(above_half):
            continue
        
        transitions = np.diff(above_half.astype(int))
        rise_idx = np.where(transitions == 1)[0]
        fall_idx = np.where(transitions == -1)[0]
        
        if len(rise_idx) == 0 or len(fall_idx) == 0:
            continue
        
        rise = rise_idx[0]
        fall_candidates = fall_idx[fall_idx > rise]
        if len(fall_candidates) == 0:
            continue
        fall = fall_candidates[0]
        
        fwhm_pixels = fall - rise
        if fwhm_pixels < 1:
            continue
        
        if check_ringing:
            pre_edge = profile_norm[max(0, rise-8):rise]
            post_edge = profile_norm[fall:min(len(profile_norm), fall+9)]
            has_ringing = False
            if len(pre_edge) > 2 and (np.max(pre_edge) > 1.08 or np.min(pre_edge) < -0.08):
                has_ringing = True
            if len(post_edge) > 2 and (np.max(post_edge) > 1.08 or np.min(post_edge) < -0.08):
                has_ringing = True
            if has_ringing:
                continue
        
        fwhm_values.append(fwhm_pixels)
    
    if len(fwhm_values) == 0:
        return float('inf'), 0
    
    return float(np.median(fwhm_values)), len(fwhm_values)


def compute_glcm_texture(image, distances=[1, 2], levels=64):
    """GLCM texture analysis. Returns: homogeneity, entropy, contrast, correlation."""
    image_std = standardize_image(image)
    
    lung_mask = np.ones_like(image_std, dtype=np.float32)
    lung_mask[image_std < 0.04] = 0.0
    lung_mask[image_std > 0.96] = 0.0
    
    masked = image_std.copy()
    masked[lung_mask < 0.3] = 0
    
    mask_pixels = masked[lung_mask >= 0.3]
    if len(mask_pixels) < 100:
        return 0.0, 0.0, 0.0, 0.0
    
    p_min, p_max = mask_pixels.min(), mask_pixels.max()
    if p_max - p_min < 1e-7:
        return 1.0, 0.0, 0.0, 1.0
    
    quantized = np.zeros_like(masked, dtype=np.uint8)
    quantized[lung_mask >= 0.3] = ((masked[lung_mask >= 0.3] - p_min) / 
                                     (p_max - p_min + 1e-10) * (levels - 1)).astype(np.uint8)
    
    try:
        glcm = graycomatrix(quantized, distances=distances, 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=levels, symmetric=True, normed=True)
        
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        glcm_flat = glcm.flatten()
        glcm_nonzero = glcm_flat[glcm_flat > 0]
        if len(glcm_nonzero) > 0:
            entropy = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
            max_entropy = np.log2(levels * levels)
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy_norm = 0.0
        
        return float(homogeneity), float(entropy_norm), float(contrast), float(correlation)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def compute_ttpi(denoised_image, refined_image):
    """Tissue Texture Preservation Index. ≈1.0 = ideal preservation."""
    denoised_image, refined_image = ensure_same_shape(denoised_image, refined_image)
    denoised_std = standardize_image(denoised_image)
    refined_std = standardize_image(refined_image)
    
    denoised_uint8 = (denoised_std * 255).astype(np.uint8)
    refined_uint8 = (refined_std * 255).astype(np.uint8)
    
    lap_denoised = cv2.Laplacian(denoised_uint8, cv2.CV_64F)
    lap_refined = cv2.Laplacian(refined_uint8, cv2.CV_64F)
    
    var_denoised = np.var(lap_denoised)
    var_refined = np.var(lap_refined)
    
    if var_denoised < 0.01:
        if var_refined > 0.01:
            return 999.0
        else:
            return 1.0
    
    ttpi = var_refined / var_denoised
    return float(np.clip(ttpi, 0.01, 100.0))


def compute_scp(denoised_image, refined_image):
    """Structural Content Preservation. ≈1.0 = perfect structural preservation."""
    denoised_image, refined_image = ensure_same_shape(denoised_image, refined_image)
    denoised_std = standardize_image(denoised_image)
    refined_std = standardize_image(refined_image)
    
    denoised_uint8 = (denoised_std * 255).astype(np.uint8)
    refined_uint8 = (refined_std * 255).astype(np.uint8)
    
    sobel_x_d = cv2.Sobel(denoised_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_d = cv2.Sobel(denoised_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_d = np.sqrt(sobel_x_d**2 + sobel_y_d**2)
    
    sobel_x_r = cv2.Sobel(refined_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_r = cv2.Sobel(refined_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_r = np.sqrt(sobel_x_r**2 + sobel_y_r**2)
    
    edge_d_flat = edge_d.flatten()
    edge_r_flat = edge_r.flatten()
    
    edge_d_norm = (edge_d_flat - np.mean(edge_d_flat)) / (np.std(edge_d_flat) + 1e-7)
    edge_r_norm = (edge_r_flat - np.mean(edge_r_flat)) / (np.std(edge_r_flat) + 1e-7)
    
    return float(np.clip(np.mean(edge_d_norm * edge_r_norm), 0.0, 2.0))


def compute_all_clinical_metrics(denoised_image, refined_image, noisy_input):
    """Wrapper computing all clinical metrics in one call."""
    denoised_image, refined_image = ensure_same_shape(denoised_image, refined_image)
    denoised_image, noisy_input = ensure_same_shape(denoised_image, noisy_input)
    refined_image, noisy_input = ensure_same_shape(refined_image, noisy_input)
    
    denoised_std = standardize_image(denoised_image)
    refined_std = standardize_image(refined_image)
    noisy_std = standardize_image(noisy_input)
    
    metrics = {}
    
    metrics['cnr_base'] = compute_cnr(denoised_std, noisy_input=noisy_std)
    metrics['cnr_refined'] = compute_cnr(refined_std, noisy_input=noisy_std)
    metrics['cnr_delta'] = metrics['cnr_refined'] - metrics['cnr_base']
    
    fwhm_base, n_base = compute_fwhm_validated(denoised_std, noisy_std)
    fwhm_ref, n_ref = compute_fwhm_validated(refined_std, noisy_std)
    metrics['fwhm_base'] = fwhm_base
    metrics['fwhm_refined'] = fwhm_ref
    metrics['fwhm_delta'] = fwhm_ref - fwhm_base if fwhm_ref < float('inf') and fwhm_base < float('inf') else 0.0
    metrics['fwhm_valid_edges_base'] = n_base
    metrics['fwhm_valid_edges_refined'] = n_ref
    
    hom_b, ent_b, con_b, cor_b = compute_glcm_texture(denoised_std)
    hom_r, ent_r, con_r, cor_r = compute_glcm_texture(refined_std)
    metrics['glcm_homogeneity_base'] = hom_b
    metrics['glcm_homogeneity_refined'] = hom_r
    metrics['glcm_homogeneity_delta'] = hom_r - hom_b
    metrics['glcm_entropy_base'] = ent_b
    metrics['glcm_entropy_refined'] = ent_r
    metrics['glcm_entropy_delta'] = ent_r - ent_b
    metrics['glcm_contrast_base'] = con_b
    metrics['glcm_contrast_refined'] = con_r
    metrics['glcm_correlation_base'] = cor_b
    metrics['glcm_correlation_refined'] = cor_r
    
    metrics['ttpi'] = compute_ttpi(denoised_std, refined_std)
    metrics['scp'] = compute_scp(denoised_std, refined_std)
    
    return metrics


def compute_all_metrics(denoised_img, post_processed_img, noisy_img, niqe_metric=None, brisque_metric=None, piqe_metric=None):
    """Compute comprehensive metrics comparing denoised vs post-processed.
    NOTE: post_processed_img should be PRE-CLAHE for honest clinical metrics."""
    denoised_img, post_processed_img = ensure_same_shape(denoised_img, post_processed_img)
    denoised_img, noisy_img = ensure_same_shape(denoised_img, noisy_img)
    post_processed_img, noisy_img = ensure_same_shape(post_processed_img, noisy_img)
    
    metrics = {}
    
    metrics['PSNR (relative)'] = compute_psnr_relative(denoised_img, post_processed_img)
    metrics['SSIM (relative)'] = compute_ssim_relative(denoised_img, post_processed_img)
    metrics['Edge Preservation Index (EPI)'] = compute_edge_preservation_index(denoised_img, post_processed_img)
    
    metrics['NIQE (post-processed)'] = compute_niqe(post_processed_img, niqe_metric)
    metrics['BRISQUE (post-processed)'] = compute_brisque(post_processed_img, brisque_metric)
    metrics['PIQE (post-processed)'] = compute_piqe(post_processed_img, piqe_metric)
    
    metrics['LaSSIM (post-processed)'] = compute_lassim(post_processed_img)
    metrics['HaarPSIMED (post-processed)'] = compute_haarpsimed(post_processed_img)
    
    metrics['Edge Density (post-processed)'] = compute_edge_density(post_processed_img)
    metrics['Mean Gradient (post-processed)'] = compute_mean_gradient(post_processed_img)
    metrics['Sharpness (post-processed)'] = compute_sharpness(post_processed_img)
    
    metrics['NIQE (denoised)'] = compute_niqe(denoised_img, niqe_metric)
    metrics['BRISQUE (denoised)'] = compute_brisque(denoised_img, brisque_metric)
    metrics['PIQE (denoised)'] = compute_piqe(denoised_img, piqe_metric)
    metrics['Edge Density (denoised)'] = compute_edge_density(denoised_img)
    metrics['Mean Gradient (denoised)'] = compute_mean_gradient(denoised_img)
    metrics['Sharpness (denoised)'] = compute_sharpness(denoised_img)
    metrics['LaSSIM (denoised)'] = compute_lassim(denoised_img)
    metrics['HaarPSIMED (denoised)'] = compute_haarpsimed(denoised_img)
    
    clinical_metrics = compute_all_clinical_metrics(denoised_img, post_processed_img, noisy_img)
    metrics.update(clinical_metrics)
    
    return metrics


def generate_difference_map(img1, img2, amplify_factor=10):
    """Generate amplified difference map between two images."""
    img1, img2 = ensure_same_shape(img1, img2)
    img1 = standardize_image(img1)
    img2 = standardize_image(img2)
    diff = np.abs(img2 - img1)
    diff_amplified = np.clip(diff * amplify_factor, 0, 1)
    return diff_amplified


def create_enhanced_difference_map(img1, img2):
    """Create an enhanced difference map with color coding."""
    img1, img2 = ensure_same_shape(img1, img2)
    img1 = standardize_image(img1)
    img2 = standardize_image(img2)
    diff = img2 - img1
    
    diff_rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.float32)
    
    positive_mask = diff > 0
    diff_rgb[positive_mask, 0] = np.clip(diff[positive_mask] * 10, 0, 1)
    
    negative_mask = diff < 0
    diff_rgb[negative_mask, 2] = np.clip(-diff[negative_mask] * 10, 0, 1)
    
    avg = (img1 + img2) / 2
    diff_rgb[:, :, 1] = avg * 0.3
    
    return np.clip(diff_rgb, 0, 1)


# ====================== MAIN APP LAYOUT ======================
def main():
    niqe_metric = None
    brisque_metric = None
    piqe_metric = None
    
    if PYIQA_AVAILABLE:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            niqe_metric = pyiqa.create_metric('niqe', device=device)
            brisque_metric = pyiqa.create_metric('brisque', device=device)
            piqe_metric = pyiqa.create_metric('piqe', device=device)
        except Exception:
            pass
    
    st.markdown(
        """
        <div class="main-header">
            <h1>🔬 Viveka Post-Processing Module</h1>
            <p>Interactive Demonstration of the Viveka Refinement Framework for Medical Image Denoising</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.header("⚙️ Viveka Parameters")
        
        auto_adapt = st.checkbox(
            "Auto-Adapt to Input Quality",
            value=True,
            help="Automatically adjust refinement strength based on input image quality"
        )
        
        refinement_strength = st.selectbox(
            "Refinement Strength",
            options=['gentle', 'balanced', 'strong'],
            index=1,
            help="Controls the intensity of CLAHE enhancement (overridden if Auto-Adapt is enabled)"
        )
        
        guide_thresh = st.slider(
            "DCT Guide Threshold (Tg)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Threshold for DCT coefficient selection from denoised image"
        )
        
        input_thresh = st.slider(
            "DCT Input Threshold (Ti)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Threshold for DCT coefficient selection from noisy image"
        )
        
        smooth_weight = st.slider(
            "Smooth Weight (λ_s)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Weight for smoothness regularization"
        )
        
        fidelity_weight = st.slider(
            "Fidelity Weight (λ_f)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Weight for fidelity to original denoised image"
        )
        
        st.markdown("---")
        st.markdown("### Components")
        
        use_dct = st.checkbox("DCT Detail Extraction", value=True)
        use_adaptive = st.checkbox("Adaptive Gains", value=True)
        use_uncertainty = st.checkbox("Uncertainty Guidance", value=True)
        use_edge_protection = st.checkbox("Edge Protection", value=True)
        use_pathology = st.checkbox("Pathology Preservation", value=True)
        
        st.markdown("---")
        st.markdown("### Difference Map Settings")
        
        amplify_factor = st.slider(
            "Difference Amplification",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Amplification factor for difference visualization"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            **Viveka** (Sanskrit: विवेक - "discrimination, discernment")
            
            A post-processing refinement module that enhances denoised 
            medical images by:
            
            - **DCT Detail Extraction**: Recovers fine details lost during denoising
            - **Adaptive Gains**: Region-specific enhancement based on anatomy
            - **Uncertainty Guidance**: Focuses refinement on uncertain regions
            - **Edge Protection**: Preserves important structural boundaries
            - **Pathology Preservation**: Maintains diagnostically relevant features
            
            **Quality-Aware Mode** automatically adjusts refinement
            based on input image quality to prevent over-enhancement.
            """
        )
        
        if not PYIQA_AVAILABLE:
            st.warning("⚠️ pyiqa/torch not installed. NIQE, BRISQUE, PIQE will use fallback values.")
        if not PYWT_AVAILABLE:
            st.warning("⚠️ PyWavelets not installed. HaarPSIMED will use fallback value.")
    
    st.header("📤 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        noisy_file = st.file_uploader(
            "Upload Noisy Input Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            key='noisy'
        )
    
    with col2:
        denoised_file = st.file_uploader(
            "Upload Denoised Output Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            key='denoised'
        )
    
    if noisy_file is not None and denoised_file is not None:
        noisy_img = load_image_from_uploaded(noisy_file)
        denoised_img = load_image_from_uploaded(denoised_file)
        
        noisy_img, denoised_img = ensure_same_shape(noisy_img, denoised_img)
        
        with st.expander("📋 Uploaded Images Preview", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(noisy_img, caption="Noisy Input", use_container_width=True)
            with col_b:
                st.image(denoised_img, caption="Denoised Output", use_container_width=True)
        
        # ===== INPUT QUALITY ASSESSMENT =====
        quality_info = estimate_input_quality(denoised_img)
        
        st.markdown("### 📊 Input Quality Assessment")
        
        if quality_info['category'] == "Excellent":
            quality_class = "quality-good"
        elif quality_info['category'] in ["Poor", "Fair"]:
            quality_class = "quality-warning"
        else:
            quality_class = ""
        
        st.markdown(f"""
        <div class="{quality_class}">
            <b>Quality Score:</b> {quality_info['score']:.1f}/100 | 
            <b>Category:</b> {quality_info['category']} | 
            <b>Sharpness:</b> {quality_info['sharpness']:.1f} | 
            <b>Edge Density:</b> {quality_info['edge_density']:.3f}
            <br><b>Recommendation:</b> {quality_info['recommendation']}
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize Viveka Refiner
        refiner = VivekaRefiner(
            guide_thresh=guide_thresh,
            input_thresh=input_thresh,
            refinement_strength=refinement_strength,
            use_adaptive_gains=use_adaptive,
            use_uncertainty_guidance=use_uncertainty,
            use_edge_protection=use_edge_protection,
            use_dct_detail=use_dct,
            use_pathology_preservation=use_pathology,
            smooth_weight=smooth_weight,
            fidelity_weight=fidelity_weight,
            auto_quality_adapt=auto_adapt
        )
        
        with st.spinner("🔄 Applying Viveka post-processing..."):
            # Get BOTH pre-CLAHE (for metrics) and post-CLAHE (for display)
            post_processed_pre_clahe, post_processed_display, adapt_info = refiner.refine(
                denoised_img.copy(),
                noisy_img.copy(),
                model_type='X-GAN'
            )
        
        # Show adapted parameters if auto-adapt is enabled
        if auto_adapt:
            with st.expander("🔧 Adapted Parameters", expanded=False):
                st.markdown(f"""
                - **Effective Tg:** {refiner.effective_Tg:.2f} (base: {guide_thresh})
                - **Effective Ti:** {refiner.effective_Ti:.2f} (base: {input_thresh})
                - **CLAHE Clip Limit:** {refiner.clahe.getClipLimit():.2f}
                """)
        
        # Difference maps computed on PRE-CLAHE (shows true refinement changes)
        diff_map = generate_difference_map(denoised_img, post_processed_pre_clahe, amplify_factor)
        enhanced_diff_map = create_enhanced_difference_map(denoised_img, post_processed_pre_clahe)
        
        # ===== VISUALIZATION SECTION =====
        st.header("🖼️ Visual Comparison")
        st.markdown("Comparison of noisy input, denoised output, Viveka post-processed result (with CLAHE for display), and difference map (pre-CLAHE).")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(noisy_img, caption="📥 Noisy Input", use_container_width=True)
        
        with col2:
            st.image(denoised_img, caption="🔧 Denoised Output", use_container_width=True)
        
        with col3:
            st.image(post_processed_display, caption="✨ Viveka Post-Processed\n(with CLAHE display)", use_container_width=True)
        
        with col4:
            st.image(diff_map, caption="📊 Difference Map, Amplified)", use_container_width=True, clamp=True)
        
        st.markdown("### 🎨 Enhanced Color-Coded Difference Map ")
        st.markdown("Red indicates enhancement (brighter), Blue indicates suppression (darker). Shows true refinement changes.")
        st.image(enhanced_diff_map, caption="Color-Coded Difference Map ", use_container_width=True, clamp=True)
        
        # ===== METRICS SECTION (computed on pre-CLAHE) =====
        st.header("📈 Comprehensive Evaluation Metrics")
        
        with st.spinner("📊 Computing comprehensive metrics..."):
            all_metrics = compute_all_metrics(denoised_img, post_processed_pre_clahe, noisy_img, 
                                              niqe_metric, brisque_metric, piqe_metric)
        
        # ===== METRICS DISPLAY =====
        st.subheader("🎯 Perceptual Quality Metrics")
        perceptual_data = {
            'Metric': ['NIQE (lower is better)', 'BRISQUE (lower is better)', 'PIQE (lower is better)'],
            'Denoised Output': [
                f"{all_metrics['NIQE (denoised)']:.4f}",
                f"{all_metrics['BRISQUE (denoised)']:.4f}",
                f"{all_metrics['PIQE (denoised)']:.4f}"
            ],
            'Viveka Post-Processed': [
                f"{all_metrics['NIQE (post-processed)']:.4f}",
                f"{all_metrics['BRISQUE (post-processed)']:.4f}",
                f"{all_metrics['PIQE (post-processed)']:.4f}"
            ],
        }
        
        perceptual_improvements = []
        for metric in ['NIQE', 'BRISQUE', 'PIQE']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val > 0 and denoised_val < 100:
                improvement = ((denoised_val - post_val) / denoised_val) * 100
                perceptual_improvements.append(f"{improvement:+.2f}%")
            else:
                perceptual_improvements.append("N/A")
        
        perceptual_data['Improvement'] = perceptual_improvements
        st.dataframe(pd.DataFrame(perceptual_data), use_container_width=True, hide_index=True)
        
        st.subheader("🏗️ Structure Preservation Metrics")
        structure_data = {
            'Metric': ['LaSSIM (higher is better)', 'HaarPSIMED (higher is better)'],
            'Denoised Output': [
                f"{all_metrics['LaSSIM (denoised)']:.4f}",
                f"{all_metrics['HaarPSIMED (denoised)']:.4f}"
            ],
            'Viveka Post-Processed': [
                f"{all_metrics['LaSSIM (post-processed)']:.4f}",
                f"{all_metrics['HaarPSIMED (post-processed)']:.4f}"
            ],
        }
        
        structure_improvements = []
        for metric in ['LaSSIM', 'HaarPSIMED']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val > 0:
                improvement = ((post_val - denoised_val) / denoised_val) * 100
                structure_improvements.append(f"{improvement:+.2f}%")
            else:
                structure_improvements.append("N/A")
        
        structure_data['Improvement'] = structure_improvements
        st.dataframe(pd.DataFrame(structure_data), use_container_width=True, hide_index=True)
        
        st.subheader("🔲 Edge-Based Metrics")
        edge_data = {
            'Metric': ['Edge Density', 'Mean Gradient', 'Sharpness (95th percentile)'],
            'Denoised Output': [
                f"{all_metrics['Edge Density (denoised)']:.4f}",
                f"{all_metrics['Mean Gradient (denoised)']:.4f}",
                f"{all_metrics['Sharpness (denoised)']:.4f}"
            ],
            'Viveka Post-Processed': [
                f"{all_metrics['Edge Density (post-processed)']:.4f}",
                f"{all_metrics['Mean Gradient (post-processed)']:.4f}",
                f"{all_metrics['Sharpness (post-processed)']:.4f}"
            ],
        }
        
        edge_improvements = []
        for metric in ['Edge Density', 'Mean Gradient', 'Sharpness']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val > 0:
                improvement = ((post_val - denoised_val) / denoised_val) * 100
                edge_improvements.append(f"{improvement:+.2f}%")
            else:
                edge_improvements.append("N/A")
        
        edge_data['Improvement'] = edge_improvements
        st.dataframe(pd.DataFrame(edge_data), use_container_width=True, hide_index=True)
        
        st.subheader("📊 Relative Comparison Metrics")
        relative_data = {
            'Metric': ['PSNR (relative)', 'SSIM (relative)', 'Edge Preservation Index (EPI)'],
            'Value': [
                f"{all_metrics['PSNR (relative)']:.4f}",
                f"{all_metrics['SSIM (relative)']:.4f}",
                f"{all_metrics['Edge Preservation Index (EPI)']:.4f}"
            ],
            'Interpretation': [
                'Higher = More change from denoised',
                'Higher = More structural similarity',
                'Higher = Better edge preservation'
            ]
        }
        st.dataframe(pd.DataFrame(relative_data), use_container_width=True, hide_index=True)
        
        st.subheader("🏥 Clinical Quality Metrics (Reference-Free")
        
        st.markdown("#### Contrast-to-Noise Ratio (CNR)")
        cnr_delta = all_metrics['cnr_delta']
        cnr_data = {
            'Metric': ['CNR'],
            'Denoised Output': [f"{all_metrics['cnr_base']:.2f}"],
            'Viveka Post-Processed': [f"{all_metrics['cnr_refined']:.2f}"],
            'Δ': [f"{cnr_delta:+.2f}"],
            'Interpretation': ['Higher = Better tissue differentiation']
        }
        st.dataframe(pd.DataFrame(cnr_data), use_container_width=True, hide_index=True)
        if cnr_delta < 0:
            st.warning("⚠️ CNR decreased! Input may already be high quality. Consider using 'gentle' refinement or enabling Auto-Adapt.")
        
        fwhm_val = all_metrics['fwhm_refined']
        fwhm_str = f"{fwhm_val:.1f}" if fwhm_val < float('inf') else 'N/A'
        fwhm_base_val = all_metrics['fwhm_base']
        fwhm_base_str = f"{fwhm_base_val:.1f}" if fwhm_base_val < float('inf') else 'N/A'
        
        st.markdown("#### Edge Sharpness (FWHM)")
        fwhm_data = {
            'Metric': ['FWHM (pixels)'],
            'Denoised Output': [fwhm_base_str],
            'Viveka Post-Processed': [fwhm_str],
            'Δ': [f"{all_metrics['fwhm_delta']:+.1f}" if all_metrics['fwhm_delta'] != 0 else "N/A"],
            'Interpretation': ['Lower = Sharper edges (better spatial resolution)']
        }
        st.dataframe(pd.DataFrame(fwhm_data), use_container_width=True, hide_index=True)
        
        st.markdown("#### GLCM Texture Analysis")
        glcm_data = {
            'Metric': ['Homogeneity', 'Entropy', 'Contrast', 'Correlation'],
            'Denoised Output': [
                f"{all_metrics['glcm_homogeneity_base']:.3f}",
                f"{all_metrics['glcm_entropy_base']:.3f}",
                f"{all_metrics['glcm_contrast_base']:.3f}",
                f"{all_metrics['glcm_correlation_base']:.3f}"
            ],
            'Viveka Post-Processed': [
                f"{all_metrics['glcm_homogeneity_refined']:.3f}",
                f"{all_metrics['glcm_entropy_refined']:.3f}",
                f"{all_metrics['glcm_contrast_refined']:.3f}",
                f"{all_metrics['glcm_correlation_refined']:.3f}"
            ],
            'Δ': [
                f"{all_metrics['glcm_homogeneity_delta']:+.3f}",
                f"{all_metrics['glcm_entropy_delta']:+.3f}",
                f"{all_metrics['glcm_contrast_refined'] - all_metrics['glcm_contrast_base']:+.3f}",
                f"{all_metrics['glcm_correlation_refined'] - all_metrics['glcm_correlation_base']:+.3f}"
            ]
        }
        st.dataframe(pd.DataFrame(glcm_data), use_container_width=True, hide_index=True)
        
        glcm_contrast_delta = all_metrics['glcm_contrast_refined'] - all_metrics['glcm_contrast_base']
        if glcm_contrast_delta > all_metrics['glcm_contrast_base'] * 1.5:
            st.warning("⚠️ GLCM Contrast increased >150%! This may indicate noise amplification. Check the difference map for uniform noise patterns.")
        
        st.markdown("#### Tissue Preservation Metrics")
        ttpi_val = all_metrics['ttpi']
        ttpi_display = f"{ttpi_val:.1f}" if ttpi_val < 100 else "999 (restored)"
        scp_val = all_metrics['scp']
        
        preservation_data = {
            'Metric': ['TTPI (Tissue Texture Preservation Index)', 'SCP (Structural Content Preservation)'],
            'Value': [ttpi_display, f"{scp_val:.3f}"],
            'Interpretation': [
                '≈1.0 = Ideal preservation | 999 = Texture restored from zero',
                '≈1.0 = Perfect structural preservation | >0.75 = Acceptable'
            ]
        }
        st.dataframe(pd.DataFrame(preservation_data), use_container_width=True, hide_index=True)
        
        st.markdown("### 🔍 Key Insights")
        
        diff_stats = {
            'Mean Absolute Difference': float(np.mean(diff_map)),
            'Max Difference': float(np.max(diff_map)),
            'Std Dev of Difference': float(np.std(diff_map)),
            'Pixels with Significant Change': float(np.sum(diff_map > 0.05) / diff_map.size * 100)
        }
        
        col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
        
        with col_insight1:
            st.metric(label="Mean Difference", value=f"{diff_stats['Mean Absolute Difference']:.4f}")
        
        with col_insight2:
            st.metric(label="Max Difference", value=f"{diff_stats['Max Difference']:.4f}")
        
        with col_insight3:
            st.metric(label="Difference Std Dev", value=f"{diff_stats['Std Dev of Difference']:.4f}")
        
        with col_insight4:
            st.metric(label="Significant Changes", value=f"{diff_stats['Pixels with Significant Change']:.1f}%")
        
        if cnr_delta > 0 and glcm_contrast_delta < all_metrics['glcm_contrast_base'] * 0.5:
            st.success("✅ **Viveka successfully enhanced this image.** CNR improved while maintaining natural texture.")
        elif cnr_delta < 0:
            st.error("⚠️ **Viveka may have degraded this image.** CNR decreased. The input was likely already high quality.")
        else:
            st.info("ℹ️ **Mixed results.** Review the metrics and visual comparison carefully.")
        
        with st.expander("📊 Detailed Analysis Visualizations", expanded=False):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            axes[0, 0].imshow(noisy_img, cmap='gray')
            axes[0, 0].set_title('Noisy Input', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(denoised_img, cmap='gray')
            axes[0, 1].set_title('Denoised Output', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(post_processed_display, cmap='gray')
            axes[0, 2].set_title('Viveka Post-Processed (with CLAHE)', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            im = axes[1, 0].imshow(diff_map, cmap='RdBu', vmin=0, vmax=1)
            axes[1, 0].set_title('Difference Map (Amplified)', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            axes[1, 1].hist(noisy_img.flatten(), bins=50, alpha=0.5, label='Noisy', color='gray')
            axes[1, 1].hist(denoised_img.flatten(), bins=50, alpha=0.5, label='Denoised', color='blue')
            axes[1, 1].hist(post_processed_display.flatten(), bins=50, alpha=0.5, label='Viveka (CLAHE)', color='red')
            axes[1, 1].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].set_xlabel('Intensity')
            axes[1, 1].set_ylabel('Frequency')
            
            def get_edges(img):
                img_uint8 = to_uint8(img)
                sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
                return np.sqrt(sobel_x**2 + sobel_y**2)
            
            edges_denoised = get_edges(denoised_img)
            edges_viveka = get_edges(post_processed_pre_clahe)
            edge_diff = np.abs(edges_viveka - edges_denoised)
            
            im2 = axes[1, 2].imshow(edge_diff, cmap='hot')
            axes[1, 2].set_title('Edge Difference', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.suptitle('Comprehensive Viveka Analysis', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # ===== DOWNLOAD SECTION =====
        st.header("💾 Download Results")
        
        # Download post-CLAHE (display) image
        post_processed_display_uint8 = (np.clip(post_processed_display, 0, 1) * 255).astype(np.uint8)
        post_processed_display_pil = Image.fromarray(post_processed_display_uint8)
        buf_display = io.BytesIO()
        post_processed_display_pil.save(buf_display, format='PNG')
        buf_display.seek(0)
        
        st.download_button(
            label="📥 Download Post-Processed Image (with CLAHE - for display)",
            data=buf_display,
            file_name="viveka_post_processed_display.png",
            mime="image/png"
        )
        
        # Download pre-CLAHE image (for metrics)
        post_processed_pre_clahe_uint8 = (np.clip(post_processed_pre_clahe, 0, 1) * 255).astype(np.uint8)
        post_processed_pre_clahe_pil = Image.fromarray(post_processed_pre_clahe_uint8)
        buf_pre = io.BytesIO()
        post_processed_pre_clahe_pil.save(buf_pre, format='PNG')
        buf_pre.seek(0)
        
        st.download_button(
            label="📥 Download Post-Processed Image (Pre-CLAHE - for metrics)",
            data=buf_pre,
            file_name="viveka_post_processed_pre_clahe.png",
            mime="image/png"
        )
        
        diff_uint8 = (np.clip(diff_map, 0, 1) * 255).astype(np.uint8)
        diff_pil = Image.fromarray(diff_uint8)
        diff_buf = io.BytesIO()
        diff_pil.save(diff_buf, format='PNG')
        diff_buf.seek(0)
        
        st.download_button(
            label="📥 Download Difference Map",
            data=diff_buf,
            file_name="viveka_difference_map.png",
            mime="image/png"
        )
        
        metrics_rows = []
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                metrics_rows.append({'Metric': key, 'Value': f"{value:.4f}"})
        df_metrics = pd.DataFrame(metrics_rows)
        csv = df_metrics.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Metrics Report",
            data=csv,
            file_name="viveka_comprehensive_metrics.csv",
            mime="text/csv"
        )
        
        comparison_data = {
            'Category': ['Perceptual', 'Perceptual', 'Perceptual', 'Structure', 'Structure', 'Edge', 'Edge', 'Edge'],
            'Metric': ['NIQE', 'BRISQUE', 'PIQE', 'LaSSIM', 'HaarPSIMED', 'Edge Density', 'Mean Gradient', 'Sharpness'],
            'Denoised': [
                f"{all_metrics['NIQE (denoised)']:.4f}",
                f"{all_metrics['BRISQUE (denoised)']:.4f}",
                f"{all_metrics['PIQE (denoised)']:.4f}",
                f"{all_metrics['LaSSIM (denoised)']:.4f}",
                f"{all_metrics['HaarPSIMED (denoised)']:.4f}",
                f"{all_metrics['Edge Density (denoised)']:.4f}",
                f"{all_metrics['Mean Gradient (denoised)']:.4f}",
                f"{all_metrics['Sharpness (denoised)']:.4f}"
            ],
            'Viveka_Post_Processed': [
                f"{all_metrics['NIQE (post-processed)']:.4f}",
                f"{all_metrics['BRISQUE (post-processed)']:.4f}",
                f"{all_metrics['PIQE (post-processed)']:.4f}",
                f"{all_metrics['LaSSIM (post-processed)']:.4f}",
                f"{all_metrics['HaarPSIMED (post-processed)']:.4f}",
                f"{all_metrics['Edge Density (post-processed)']:.4f}",
                f"{all_metrics['Mean Gradient (post-processed)']:.4f}",
                f"{all_metrics['Sharpness (post-processed)']:.4f}"
            ],
        }
        df_comparison = pd.DataFrame(comparison_data)
        csv_comparison = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table",
            data=csv_comparison,
            file_name="viveka_comparison_table.csv",
            mime="text/csv"
        )
    
    elif noisy_file is not None or denoised_file is not None:
        st.warning("⚠️ Please upload **both** noisy input and denoised output images to proceed.")
    
    else:
        st.info("👆 Upload both images above to see the Viveka post-processing in action!")
        
        st.markdown("### How it works:")
        st.markdown(
            """
            1. **Upload Noisy Image**: The original noisy medical image
            2. **Upload Denoised Image**: Output from your denoising model (e.g., X-GAN, RIDNet)
            3. **Quality Assessment**: Automatic input quality evaluation
            4. **Adjust Parameters**: Fine-tune Viveka's refinement behavior
            5. **View Results**: See side-by-side comparison and comprehensive metrics
            
            The Viveka module enhances the denoised output by:
            - Recovering fine details through DCT-based processing
            - Applying region-specific gains (bone, lung, background)
            - Using uncertainty maps to focus refinement
            - Protecting edges and preserving pathology
            
            **Key Design Principle:**
            - **Metrics computed on pre-CLAHE output** for honest clinical evaluation
            - **CLAHE applied only for visual display** (what clinicians see)
            - This separation ensures metrics reflect true refinement quality
            
            **Quality-Aware Mode:**
            Automatically adapts refinement strength based on input image quality:
            - **Excellent inputs** → Gentle refinement (do no harm)
            - **Fair inputs** → Balanced refinement
            - **Poor inputs** → Strong refinement (recover lost details)
            
            **Metrics Computed:**
            - **Perceptual Quality**: NIQE, BRISQUE, PIQE (lower is better)
            - **Structure Preservation**: LaSSIM, HaarPSIMED (higher is better)
            - **Edge Metrics**: Edge Density, Mean Gradient, Sharpness
            - **Relative Metrics**: PSNR, SSIM, Edge Preservation Index (EPI)
            - **Clinical Quality Metrics** (Reference-Free): CNR, FWHM, GLCM Texture, TTPI, SCP
            """
        )
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><b>Viveka Post-Processing Demo</b> | Amrita School of AI</p>
            <p style='font-size: 0.9em;'>Ideated by: Siju K S | Supervised by: Dr. Vipin V</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
