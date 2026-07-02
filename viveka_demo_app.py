import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Check for optional dependencies for advanced metrics
PYIQA_AVAILABLE = False
PYWT_AVAILABLE = False

try:
    import pywt  # PyWavelets for HaarPSIMED
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# ====================== PAGE CONFIGURATION ======================
st.set_page_config(
    page_title="PAR Post-Processing Demo",
    page_icon="amrita_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF9933 0%, #FF8800 100%);
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
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    }
    .logo-container img {
        height: 60px;
        width: auto;
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
    .diff-map-container {
        position: relative;
        display: inline-block;
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
    """Ensure two images have the same shape by cropping to minimum dimensions."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        min_h = min(h1, h2)
        min_w = min(w1, w2)
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    return img1, img2

def standardize_image(image):
    """Standardize image to 0-1 range."""
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0, 1)

def to_uint8(image):
    """Convert to uint8."""
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

def to_tensor_rgb(image):
    """Convert to torch tensor (RGB)."""
    import torch
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def load_image_from_uploaded(uploaded_file):
    """Load image from uploaded file and convert to grayscale numpy array."""
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array

def resize_to_match(reference, target):
    """Resize target image to match reference shape."""
    if reference.shape != target.shape:
        target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
    return target


# ====================== PAR REFINER (Same as original implementation) ======================
class VivekaRefiner:
    def __init__(self, guide_thresh=1.5, input_thresh=3.0, spins=2,
                 use_adaptive_gains=True, use_pathology_preservation=True,
                 use_uncertainty_guidance=True,
                 use_edge_protection=True,
                 use_dct_detail=True,
                 refinement_strength='balanced',
                 smooth_weight=0.1, fidelity_weight=1.0, clinical_weight=0.05):
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

        if refinement_strength == 'gentle':
            self.clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8,8))
        elif refinement_strength == 'strong':
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        else:
            self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))

        self.default_bone_gain = 1.0
        self.default_lung_gain = 0.3
        self.default_bg_gain = 0.0

    def anscombe_forward(self, img):
        return 2.0 * np.sqrt(img + (3.0/8.0))

    def anscombe_inverse(self, img):
        return np.maximum(0, (img / 2.0)**2 - (3.0/8.0))

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
        global_var = np.var(gan_output)
        if global_var > 0:
            uncertainty = np.clip(local_var / global_var, 0, 1)
        else:
            uncertainty = np.zeros_like(gan_output)
        diff_from_noisy = np.abs(gan_output - noisy_input)
        uncertainty = 0.5 * uncertainty + 0.5 * np.clip(diff_from_noisy * 1.0, 0, 1)
        return uncertainty

    def pathology_preservation_loss(self, refined, original):
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
        output = np.zeros_like(gan_vst)
        weights = np.zeros_like(gan_vst)
        P = 8
        stride = 4

        h_adj = (h // P) * P
        w_adj = (w // P) * P

        for y in range(0, h_adj - P + 1, stride):
            for x in range(0, w_adj - P + 1, stride):
                patch_g = gan_vst[y:y+P, x:x+P]
                patch_n = noisy_vst[y:y+P, x:x+P]
                dct_g = cv2.dct(patch_g)
                dct_n = cv2.dct(patch_n)
                mask = np.logical_or(np.abs(dct_g) > self.Tg, np.abs(dct_n) > self.Ti)
                dct_filtered = dct_n * mask.astype(np.float32)
                output[y:y+P, x:x+P] += cv2.idct(dct_filtered)
                weights[y:y+P, x:x+P] += 1.0

        if h_adj < h or w_adj < w:
            output[h_adj:, :] = gan_vst[h_adj:, :]
            output[:, w_adj:] = gan_vst[:, w_adj:]
            weights[h_adj:, :] = 1.0
            weights[:, w_adj:] = 1.0

        result = output / (weights + 1e-5)

        if result.shape != (h, w):
            result = cv2.resize(result, (w, h))

        return result

    def refine(self, gan_prediction, original_noisy_input, model_type='X-GAN',
               baseline_quality=None, baseline_piqe=None):
        manas_clean = np.clip(np.squeeze(gan_prediction), 0, 1)

        manas_clean, original_noisy_input = ensure_same_shape(manas_clean, original_noisy_input)

        gan_scaled = manas_clean * 255.0
        noisy_scaled = np.clip(original_noisy_input, 0, 1) * 255.0

        gan_scaled, noisy_scaled = ensure_same_shape(gan_scaled, noisy_scaled)

        gan_vst = self.anscombe_forward(gan_scaled)
        noisy_vst = self.anscombe_forward(noisy_scaled)

        shifts = [(0,0), (4,4)]
        acc = np.zeros_like(gan_vst)
        for dy, dx in shifts:
            res = self.run_oracle_dct(np.roll(gan_vst, (dy,dx), (0,1)),
                                       np.roll(noisy_vst, (dy,dx), (0,1)))
            acc += np.roll(res, (-dy,-dx), (0,1))
        viveka_sharp = self.anscombe_inverse(acc / len(shifts)) / 255.0
        viveka_sharp = np.clip(viveka_sharp, 0, 1)

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
            uncertainty = self.compute_uncertainty_map(manas_clean, original_noisy_input)
            uncertainty_weight = 1.0 + uncertainty
        else:
            uncertainty_weight = np.ones_like(manas_clean)

        if self.use_edge_protection:
            sobel_x = cv2.Sobel((manas_clean * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel((manas_clean * 255).astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
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

        final_uint8 = (np.clip(final_image, 0, 1) * 255).astype(np.uint8)
        final_pop = self.clahe.apply(final_uint8)
        return final_pop.astype(np.float32) / 255.0


# ====================== EVALUATION METRICS ======================

# --- Perceptual Quality Metrics ---
def compute_niqe(image, niqe_metric=None):
    """Compute NIQE (Natural Image Quality Evaluator) - lower is better."""
    image_std = standardize_image(image)
    if niqe_metric is not None:
        try:
            import torch
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = niqe_metric(tensor)
            return score.item()
        except Exception as e:
            pass
    return 50.0  # Default fallback

def compute_brisque(image, brisque_metric=None):
    """Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) - lower is better."""
    image_std = standardize_image(image)
    if brisque_metric is not None:
        try:
            import torch
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = brisque_metric(tensor)
            return score.item()
        except Exception as e:
            pass
    return 50.0  # Default fallback

def compute_piqe(image, piqe_metric=None):
    """Compute PIQE (Perception based Image Quality Evaluator) - lower is better."""
    image_std = standardize_image(image)
    if piqe_metric is not None:
        try:
            import torch
            tensor = to_tensor_rgb(image_std)
            with torch.no_grad():
                score = piqe_metric(tensor)
            return score.item()
        except Exception as e:
            pass
    return 50.0  # Default fallback

# --- Structure Preservation Metric: LaSSIM ---
def compute_lassim(image, levels=4):
    """Compute Laplacian Pyramid SSIM - higher is better."""
    image_std = standardize_image(image)
    
    pyramid = []
    current = image_std.copy()
    
    for level in range(levels):
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        if current.shape[0] // 2 > 0 and current.shape[1] // 2 > 0:
            downsampled = cv2.resize(blurred, (current.shape[1] // 2, current.shape[0] // 2))
            upsampled = cv2.resize(downsampled, (current.shape[1], current.shape[0]))
        else:
            upsampled = blurred
        laplacian = current - upsampled
        pyramid.append(laplacian)
        current = downsampled if current.shape[0] // 2 > 0 and current.shape[1] // 2 > 0 else current
    pyramid.append(current)
    
    lassim_score = 0
    total_weight = 0
    
    for i in range(len(pyramid) - 1):
        img1 = pyramid[i]
        img2 = cv2.resize(pyramid[i+1], (img1.shape[1], img1.shape[0]))
        
        try:
            ssim_val = ssim(img1, img2, data_range=1.0)
        except:
            ssim_val = 0.5
        
        weight = 0.5 ** i
        lassim_score += ssim_val * weight
        total_weight += weight
    
    if total_weight > 0:
        lassim_score = lassim_score / total_weight
    else:
        lassim_score = 0.5
    
    return np.clip(lassim_score, 0, 1)

# --- X-ray Optimized Metric: HaarPSIMED ---
def compute_haarpsimed(image):
    """Compute HaarPSI-based Medical Image Quality Metric - higher is better."""
    image_std = standardize_image(image)
    
    if not PYWT_AVAILABLE:
        return 0.5  # Fallback
    
    try:
        coeffs = pywt.wavedec2(image_std, 'haar', level=3)
        
        scores = []
        weights = [0.5, 0.3, 0.2]
        
        for level in range(1, 4):
            if level < len(coeffs):
                cA, (cH, cV, cD) = coeffs[0], coeffs[level]
                
                energy_H = np.sum(cH ** 2)
                energy_V = np.sum(cV ** 2)
                energy_D = np.sum(cD ** 2)
                
                structural_score = (energy_H + energy_V + energy_D) / (np.sum(cA ** 2) + 1e-7)
                scores.append(structural_score * weights[level-1])
        
        haarpsi_score = np.sum(scores)
        alpha = 5.8
        C = 5
        haarpesimed = 1.0 / (1.0 + alpha * np.exp(-C * haarpsi_score))
        
        return np.clip(haarpesimed, 0, 1)
    except:
        return 0.5  # Fallback

# --- Edge-Based Metrics ---
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

# --- Edge Preservation Index ---
def compute_edge_preservation_index(ref_image, test_image):
    """EPI - correlation of edge magnitudes between reference and test."""
    ref_uint8 = to_uint8(ref_image)
    test_uint8 = to_uint8(test_image)
    
    sobel_ref = cv2.Sobel(ref_uint8, cv2.CV_64F, 1, 1, ksize=3)
    sobel_test = cv2.Sobel(test_uint8, cv2.CV_64F, 1, 1, ksize=3)
    
    flat_ref = np.abs(sobel_ref).flatten()
    flat_test = np.abs(sobel_test).flatten()
    
    if np.var(flat_ref) == 0 or np.var(flat_test) == 0:
        return 0.0
    corr = np.corrcoef(flat_ref, flat_test)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

# --- Relative PSNR and SSIM ---
def compute_psnr_relative(clean, denoised):
    """PSNR for relative comparison."""
    mse = np.mean((clean - denoised) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))

def compute_ssim_relative(clean, denoised):
    """SSIM for relative comparison."""
    clean = np.clip(clean, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    return float(ssim(clean, denoised, data_range=1.0))


def compute_all_metrics(denoised_img, post_processed_img, noisy_img, niqe_metric=None, brisque_metric=None, piqe_metric=None):
    """Compute comprehensive metrics comparing denoised vs post-processed."""
    # Ensure same shape
    denoised_img, post_processed_img = ensure_same_shape(denoised_img, post_processed_img)
    denoised_img, noisy_img = ensure_same_shape(denoised_img, noisy_img)
    
    metrics = {}
    
    # Relative metrics (denoised vs post-processed)
    metrics['PSNR (relative)'] = compute_psnr_relative(denoised_img, post_processed_img)
    metrics['SSIM (relative)'] = compute_ssim_relative(denoised_img, post_processed_img)
    metrics['Edge Preservation Index (EPI)'] = compute_edge_preservation_index(denoised_img, post_processed_img)
    
    # Perceptual quality metrics (on post-processed image)
    metrics['NIQE (post-processed)'] = compute_niqe(post_processed_img, niqe_metric)
    metrics['BRISQUE (post-processed)'] = compute_brisque(post_processed_img, brisque_metric)
    metrics['PIQE (post-processed)'] = compute_piqe(post_processed_img, piqe_metric)
    
    # Structure preservation
    metrics['LaSSIM (post-processed)'] = compute_lassim(post_processed_img)
    metrics['HaarPSIMED (post-processed)'] = compute_haarpsimed(post_processed_img)
    
    # Edge-based metrics
    metrics['Edge Density (post-processed)'] = compute_edge_density(post_processed_img)
    metrics['Mean Gradient (post-processed)'] = compute_mean_gradient(post_processed_img)
    metrics['Sharpness (post-processed)'] = compute_sharpness(post_processed_img)
    
    # Also compute for denoised for comparison
    metrics['NIQE (denoised)'] = compute_niqe(denoised_img, niqe_metric)
    metrics['BRISQUE (denoised)'] = compute_brisque(denoised_img, brisque_metric)
    metrics['PIQE (denoised)'] = compute_piqe(denoised_img, piqe_metric)
    metrics['Edge Density (denoised)'] = compute_edge_density(denoised_img)
    metrics['Mean Gradient (denoised)'] = compute_mean_gradient(denoised_img)
    metrics['Sharpness (denoised)'] = compute_sharpness(denoised_img)
    metrics['LaSSIM (denoised)'] = compute_lassim(denoised_img)
    metrics['HaarPSIMED (denoised)'] = compute_haarpsimed(denoised_img)
    
    return metrics


def generate_difference_map(denoised, post_processed, amplify_factor=10):
    """Generate amplified difference map between denoised and post-processed images."""
    denoised, post_processed = ensure_same_shape(denoised, post_processed)
    diff = np.abs(post_processed - denoised)
    # Amplify for visibility
    diff_amplified = np.clip(diff * amplify_factor, 0, 1)
    return diff_amplified


def create_enhanced_difference_map(denoised, post_processed):
    """Create an enhanced difference map with color coding."""
    denoised, post_processed = ensure_same_shape(denoised, post_processed)
    diff = post_processed - denoised
    
    # Create RGB difference map
    # Red = post-processed is brighter (enhancement)
    # Blue = post-processed is darker (suppression)
    # Green = no change
    diff_rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.float32)
    
    # Positive differences (enhancement) -> Red
    positive_mask = diff > 0
    diff_rgb[positive_mask, 0] = np.clip(diff[positive_mask] * 10, 0, 1)  # Red channel
    
    # Negative differences (suppression) -> Blue
    negative_mask = diff < 0
    diff_rgb[negative_mask, 2] = np.clip(-diff[negative_mask] * 10, 0, 1)  # Blue channel
    
    # Base intensity from average
    avg = (denoised + post_processed) / 2
    diff_rgb[:, :, 1] = avg * 0.3  # Subtle green for context
    
    return np.clip(diff_rgb, 0, 1)


# ====================== MAIN APP LAYOUT ======================
def main():
    # Initialize pyiqa metrics if available (lazy loading to avoid torch issues)
    niqe_metric = None
    brisque_metric = None
    piqe_metric = None
    
    try:
        import torch
        import pyiqa
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            niqe_metric = pyiqa.create_metric('niqe', device=device)
            brisque_metric = pyiqa.create_metric('brisque', device=device)
            piqe_metric = pyiqa.create_metric('piqe', device=device)
        except Exception as e:
            st.warning(f"Could not initialize pyiqa metrics: {e}")
    except ImportError:
        pass  # pyiqa not available, will use fallback values
    
    # Header with Amrita logo
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("amrita_logo.svg", width=150)
    
    st.markdown(
        """
        <div class="main-header">
            <h1>🔬 PAR Post-Processing Module</h1>
            <p>Interactive Demonstration of the Physics Aware Refinement (PAR) Framework for Medical Image Denoising</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ PAR Parameters")
        
        refinement_strength = st.selectbox(
            "Refinement Strength",
            options=['gentle', 'balanced', 'strong'],
            index=1,
            key="refinement_strength",
            help="Controls the intensity of CLAHE enhancement"
        )
        
        guide_thresh = st.slider(
            "DCT Guide Threshold (Tg)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="guide_thresh",
            help="Threshold for DCT coefficient selection from denoised image"
        )
        
        input_thresh = st.slider(
            "DCT Input Threshold (Ti)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key="input_thresh",
            help="Threshold for DCT coefficient selection from noisy image"
        )
        
        smooth_weight = st.slider(
            "Smooth Weight (λ_s)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            key="smooth_weight",
            help="Weight for smoothness regularization"
        )
        
        fidelity_weight = st.slider(
            "Fidelity Weight (λ_f)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="fidelity_weight",
            help="Weight for fidelity to original denoised image"
        )
        
        st.markdown("---")
        st.markdown("### Components")
        
        use_dct = st.checkbox("DCT Detail Extraction", value=True, key="use_dct")
        use_adaptive = st.checkbox("Adaptive Gains", value=True, key="use_adaptive")
        use_uncertainty = st.checkbox("Uncertainty Guidance", value=True, key="use_uncertainty")
        use_edge_protection = st.checkbox("Edge Protection", value=True, key="use_edge_protection")
        use_pathology = st.checkbox("Pathology Preservation", value=True, key="use_pathology")
        
        st.markdown("---")
        
        # Update/Apply button
        update_clicked = st.button("✅ Update & Apply Changes", use_container_width=True, type="primary")
        if update_clicked:
            # Store current values in session state and trigger rerun
            st.session_state.params_updated = True
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Difference Map Settings")
        
        amplify_factor = st.slider(
            "Difference Amplification",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="amplify_factor",
            help="Amplification factor for difference visualization"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            **PAR** (Physics Aware Refinement) is a post-processing refinement module designed to enhance denoised medical images. It leverages advanced techniques to recover fine details, adaptively enhance regions based on anatomical structures, and preserve diagnostically relevant features while minimizing artifacts.
            
            A post-processing refinement module that enhances denoised 
            medical images by:
            
            - **DCT Detail Extraction**: Recovers fine details lost during denoising
            - **Adaptive Gains**: Region-specific enhancement based on anatomy
            - **Uncertainty Guidance**: Focuses refinement on uncertain regions
            - **Edge Protection**: Preserves important structural boundaries
            - **Pathology Preservation**: Maintains diagnostically relevant features
            """
        )
        
        if niqe_metric is None:
            st.warning("⚠️ pyiqa not installed. Some perceptual metrics (NIQE, BRISQUE, PIQE) will use fallback values.")
        if not PYWT_AVAILABLE:
            st.warning("⚠️ PyWavelets not installed. HaarPSIMED metric will use fallback value.")
    
    # File upload section
    st.header("📤 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        noisy_file = st.file_uploader(
            "Upload Noisy Input Image",
            type=['png', 'jpg', 'jpeg'],
            key='noisy'
        )
    
    with col2:
        denoised_file = st.file_uploader(
            "Upload Denoised Output Image",
            type=['png', 'jpg', 'jpeg'],
            key='denoised'
        )
    
    # Process images if both are uploaded
    if noisy_file is not None and denoised_file is not None:
        # Load images
        noisy_img = load_image_from_uploaded(noisy_file)
        denoised_img = load_image_from_uploaded(denoised_file)
        
        # Ensure same shape
        min_h = min(noisy_img.shape[0], denoised_img.shape[0])
        min_w = min(noisy_img.shape[1], denoised_img.shape[1])
        noisy_img = noisy_img[:min_h, :min_w]
        denoised_img = denoised_img[:min_h, :min_w]
        
        # Display uploaded images preview
        with st.expander("📋 Uploaded Images Preview", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(noisy_img, caption="Noisy Input", use_container_width=True)
            with col_b:
                st.image(denoised_img, caption="Denoised Output", use_container_width=True)
        
        # Show info message about Update button
        if not st.session_state.get('params_updated', False):
            st.info("💡 Adjust parameters in the sidebar and click **✅ Update & Apply Changes** to process with new settings.")
        
        # Initialize PAR Refiner with sidebar parameters
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
            fidelity_weight=fidelity_weight
        )
        
        # Show processing status
        status_text = "🔄 Applying PAR post-processing..."
        if st.session_state.get('params_updated', False):
            status_text = "✅ Applying updated parameters..."
        
        # Apply PAR refinement
        with st.spinner(status_text):
            post_processed_img = refiner.refine(
                denoised_img.copy(),
                noisy_img.copy(),
                model_type='X-GAN'
            )
        
        # Reset the update flag after processing
        if st.session_state.get('params_updated', False):
            st.session_state.params_updated = False
        
        # Generate difference maps
        diff_map = generate_difference_map(denoised_img, post_processed_img, amplify_factor)
        enhanced_diff_map = create_enhanced_difference_map(denoised_img, post_processed_img)
        
        # ====================== VISUALIZATION SECTION ======================
        st.header("🖼️ Visual Comparison")
        st.markdown("Comparison of noisy input, denoised output, PAR post-processed result, and difference map.")
        
        # Display all four images in one row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(noisy_img, caption="📥 Noisy Input", use_container_width=True)
        
        with col2:
            st.image(denoised_img, caption="🔧 Denoised Output", use_container_width=True)
        
        with col3:
            st.image(post_processed_img, caption="✨ PAR Post-Processed", use_container_width=True)
        
        with col4:
            st.image(diff_map, caption="📊 Difference Map (Amplified)", use_container_width=True, clamp=True)
        
        # Enhanced difference map with color coding
        st.markdown("### 🎨 Enhanced Color-Coded Difference Map")
        st.markdown("Red indicates enhancement (brighter), Blue indicates suppression (darker).")
        st.image(enhanced_diff_map, caption="Color-Coded Difference Map", use_container_width=True, clamp=True)
        
        # ====================== METRICS SECTION ======================
        st.header("📈 Comprehensive Evaluation Metrics")
        st.markdown("Quantitative comparison between denoised output and PAR post-processed result.")
        
        # Compute all metrics
        with st.spinner("📊 Computing comprehensive metrics..."):
            all_metrics = compute_all_metrics(denoised_img, post_processed_img, noisy_img, 
                                              niqe_metric, brisque_metric, piqe_metric)
        
        # Create comparison dataframe for perceptual metrics
        st.subheader("🎯 Perceptual Quality Metrics")
        perceptual_data = {
            'Metric': ['NIQE (lower is better)', 'BRISQUE (lower is better)', 'PIQE (lower is better)'],
            'Denoised Output': [
                f"{all_metrics['NIQE (denoised)']:.4f}",
                f"{all_metrics['BRISQUE (denoised)']:.4f}",
                f"{all_metrics['PIQE (denoised)']:.4f}"
            ],
            'PAR Post-Processed': [
                f"{all_metrics['NIQE (post-processed)']:.4f}",
                f"{all_metrics['BRISQUE (post-processed)']:.4f}",
                f"{all_metrics['PIQE (post-processed)']:.4f}"
            ],
        }
        
        # Calculate improvements (negative improvement is good for these metrics)
        perceptual_improvements = []
        for metric in ['NIQE', 'BRISQUE', 'PIQE']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val != 0:
                # For these metrics, lower is better, so improvement = (denoised - post) / denoised
                improvement = ((denoised_val - post_val) / denoised_val) * 100
                perceptual_improvements.append(f"{improvement:+.2f}%")
            else:
                perceptual_improvements.append("N/A")
        
        perceptual_data['Improvement'] = perceptual_improvements
        
        st.dataframe(
            pd.DataFrame(perceptual_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Structure preservation metrics
        st.subheader("🏗️ Structure Preservation Metrics")
        structure_data = {
            'Metric': ['LaSSIM (higher is better)', 'HaarPSIMED (higher is better)'],
            'Denoised Output': [
                f"{all_metrics['LaSSIM (denoised)']:.4f}",
                f"{all_metrics['HaarPSIMED (denoised)']:.4f}"
            ],
            'PAR Post-Processed': [
                f"{all_metrics['LaSSIM (post-processed)']:.4f}",
                f"{all_metrics['HaarPSIMED (post-processed)']:.4f}"
            ],
        }
        
        structure_improvements = []
        for metric in ['LaSSIM', 'HaarPSIMED']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val != 0:
                improvement = ((post_val - denoised_val) / denoised_val) * 100
                structure_improvements.append(f"{improvement:+.2f}%")
            else:
                structure_improvements.append("N/A")
        
        structure_data['Improvement'] = structure_improvements
        
        st.dataframe(
            pd.DataFrame(structure_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Edge-based metrics
        st.subheader("🔲 Edge-Based Metrics")
        edge_data = {
            'Metric': ['Edge Density', 'Mean Gradient', 'Sharpness (95th percentile)'],
            'Denoised Output': [
                f"{all_metrics['Edge Density (denoised)']:.4f}",
                f"{all_metrics['Mean Gradient (denoised)']:.4f}",
                f"{all_metrics['Sharpness (denoised)']:.4f}"
            ],
            'PAR Post-Processed': [
                f"{all_metrics['Edge Density (post-processed)']:.4f}",
                f"{all_metrics['Mean Gradient (post-processed)']:.4f}",
                f"{all_metrics['Sharpness (post-processed)']:.4f}"
            ],
        }
        
        edge_improvements = []
        for metric in ['Edge Density', 'Mean Gradient', 'Sharpness']:
            denoised_val = all_metrics[f'{metric} (denoised)']
            post_val = all_metrics[f'{metric} (post-processed)']
            if denoised_val != 0:
                improvement = ((post_val - denoised_val) / denoised_val) * 100
                edge_improvements.append(f"{improvement:+.2f}%")
            else:
                edge_improvements.append("N/A")
        
        edge_data['Improvement'] = edge_improvements
        
        st.dataframe(
            pd.DataFrame(edge_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Relative metrics (denoised vs post-processed)
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
        
        st.dataframe(
            pd.DataFrame(relative_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Additional insights
        st.markdown("### 🔍 Key Insights")
        
        # Calculate some summary statistics
        diff_stats = {
            'Mean Absolute Difference': float(np.mean(diff_map)),
            'Max Difference': float(np.max(diff_map)),
            'Std Dev of Difference': float(np.std(diff_map)),
            'Pixels with Significant Change': float(np.sum(diff_map > 0.05) / diff_map.size * 100)
        }
        
        col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
        
        with col_insight1:
            st.metric(
                label="Mean Difference",
                value=f"{diff_stats['Mean Absolute Difference']:.4f}"
            )
        
        with col_insight2:
            st.metric(
                label="Max Difference",
                value=f"{diff_stats['Max Difference']:.4f}"
            )
        
        with col_insight3:
            st.metric(
                label="Difference Std Dev",
                value=f"{diff_stats['Std Dev of Difference']:.4f}"
            )
        
        with col_insight4:
            st.metric(
                label="Significant Changes",
                value=f"{diff_stats['Pixels with Significant Change']:.1f}%"
            )
        
        # ====================== ADDITIONAL VISUALIZATIONS ======================
        with st.expander("📊 Detailed Analysis Visualizations", expanded=False):
            # Create a comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Noisy image
            axes[0, 0].imshow(noisy_img, cmap='gray')
            axes[0, 0].set_title('Noisy Input', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. Denoised image
            axes[0, 1].imshow(denoised_img, cmap='gray')
            axes[0, 1].set_title('Denoised Output', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. Post-processed image
            axes[0, 2].imshow(post_processed_img, cmap='gray')
            axes[0, 2].set_title('PAR Post-Processed', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # 4. Difference map (color)
            im = axes[1, 0].imshow(diff_map, cmap='RdBu', vmin=0, vmax=1)
            axes[1, 0].set_title('Difference Map (Amplified)', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # 5. Histogram comparison
            axes[1, 1].hist(noisy_img.flatten(), bins=50, alpha=0.5, label='Noisy', color='gray')
            axes[1, 1].hist(denoised_img.flatten(), bins=50, alpha=0.5, label='Denoised', color='blue')
            axes[1, 1].hist(post_processed_img.flatten(), bins=50, alpha=0.5, label='PAR', color='red')
            axes[1, 1].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].set_xlabel('Intensity')
            axes[1, 1].set_ylabel('Frequency')
            
            # 6. Edge comparison
            def get_edges(img):
                sobel_x = cv2.Sobel(to_uint8(img), cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(to_uint8(img), cv2.CV_64F, 0, 1, ksize=3)
                return np.sqrt(sobel_x**2 + sobel_y**2)
            
            edges_denoised = get_edges(denoised_img)
            edges_viveka = get_edges(post_processed_img)
            edge_diff = np.abs(edges_viveka - edges_denoised)
            
            im2 = axes[1, 2].imshow(edge_diff, cmap='hot')
            axes[1, 2].set_title('Edge Difference', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.suptitle('Comprehensive PAR Analysis', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            st.pyplot(fig)
        
        # ====================== DOWNLOAD SECTION ======================
        st.header("💾 Download Results")
        
        # Convert post-processed image to PIL for download
        post_processed_uint8 = (post_processed_img * 255).astype(np.uint8)
        post_processed_pil = Image.fromarray(post_processed_uint8)
        
        # Save to buffer
        buf = io.BytesIO()
        post_processed_pil.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Post-Processed Image",
            data=buf,
            file_name="par_post_processed.png",
            mime="image/png"
        )
        
        # Download difference map
        diff_uint8 = (diff_map * 255).astype(np.uint8)
        diff_pil = Image.fromarray(diff_uint8)
        diff_buf = io.BytesIO()
        diff_pil.save(diff_buf, format='PNG')
        diff_buf.seek(0)
        
        st.download_button(
            label="📥 Download Difference Map",
            data=diff_buf,
            file_name="par_difference_map.png",
            mime="image/png"
        )
        
        # Download metrics as CSV
        # Create comprehensive metrics dataframe
        metrics_rows = []
        for key, value in all_metrics.items():
            metrics_rows.append({'Metric': key, 'Value': f"{value:.4f}"})
        df_metrics = pd.DataFrame(metrics_rows)
        csv = df_metrics.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Metrics Report",
            data=csv,
            file_name="par_comprehensive_metrics.csv",
            mime="text/csv"
        )
        
        # Download comparison table
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
            'PAR_Post_Processed': [
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
            file_name="par_comparison_table.csv",
            mime="text/csv"
        )
    
    elif noisy_file is not None or denoised_file is not None:
        st.warning("⚠️ Please upload **both** noisy input and denoised output images to proceed.")
    
    else:
        # Show placeholder
        st.info("👆 Upload both images above to see the PAR post-processing in action!")
        
        # Show example of what the app does
        st.markdown("### How it works:")
        st.markdown(
            """
            1. **Upload Noisy Image**: The original noisy medical image
            2. **Upload Denoised Image**: Output from your denoising model (e.g., X-GAN)
            3. **Adjust Parameters**: Fine-tune PAR's refinement behavior
            4. **View Results**: See side-by-side comparison and comprehensive metrics
            
            The PAR module enhances the denoised output by:
            - Recovering fine details through DCT-based processing
            - Applying region-specific gains (bone, lung, background)
            - Using uncertainty maps to focus refinement
            - Protecting edges and preserving pathology
            
            **Metrics Computed:**
            - **Perceptual Quality**: NIQE, BRISQUE, PIQE (lower is better)
            - **Structure Preservation**: LaSSIM, HaarPSIMED (higher is better)
            - **Edge Metrics**: Edge Density, Mean Gradient, Sharpness
            - **Relative Metrics**: PSNR, SSIM, Edge Preservation Index (EPI)
            """
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><b>PAR Post-Processing Demo</b> | Amrita School of AI</p>
            <p style='font-size: 0.9em;'>Ideated by: Siju K S | Supervised by: Dr. Vipin V</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()