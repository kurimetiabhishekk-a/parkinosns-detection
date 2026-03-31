

import numpy as np
import os
import joblib

np.set_printoptions(suppress=True)

_base = os.path.dirname(os.path.abspath(__file__))

PD_THRESHOLD = 12.0

_feat_model  = None
_feat_scaler = None
try:
    _mpath = os.path.join(_base, 'drawing_model.pkl')
    _spath = os.path.join(_base, 'drawing_scaler.pkl')
    if os.path.exists(_mpath) and os.path.exists(_spath):
        _feat_model  = joblib.load(_mpath)
        _feat_scaler = joblib.load(_spath)
        print("DEBUG [utils]: Feature-based drawing model loaded OK.")
    else:
        print("DEBUG [utils]: drawing_model.pkl not found — geometric-only mode.")
except Exception as e:
    print(f"DEBUG [utils]: Feature model load error: {e}")
    _feat_model = _feat_scaler = None

model = None
try:
    import tensorflow.keras as keras
    _keras_path = os.path.join(_base, 'keras_model.h5')
    if os.path.exists(_keras_path):
        model = keras.models.load_model(_keras_path)
        print("DEBUG [utils]: Keras CNN model loaded (fallback).")
except Exception as e:
    print(f"DEBUG [utils]: Keras load error: {e}")
    model = None

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

HEALTHY_TIPS = [
    'Your drawing shows smooth, consistent lines. Maintain hand-eye coordination exercises.',
    'Healthy pattern detected. Antioxidant-rich diets (berries, nuts) support neurological health.',
    'Smooth control observed. Regular aerobic exercise is one of the best ways to protect brain function.',
    'Excellent motor control. Ensure 7-9 hours of quality sleep to maintain cognitive sharpness.',
    'Steady patterns detected. Staying hydrated and reducing stress helps maintain motor skills.',
]
WEAK_TIPS = [
    'Irregular lines detected. Consider practising larger, deliberate writing movements.',
    'Irregularity observed. Physical therapy focused on balance can be highly beneficial.',
    'Weak pattern detected. A Mediterranean diet is often recommended for neurological health.',
    'Tremor-like patterns noted. Consult a neurologist for a professional screening.',
    'Shaky lines detected. Simple finger-tapping exercises can help monitor motor skills.',
]

def _select_tip(tips, image_path):
    
    import hashlib
    try:
        with open(image_path, 'rb') as f:
            h = int(hashlib.md5(f.read()).hexdigest(), 16)
    except Exception:
        h = 0
    return tips[h % len(tips)]

def _geometric_spiral_analysis(gray_img):
    
    arr = np.asarray(gray_img, dtype=np.float32)

    inv = 255.0 - arr

    threshold = 20.0
    drawn_mask = inv > threshold
    n_drawn = int(np.sum(drawn_mask))

    print(f"DEBUG [geom]: drawn_pixels={n_drawn}")

    if n_drawn < 80:
        return True, 0.0, {}  # blank

    ys, xs = np.where(drawn_mask)

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # -π to π

    r_min_cutoff = 3.0
    valid = r > r_min_cutoff
    r = r[valid]
    theta = theta[valid]

    if len(r) < 50:
        return True, 0.0, {}

    n_sectors = 12
    sector_size = 2 * np.pi / n_sectors
    theta_pos = theta + np.pi
    sectors_covered = len(set((theta_pos / sector_size).astype(int) % n_sectors))
    print(f"DEBUG [geom]: sectors_covered={sectors_covered}")
    if sectors_covered < 2:
        return 'not_spiral', 0.0, {}

    w = float(xs.max() - xs.min()) + 1.0
    h = float(ys.max() - ys.min()) + 1.0
    aspect = max(w, h) / (min(w, h) + 1e-6)
    print(f"DEBUG [geom]: aspect={aspect:.2f}")
    if aspect > 12.0:
        return 'not_spiral', 0.0, {}

    r_mean = float(np.mean(r))
    r_range = float(r.max() - r.min())
    r_variation = r_range / (r_mean + 1e-6)
    print(f"DEBUG [geom]: r_variation={r_variation:.3f}, r_mean={r_mean:.1f}")
    if r_mean < 1.0 or r_variation < 0.10:
        return 'not_spiral', 0.0, {}

    img_w = gray_img.size[0] if hasattr(gray_img, 'size') else gray_img.shape[1]
    img_h = gray_img.size[1] if hasattr(gray_img, 'size') else gray_img.shape[0]
    bbox_area = w * h
    canvas_area = float(img_w * img_h)
    coverage = bbox_area / canvas_area
    print(f"DEBUG [geom]: bbox_coverage={coverage:.3f}")
    if coverage < 0.01:
        return 'not_spiral', 0.0, {}

    order = np.argsort(theta)
    r_sorted = r[order]
    theta_sorted = theta[order]

    window = max(10, len(r_sorted) // 30)
    def moving_avg(x, w):
        return np.convolve(x, np.ones(w) / w, mode='same')

    r_expected = moving_avg(r_sorted, window)
    residuals = r_sorted - r_expected

    local_window = max(8, len(residuals) // 40)
    rms_list = []
    step = max(1, local_window // 2)
    for i in range(0, len(residuals) - local_window, step):
        chunk = residuals[i:i + local_window]
        rms_list.append(float(np.sqrt(np.mean(chunk**2))))

    if not rms_list:
        return False, 0.0, {}

    tremor_rms = float(np.mean(rms_list))

    dr = np.diff(r_sorted)
    sign_changes = int(np.sum(np.diff(np.sign(dr)) != 0))
    reversal_rate = sign_changes / max(len(dr), 1)

    angle_gaps = np.diff(theta_sorted)
    large_gaps = np.sum(np.abs(angle_gaps) > 0.5)
    gap_ratio = large_gaps / max(len(angle_gaps), 1)

    SMALL_SPIRAL_CUTOFF = 70.0  # pixels

    if r_mean <= SMALL_SPIRAL_CUTOFF:

        scaled_tremor = (tremor_rms / (r_mean + 1e-6)) * 40.0
        print(f"DEBUG [geom]: small-spiral path, scaled_tremor={scaled_tremor:.3f}")
    else:

        scaled_tremor = tremor_rms * 0.6
        print(f"DEBUG [geom]: large-spiral path, scaled_tremor={scaled_tremor:.3f}")

    combined_tremor = scaled_tremor * 0.6 + reversal_rate * 20.0 * 0.3 + gap_ratio * 12.0 * 0.1

    metrics = {
        'tremor_rms': round(tremor_rms, 4),
        'reversal_rate': round(reversal_rate, 4),
        'gap_ratio': round(gap_ratio, 4),
        'combined_tremor': round(combined_tremor, 4),
        'sectors_covered': sectors_covered,
        'aspect': round(aspect, 2),
        'r_variation': round(r_variation, 3),
        'drawn_pixels': n_drawn,
    }

    print(f"DEBUG [geom]: tremor_rms={tremor_rms:.4f}, reversal_rate={reversal_rate:.4f}, "
          f"gap_ratio={gap_ratio:.4f}, combined_tremor={combined_tremor:.4f}")

    return False, combined_tremor, metrics

def _geometric_classify(tremor_index, metrics):
    
    if tremor_index > PD_THRESHOLD:

        excess = tremor_index - PD_THRESHOLD

        conf = round(min(70.0 + min(25.0, excess * 3.5), 97.0), 2)

        if tremor_index > 16.0:
            display_label = "Strong Parkinson's Indicators Detected"
        elif tremor_index > 11.0:
            display_label = "Parkinson's Pattern Observed"
        else:
            display_label = "Weak Parkinson's Indicators Detected"

        print(f"DEBUG [classify]: -> Parkinson (tremor={tremor_index:.3f} > threshold={PD_THRESHOLD}), conf={conf}%")
        return 'Parkinson', display_label, conf

    else:

        stability = PD_THRESHOLD - tremor_index
        conf = round(min(68.0 + min(28.0, stability * 4.0), 96.0), 2)

        if conf > 88:
            display_label = "Healthy Control Sample"
        else:
            display_label = "Likely Healthy Sample"

        print(f"DEBUG [classify]: -> Healthy (tremor={tremor_index:.3f} < threshold={PD_THRESHOLD}), conf={conf}%")
        return 'Healthy', display_label, conf

def predictImg(image_path='static/img/test.jpg'):
    
    from PIL import Image, ImageOps

    print(f"\n{'='*60}")
    print(f"DEBUG [predictImg]: Analysing {image_path}")

    if not os.path.exists(image_path):
        return (None, 'No Image Uploaded',
                'Please draw or upload a spiral image first, then click Analyse.', '0')

    try:
        image_raw = Image.open(image_path)
        print(f"DEBUG [predictImg]: Image mode={image_raw.mode}, size={image_raw.size}")
        if image_raw.mode in ('RGBA', 'LA') or (image_raw.mode == 'P' and 'transparency' in image_raw.info):
            bg = Image.new('RGB', image_raw.size, (255, 255, 255))
            mask = image_raw.convert('RGBA').split()[3]
            bg.paste(image_raw, mask)
            image_raw = bg
        else:
            image_raw = image_raw.convert('RGB')
    except Exception as e:
        print(f"DEBUG [predictImg]: Image load failed: {e}")
        return (None, 'Invalid Image',
                'The uploaded file could not be read. Please try again.', '0')

    gray = image_raw.convert('L')
    healthy_tip = _select_tip(HEALTHY_TIPS, image_path)
    weak_tip    = _select_tip(WEAK_TIPS, image_path)

    status, tremor_index, metrics = _geometric_spiral_analysis(gray)

    if status is True:
        return (None, 'No Drawing Detected',
                'The canvas appears empty. Please draw a spiral before clicking Analyse.', '0')

    if status == 'not_spiral':
        return (None, 'Not a Spiral Drawing',
                'Please draw a spiral pattern (a coil starting from the centre outward). '
                'Random shapes or straight lines cannot be analysed for Parkinson\'s indicators.', '0')

    gray_arr = np.asarray(gray, dtype=np.float32)
    inv_arr = 255.0 - gray_arr
    bg_mask = inv_arr < 20.0
    is_digital_canvas = False
    if np.sum(bg_mask) > 0:
        bg_std = float(np.std(gray_arr[bg_mask]))
        if bg_std < 8.0:
            is_digital_canvas = True
    print(f"DEBUG [predictImg]: is_digital_canvas={is_digital_canvas}")

    label, display_label, base_conf = _geometric_classify(tremor_index, metrics)
    suggestion = weak_tip if label == 'Parkinson' else healthy_tip

    print(f"DEBUG [predictImg]: Geometric decision: label={label}, conf={base_conf}%")

    if not is_digital_canvas and _feat_model is not None and _feat_scaler is not None:
        try:
            from skimage.feature import hog, local_binary_pattern
            from skimage.transform import resize

            arr_norm = np.asarray(gray, dtype=np.float32) / 255.0
            arr_resized = resize(arr_norm, (128, 128), anti_aliasing=True)

            hog_feats = hog(arr_resized, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
            lbp = local_binary_pattern(arr_resized, P=24, R=3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)

            feats = np.concatenate([hog_feats, lbp_hist]).reshape(1, -1)
            feats_scaled = _feat_scaler.transform(feats)

            pred_label_idx = int(_feat_model.predict(feats_scaled)[0])

            svm_is_parkinson = (pred_label_idx == 1)

            if hasattr(_feat_model, 'predict_proba'):
                pred_proba = _feat_model.predict_proba(feats_scaled)[0]
                raw_conf = float(max(pred_proba))
            else:
                raw_conf = 0.75  # default if no proba

            print(f"DEBUG [predictImg]: SVM pred={pred_label_idx} (PD={svm_is_parkinson}), raw_conf={raw_conf:.3f}")

            svm_label = 'Parkinson' if svm_is_parkinson else 'Healthy'
            svm_conf  = round(min(98.0, raw_conf * 100.0), 2)

            if svm_label == 'Parkinson':
                if svm_conf > 88: disp = "Strong Parkinson's Indicators Detected"
                elif svm_conf > 78: disp = "Parkinson's Pattern Observed"
                else: disp = "Weak Parkinson's Indicators Detected"
                sug = weak_tip
            else:
                if svm_conf > 85: disp = "Healthy Control Sample"
                else: disp = "Likely Healthy Sample"
                sug = healthy_tip

            print(f"DEBUG [predictImg]: SVM overrides geometric: label={svm_label}, conf={svm_conf}%")
            return svm_label, disp, sug, f'{svm_conf:.2f}'

        except Exception as e:
            print(f"DEBUG [predictImg]: SVM error: {e} — falling back to geometric")

    if not is_digital_canvas and model is not None:
        try:
            size = (224, 224)
            try:
                resample = image_raw.ANTIALIAS
            except AttributeError:
                from PIL import Image as _PIL
                resample = _PIL.Resampling.LANCZOS

            img_resized = ImageOps.fit(image_raw, size, resample)
            img_arr = np.asarray(img_resized, dtype=np.float32)
            data[0] = (img_arr / 127.0) - 1
            prediction = model.predict(data)
            idx = int(np.argmax(prediction))
            cnn_label = 'Parkinson' if idx == 1 else 'Healthy'
            cnn_conf  = round(float(np.max(prediction)) * 100, 2)
            print(f"DEBUG [predictImg]: Keras CNN: label={cnn_label}, conf={cnn_conf:.1f}%")

            if cnn_label == 'Parkinson':
                cnn_disp = "Parkinson's Pattern Observed"
                cnn_sug  = weak_tip
            else:
                cnn_disp = "Likely Healthy Sample"
                cnn_sug  = healthy_tip

            return cnn_label, cnn_disp, cnn_sug, f'{cnn_conf:.2f}'

        except Exception as e:
            print(f"DEBUG [predictImg]: Keras error: {e}")

    print(f"DEBUG [predictImg]: Returning pure geometric result: {label}, {base_conf}%")
    print('='*60)
    return label, display_label, suggestion, f'{base_conf:.2f}'
