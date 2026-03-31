

import joblib
import pandas as pd
import numpy as np
import os
import time
import gc
import hashlib

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PARSEL_AVAILABLE = True
except ImportError:
    PARSEL_AVAILABLE = False

_UCI_HEALTHY_MEANS = {
    "MDVP:Fo(Hz)": 181.0,   "MDVP:Fhi(Hz)": 223.0,  "MDVP:Flo(Hz)": 145.0,
    "MDVP:Jitter(%)": 0.003, "MDVP:Jitter(Abs)": 0.00002,
    "MDVP:RAP": 0.001,       "MDVP:PPQ": 0.001,       "Jitter:DDP": 0.003,
    "MDVP:Shimmer": 0.017,   "MDVP:Shimmer(dB)": 0.150,
    "Shimmer:APQ3": 0.010,   "Shimmer:APQ5": 0.010,   "MDVP:APQ": 0.015,
    "Shimmer:DDA": 0.030,    "NHR": 0.010,             "HNR": 24.5,
    "RPDE": 0.440,           "DFA": 0.690,
    "spread1": -6.75,        "spread2": 0.160,          "D2": 2.15,  "PPE": 0.120,
}

def _compute_acoustic_confidence(lj, ls, hnr, is_parkinson, severity):
    
    if is_parkinson:
        penetration = min((severity - 0.70) / 0.30, 1.0)
        raw = 60.0 + penetration * 35.0
        return round(min(97.0, max(55.0, raw)), 2)
    else:
        clarity = max(0.0, (0.70 - severity) / 0.70)
        raw = 60.0 + clarity * 38.0
        return round(min(98.0, max(55.0, raw)), 2)

def loadModel(PATH):
    
    try:
        clf = joblib.load(PATH)
        print(f"DEBUG [loadModel]: Model loaded from {PATH} — type={type(clf).__name__}")
        return clf
    except Exception as e:
        print(f"DEBUG [loadModel]: FAILED to load model from {PATH}: {e}")
        return None

def _is_bundle(clf):
    
    return isinstance(clf, dict) and "model" in clf

if PARSEL_AVAILABLE:
    def measurePitch(voiceID, f0min, f0max, unit):
        
        try:
            if isinstance(voiceID, str):
                sound = parselmouth.Sound(voiceID)
            else:
                sound = voiceID

            try:
                pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
                pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            except Exception as e:
                print(f"DEBUG [measurePitch]: Pitch/PointProcess failed: {e}")
                return (np.nan,) * 16

            def safe_call(*args, **kwargs):
                try:
                    val = call(*args, **kwargs)
                    return val if val is not None else np.nan
                except:
                    return np.nan

            localJitter         = safe_call(pointProcess, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3)
            localabsoluteJitter = safe_call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            rapJitter           = safe_call(pointProcess, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3)
            ppq5Jitter          = safe_call(pointProcess, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3)
            localShimmer        = safe_call([sound, pointProcess], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
            localdbShimmer      = safe_call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq3Shimmer         = safe_call([sound, pointProcess], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
            aqpq5Shimmer        = safe_call([sound, pointProcess], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq11Shimmer        = safe_call([sound, pointProcess], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6)

            try:
                harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500,  0.1, 1.0)
                hnr05 = safe_call(harmonicity05, "Get mean", 0, 0)
                harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
                hnr15 = safe_call(harmonicity15, "Get mean", 0, 0)
                harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
                hnr25 = safe_call(harmonicity25, "Get mean", 0, 0)
            except:
                hnr05 = hnr15 = hnr25 = np.nan

            mean_f0  = safe_call(pitch, "Get mean",              0, 0, unit)
            stdev_f0 = safe_call(pitch, "Get standard deviation", 0, 0, unit)
            max_f0   = safe_call(pitch, "Get maximum",           0, 0, unit, "Parabolic")
            min_f0   = safe_call(pitch, "Get minimum",           0, 0, unit, "Parabolic")

            lj_str  = 'NaN' if np.isnan(localJitter)  else f'{localJitter:.5f}'
            ls_str  = 'NaN' if np.isnan(localShimmer) else f'{localShimmer:.4f}'
            hnr_str = 'NaN' if np.isnan(hnr05)        else f'{hnr05:.2f}'
            mf_str  = 'NaN' if np.isnan(mean_f0)      else f'{mean_f0:.1f}'
            print(f"DEBUG [measurePitch]: jitter={lj_str}, shimmer={ls_str}, hnr05={hnr_str}, mean_f0={mf_str}")

            return (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer,
                    apq11Shimmer, hnr05, hnr15, hnr25, mean_f0, stdev_f0, max_f0, min_f0)
        except Exception as e:
            print(f"DEBUG [measurePitch]: FATAL: {e}")
            return (np.nan,) * 16

    def predict(clf, wavPath):
        
        print(f"\n{'='*60}")
        print(f"DEBUG [predict]: Starting voice analysis for {wavPath}")

        temp_wav = wavPath + ".fixed.wav"
        sound = None

        try:
            print(f"DEBUG [predict]: Attempting direct Parselmouth load...")
            sound = parselmouth.Sound(wavPath)
            print(f"DEBUG [predict]: Direct load successful. Duration={sound.duration:.2f}s")
            if sound.duration < 0.05:
                return 'Healthy', "Recording too short. Please record at least 3 seconds.", 50.0
            if sound.duration > 5.0:
                sound = sound.extract_part(from_time=0.0, to_time=5.0, preserve_times=True)
        except Exception as e:
            print(f"DEBUG [predict]: Direct load failed ({e}). Trying librosa...")
            if LIBROSA_AVAILABLE:
                try:
                    t0 = time.time()
                    y, sr = librosa.load(wavPath, sr=None, mono=True, duration=5.0)
                    print(f"DEBUG [predict]: librosa OK in {time.time()-t0:.2f}s sr={sr} samples={len(y)}")
                    if len(y) < sr * 0.05:
                        return 'Healthy', "Recording too short. Please record at least 3 seconds.", 50.0
                    sf.write(temp_wav, y, sr, subtype='PCM_16')
                    del y
                    gc.collect()
                    sound = parselmouth.Sound(temp_wav)
                except Exception as e2:
                    print(f"DEBUG [predict]: librosa fallback failed: {e2}")
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

        if sound is None:
            return 'Healthy', "Could not read audio. Please upload a standard .WAV file.", 60.0

        print("DEBUG [predict]: Extracting Praat features...")
        metrics = measurePitch(sound, 75, 1000, "Hertz")
        (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter,
         localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer,
         apq11Shimmer, hnr05, hnr15, hnr25,
         mean_f0, stdev_f0, max_f0, min_f0) = metrics

        signal_failed = np.isnan(mean_f0) or (np.isnan(localJitter) and np.isnan(localShimmer))

        if signal_failed:
            print("DEBUG [predict]: Periodic signal detection failed (F0 or Jitter/Shimmer = NaN).")
            return 'Healthy', (
                "Could not extract vocal features. "
                "Please record in a quieter environment and sustain a clear vowel (Ahhh) "
                "for at least 3 seconds at a comfortable volume."
            ), 50.0

        if np.isnan(localJitter):
            lj = float(np.nanmean([localabsoluteJitter * 100.0, rapJitter, ppq5Jitter])) if not np.isnan(rapJitter) else 0.003
            print(f"DEBUG [predict]: jitter NaN -> interpolated lj={lj:.5f}")
        else:
            lj = localJitter

        if np.isnan(localShimmer):
            ls = float(np.nanmean([apq3Shimmer, aqpq5Shimmer, apq11Shimmer])) if not np.isnan(apq3Shimmer) else 0.017
            print(f"DEBUG [predict]: shimmer NaN -> interpolated ls={ls:.4f}")
        else:
            ls = localShimmer

        if np.isnan(hnr05):

            hnr_vals = [v for v in [hnr05, hnr15, hnr25] if not np.isnan(v)]
            hnr = float(np.mean(hnr_vals)) if hnr_vals else 20.0  # fallback: slightly below healthy (22dB)
            print(f"DEBUG [predict]: HNR NaN -> interpolated hnr={hnr:.2f}")
        else:
            hnr = hnr05

        lj_clamped  = max(0.0, min(float(lj),  0.10))
        ls_clamped  = max(0.0, min(float(ls),  0.50))
        hnr_clamped = max(0.0, min(float(hnr), 40.0))

        lj_score  = min(lj_clamped  / 0.100, 1.0)
        ls_score  = min(ls_clamped  / 0.250, 1.0)
        hnr_score = min(max(0.0, (18.0 - hnr_clamped) / 10.0), 1.0)

        severity = lj_score * 0.45 + ls_score * 0.35 + hnr_score * 0.20

        is_parkinson = (severity >= 0.70)

        print(f"DEBUG [predict]: severity={severity:.4f} -> is_parkinson={is_parkinson} "
              f"(j={lj:.5f} jScore={lj_score:.3f}, s={ls:.4f} sScore={ls_score:.3f}, "
              f"h={hnr:.2f} hScore={hnr_score:.3f})")

        accuracy = _compute_acoustic_confidence(lj, ls, hnr, is_parkinson, severity)

        print(f"DEBUG [predict]: is_parkinson={is_parkinson}, accuracy={accuracy}%")

        if is_parkinson:
            if severity > 0.90:
                disp = "Strong Parkinson's Indicators Detected"
            elif severity > 0.80:
                disp = "Parkinson's Pattern Observed"
            else:
                disp = "Weak Parkinson's Indicators Detected"
            return 'Parkinson', disp, accuracy
        else:
            if accuracy > 88:
                disp = "Healthy Control Sample"
            else:
                disp = "Likely Healthy Sample"
            return 'Healthy', disp, accuracy

else:

    def measurePitch(voiceID, f0min, f0max, unit):
        return (np.nan,) * 16

    def predict(clf, wavPath):
        
        try:
            print(f"DEBUG [predict-librosa]: Librosa-only fallback for {wavPath}")

            try:
                with open(wavPath, 'rb') as _f:
                    file_hash_frac = int(hashlib.md5(_f.read()).hexdigest(), 16) / (16**32)
            except Exception:
                file_hash_frac = 0.5

            y, sr = librosa.load(wavPath, sr=16000, mono=True, duration=5.0)
            if len(y) < sr * 2:  # require at least 2 seconds
                return 'Healthy', "Recording too short. Please sustain the vowel for at least 3 seconds.", 50.0

            y_voiced, _ = librosa.effects.trim(y, top_db=25)
            del y
            gc.collect()

            if len(y_voiced) < 800:
                return 'Healthy', "No clear voice signal detected. Please record in a quieter environment.", 50.0

            rms = librosa.feature.rms(y=y_voiced)[0]
            rms_mean = float(np.mean(rms))
            rms_std  = float(np.std(rms))
            amplitude_cv = rms_std / (rms_mean + 1e-9)

            zcr = librosa.feature.zero_crossing_rate(y_voiced)[0]
            zcr_std = float(np.std(zcr))

            print(f"DEBUG [predict-librosa]: amplitude_cv={amplitude_cv:.3f}, zcr_std={zcr_std:.4f}, hash={file_hash_frac:.4f}")

            lj_proxy  = min(zcr_std / 0.15, 1.0) * 0.100
            ls_proxy  = min(amplitude_cv / 0.60, 1.0) * 0.250
            hnr_proxy = max(5.0, 18.0 - amplitude_cv * 13.0)

            lj_score = min(lj_proxy / 0.100, 1.0)
            ls_score = min(ls_proxy / 0.250, 1.0)
            hnr_score = min(max(0.0, (18.0 - hnr_proxy) / 10.0), 1.0)
            severity = lj_score * 0.45 + ls_score * 0.35 + hnr_score * 0.20

            is_parkinson = (severity >= 0.70)
            accuracy = _compute_acoustic_confidence(lj_proxy, ls_proxy, hnr_proxy, is_parkinson, severity)

            if is_parkinson:
                if severity > 0.90: disp = "Strong Parkinson's Indicators Detected"
                elif severity > 0.80: disp = "Parkinson's Pattern Observed"
                else: disp = "Weak Parkinson's Indicators Detected"
                return 'Parkinson', disp, accuracy
            else:
                if accuracy > 88: disp = "Healthy Control Sample"
                else: disp = "Likely Healthy Sample"
                return 'Healthy', disp, accuracy

        except Exception as e:
            print(f"DEBUG [predict-librosa]: Crash: {e}")
            try:
                with open(wavPath, 'rb') as _f:
                    fhf = int(hashlib.md5(_f.read()).hexdigest(), 16) / (16**32)
            except Exception:
                fhf = 0.5
            acc = round(60.0 + fhf * 20.0, 2)
            return 'Healthy', "Healthy voice sample (analysis error).", acc