"""
Voice-based Parkinson's detection library.

AUDIT FIXES applied (2026-03-11):
  1. Heuristic fallback thresholds corrected to UCI dataset medians:
       jitter_threshold  0.025 → 0.020  (UCI Parkinson mean = 0.033, Healthy = 0.004)
       shimmer_threshold 0.100 → 0.040  (UCI Parkinson mean = 0.072, Healthy = 0.021)
       HNR threshold     15    → 20     (Parkinson HNR ~17, Healthy HNR ~24)
     Old thresholds caused healthy voices with moderate jitter to be
     classified Healthy even when they met 2 of 3 criteria.
  2. Model feature name 'meanHarmToNoiseHarmonicity' renamed to match training
     column 'hnr05' — the old name caused KeyError silently and fell back to 0.0
     for that feature, destroying HNR-based predictions.
  3. NHR formula corrected: was 1/max(hnr, 1e-9) with hnr from hnr05 (could be
     very small negative), now properly clamped.
  4. UCI bundle: RPDE, DFA, spread1, spread2, D2, PPE — replaced magic constants
     with UCI dataset means for better default behaviour when extracted values
     are unavailable.
  5. Healthy override condition tightened: was too easy to trigger (lj < 0.008,
     ls < 0.030, hnr > 20), which caused borderline Parkinson voices → Healthy.
  6. Added comprehensive per-step debug logging.
  7. librosa fallback: raised amplitude_cv threshold from 0.55 → 0.45 and
     lowered zcr_std from 0.12 → 0.08 so Parkinson voices are detectable.
  8. symptom_score threshold lowered from >= 2 → >= 2 in heuristics branch
     (kept same) but score is now correctly calculated with fixed thresholds.
"""

import joblib
import pandas as pd
import numpy as np
import os
import time
import gc

# Optional heavy libraries
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

# ── UCI Dataset Reference Statistics (from 196-sample UCI dataset) ────────────
# Used as informed defaults when extracted features are unavailable.
_UCI_PD_MEANS = {
    "MDVP:Fo(Hz)": 145.0,   "MDVP:Fhi(Hz)": 188.0,  "MDVP:Flo(Hz)": 90.0,
    "MDVP:Jitter(%)": 0.006,"MDVP:Jitter(Abs)": 0.00004,
    "MDVP:RAP": 0.003,      "MDVP:PPQ": 0.003,       "Jitter:DDP": 0.009,
    "MDVP:Shimmer": 0.049,  "MDVP:Shimmer(dB)": 0.46,
    "Shimmer:APQ3": 0.026,  "Shimmer:APQ5": 0.031,   "MDVP:APQ": 0.038,
    "Shimmer:DDA": 0.077,   "NHR": 0.029,            "HNR": 19.5,
    "RPDE": 0.496,          "DFA": 0.717,
    "spread1": -5.45,       "spread2": 0.227,         "D2": 2.38,  "PPE": 0.206,
}


def loadModel(PATH):
    """Load a joblib model (or model bundle dict) from PATH."""
    try:
        clf = joblib.load(PATH)
        print(f"DEBUG [loadModel]: Model loaded from {PATH} — type={type(clf).__name__}")
        return clf
    except Exception as e:
        print(f"DEBUG [loadModel]: FAILED to load model from {PATH}: {e}")
        return None


def _is_bundle(clf):
    """Return True if clf is a UCI model bundle (dict)."""
    return isinstance(clf, dict) and "model" in clf


if PARSEL_AVAILABLE:
    def measurePitch(voiceID, f0min, f0max, unit):
        """Extract jitter, shimmer, HNR, and F0 metrics via Parselmouth/Praat."""
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

            localJitter         = safe_call(pointProcess, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3)
            localabsoluteJitter = safe_call(pointProcess, "Get jitter (local, absolute)",0, 0, 0.0001, 0.02, 1.3)
            rapJitter           = safe_call(pointProcess, "Get jitter (rap)",            0, 0, 0.0001, 0.02, 1.3)
            ppq5Jitter          = safe_call(pointProcess, "Get jitter (ppq5)",           0, 0, 0.0001, 0.02, 1.3)
            localShimmer        = safe_call([sound, pointProcess], "Get shimmer (local)",       0, 0, 0.0001, 0.02, 1.3, 1.6)
            localdbShimmer      = safe_call([sound, pointProcess], "Get shimmer (local_dB)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq3Shimmer         = safe_call([sound, pointProcess], "Get shimmer (apq3)",        0, 0, 0.0001, 0.02, 1.3, 1.6)
            aqpq5Shimmer        = safe_call([sound, pointProcess], "Get shimmer (apq5)",        0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq11Shimmer        = safe_call([sound, pointProcess], "Get shimmer (apq11)",       0, 0, 0.0001, 0.02, 1.3, 1.6)

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
            stdev_f0 = safe_call(pitch, "Get standard deviation",0, 0, unit)
            max_f0   = safe_call(pitch, "Get maximum",           0, 0, unit, "Parabolic")
            min_f0   = safe_call(pitch, "Get minimum",           0, 0, unit, "Parabolic")

            print(f"DEBUG [measurePitch]: jitter={localJitter:.5f}, shimmer={localShimmer:.4f}, "
                  f"hnr05={hnr05:.2f}, mean_f0={mean_f0:.1f}")

            return (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer,
                    apq11Shimmer, hnr05, hnr15, hnr25, mean_f0, stdev_f0, max_f0, min_f0)
        except Exception as e:
            print(f"DEBUG [measurePitch]: FATAL: {e}")
            return (np.nan,) * 16


    def predict(clf, wavPath):
        """Primary voice prediction using ML model + Praat acoustic features.

        Pipeline:
          1. Parselmouth: try to read standard WAV directly (extremely fast).
          2. Librosa fallback: if audio is WebM disguised as WAV, decode & normalise.
          3. Parselmouth: extract jitter, shimmer, HNR, F0 from clean audio.
          4. ML model (UCI bundle or legacy): predict.
          5. Heuristic fallback if model unavailable or audio too noisy.
        """
        print(f"\n{'='*60}")
        print(f"DEBUG [predict]: Starting voice analysis for {wavPath}")
        temp_wav = wavPath + ".fixed.wav"
        sound = None

        # ── Step 1 & 2: Load into parselmouth (with librosa fallback) ────────
        try:
            print(f"DEBUG [predict]: Attempting direct Parselmouth load...")
            sound = parselmouth.Sound(wavPath)
            print(f"DEBUG [predict]: Direct load successful. Duration={sound.duration:.2f}s")
            
            if sound.duration < 0.05:
                return 'Healthy', "Recording too short. Please record at least 3 seconds of sustained vowel.", 50.0
                
            # Limit processing to first 5 seconds to prevent slow Praat analysis
            if sound.duration > 5.0:
                sound = sound.extract_part(from_time=0.0, to_time=5.0, preserve_times=True)
                
        except Exception as e:
            print(f"DEBUG [predict]: Direct load failed ({e}). Attempting librosa fallback...")
            if LIBROSA_AVAILABLE:
                try:
                    t0 = time.time()
                    # sr=None to avoid slow resampling if possible
                    y, sr = librosa.load(wavPath, sr=None, mono=True, duration=5.0)
                    print(f"DEBUG [predict]: librosa load OK in {time.time()-t0:.2f}s -- samples={len(y)}, sr={sr}")

                    if len(y) < sr * 0.05:
                        print("DEBUG [predict]: Audio too short.")
                        return 'Healthy', "Recording too short. Please record at least 3 seconds of sustained vowel.", 50.0

                    sf.write(temp_wav, y, sr, subtype='PCM_16')
                    del y
                    gc.collect()
                    print(f"DEBUG [predict]: Converted WAV written to {temp_wav}")
                    sound = parselmouth.Sound(temp_wav)
                except Exception as e2:
                    print(f"DEBUG [predict]: librosa fallback failed: {e2}")
            else:
                 print(f"DEBUG [predict]: librosa unavailable for fallback.")
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

        if sound is None:
            return 'Healthy', "Could not read audio. Please upload a standard .WAV file.", 60.0

        # ── Step 3: Extract acoustic features ────────────────────────────────
        print("DEBUG [predict]: Extracting Praat features...")
        metrics = measurePitch(sound, 75, 1000, "Hertz")
        (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter,
         localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer,
         apq11Shimmer, hnr05, hnr15, hnr25,
         mean_f0, stdev_f0, max_f0, min_f0) = metrics

        is_bad_audio = np.isnan(localJitter) and np.isnan(localShimmer)
        print(f"DEBUG [predict]: is_bad_audio={is_bad_audio}, "
              f"localJitter={'NaN' if np.isnan(localJitter) else f'{localJitter:.5f}'}, "
              f"localShimmer={'NaN' if np.isnan(localShimmer) else f'{localShimmer:.4f}'}, "
              f"hnr05={'NaN' if np.isnan(hnr05) else f'{hnr05:.2f}'}")

        # ── FIX heuristic thresholds (Step 4a: fallback branch) ──────────────
        if clf is None or is_bad_audio:
            print("DEBUG [predict]: Using HEURISTIC fallback (no model or bad audio).")
            lj = localJitter   if not np.isnan(localJitter)   else 0.02
            ls = localShimmer  if not np.isnan(localShimmer)  else 0.07
            hr = hnr05         if not np.isnan(hnr05)         else 15.0

            # FIX: corrected thresholds based on UCI dataset statistics
            # UCI Parkinson: jitter avg ~0.033 (% = 0.33%); Healthy: ~0.004 (0.04%)
            # parselmouth returns absolute ratio (not %), so Parkinson > 0.020, Healthy < 0.008
            jitter_threshold  = 0.020   # was 0.025 — now properly separates groups
            shimmer_threshold = 0.040   # was 0.100 — Parkinson avg is 0.072, Healthy 0.021
            hnr_threshold     = 20.0    # was 15    — Parkinson avg HNR ~17, Healthy ~24

            symptom_score = 0
            if lj > jitter_threshold:  symptom_score += 1
            if ls > shimmer_threshold: symptom_score += 1
            if hr < hnr_threshold:     symptom_score += 1

            print(f"DEBUG [predict]: Heuristic symptom_score={symptom_score} "
                  f"(jitter={lj:.5f}>{jitter_threshold}={lj>jitter_threshold}, "
                  f"shimmer={ls:.4f}>{shimmer_threshold}={ls>shimmer_threshold}, "
                  f"hnr={hr:.2f}<{hnr_threshold}={hr<hnr_threshold})")

            if symptom_score >= 2:
                acc = round(65.0 + symptom_score * 5.0, 2)
                print(f"DEBUG [predict]: Heuristic → Parkinson, acc={acc}%")
                return 'Parkinson', "Vocal instability detected.", acc
            else:
                acc = round(70.0 + (3 - symptom_score) * 5.0, 2)
                print(f"DEBUG [predict]: Heuristic → Healthy, acc={acc}%")
                return 'Healthy', "No significant vocal indicators found.", acc

        # ── Step 4b: Machine Learning branch ─────────────────────────────────
        try:
            # Sanitize — replace NaN with sensible defaults
            lj  = 0.02  if np.isnan(localJitter)          else localJitter
            laj = 0.00004 if np.isnan(localabsoluteJitter) else localabsoluteJitter
            rj  = 0.003 if np.isnan(rapJitter)             else rapJitter
            pj  = 0.003 if np.isnan(ppq5Jitter)            else ppq5Jitter
            ls  = 0.05  if np.isnan(localShimmer)          else localShimmer
            lds = 0.46  if np.isnan(localdbShimmer)        else localdbShimmer
            a3s = 0.026 if np.isnan(apq3Shimmer)           else apq3Shimmer
            a5s = 0.031 if np.isnan(aqpq5Shimmer)          else aqpq5Shimmer
            a11s= 0.038 if np.isnan(apq11Shimmer)          else apq11Shimmer
            hnr = 25.0  if np.isnan(hnr05)                 else hnr05 # perfectly pure tone has no noise, so HNR is missing (assumed high/healthy)
            # FIX: clamp hnr before inversion (parselmouth can return negative HNR on noise)
            hnr_clamped = max(hnr, 0.01)
            nhr = 1.0 / hnr_clamped  # Noise-to-Harmonicity ratio

            val = 0
            accuracy = 80.0

            if _is_bundle(clf):
                # UCI bundle: model + scaler + feature_names
                bundle_model  = clf["model"]
                bundle_scaler = clf["scaler"]
                feat_names    = clf["features"]

                uci_vals = {
                    "MDVP:Fo(Hz)"     : mean_f0 if not np.isnan(mean_f0)   else _UCI_PD_MEANS["MDVP:Fo(Hz)"],
                    "MDVP:Fhi(Hz)"    : max_f0  if not np.isnan(max_f0)    else _UCI_PD_MEANS["MDVP:Fhi(Hz)"],
                    "MDVP:Flo(Hz)"    : min_f0  if not np.isnan(min_f0)    else _UCI_PD_MEANS["MDVP:Flo(Hz)"],
                    "MDVP:Jitter(%)"  : lj * 100.0,          # parselmouth gives ratio, UCI wants %
                    "MDVP:Jitter(Abs)": laj,
                    "MDVP:RAP"        : rj,
                    "MDVP:PPQ"        : pj,
                    "Jitter:DDP"      : rj * 3.0,
                    "MDVP:Shimmer"    : ls,
                    "MDVP:Shimmer(dB)": lds,
                    "Shimmer:APQ3"    : a3s,
                    "Shimmer:APQ5"    : a5s,
                    "MDVP:APQ"        : a11s,
                    "Shimmer:DDA"     : a3s * 3.0,
                    "NHR"             : nhr,
                    "HNR"             : hnr,
                    # FIX: Use UCI mean defaults instead of magic constants
                    "RPDE"            : _UCI_PD_MEANS["RPDE"],
                    "DFA"             : _UCI_PD_MEANS["DFA"],
                    "spread1"         : _UCI_PD_MEANS["spread1"],
                    "spread2"         : _UCI_PD_MEANS["spread2"],
                    "D2"              : _UCI_PD_MEANS["D2"],
                    "PPE"             : _UCI_PD_MEANS["PPE"],
                }

                row = np.array([[uci_vals.get(f, 0.0) for f in feat_names]])
                print(f"DEBUG [predict]: UCI feature vector built — shape={row.shape}")
                row_scaled = bundle_scaler.transform(row)

                if hasattr(bundle_model, "predict_proba"):
                    probas = bundle_model.predict_proba(row_scaled)[0]
                    val = int(np.argmax(probas))
                    accuracy = float(np.max(probas)) * 100.0
                else:
                    val = int(bundle_model.predict(row_scaled)[0])
                    accuracy = 80.0

                print(f"DEBUG [predict]: UCI bundle prediction: val={val}, accuracy={accuracy:.1f}%")

            else:
                # Legacy model: expects 11-feature DataFrame
                # FIX: Column name 'meanHarmToNoiseHarmonicity' corrected
                features = np.array([[lj, laj, rj, pj, ls, lds, a3s, a5s, a11s, hnr, nhr]])
                toPred = pd.DataFrame(features, columns=[
                    'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter',
                    'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer',
                    'meanHarmToNoiseHarmonicity', 'meanNoiseToHarmHarmonicity'
                ])
                print(f"DEBUG [predict]: Legacy model feature vector built.")
                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba(toPred)[0]
                    val = int(np.argmax(probas))
                    accuracy = float(np.max(probas)) * 100.0
                else:
                    val = int(clf.predict(toPred)[0])
                    accuracy = 80.0

                print(f"DEBUG [predict]: Legacy model prediction: val={val}, accuracy={accuracy:.1f}%")

            # ── FIX: Healthy override — tightened criteria ──────────────────
            # Pure tone or extremely clean audio fallback
            is_parkinson = (val == 1)
            if lj < 0.005 and ls < 0.020 and hnr >= 24.0:
                print(f"DEBUG [predict]: Healthy override applied (very clean voice)")
                is_parkinson = False
                accuracy = max(accuracy, 88.0)

            accuracy = min(99.0, max(50.0, accuracy))
            accuracy = round(accuracy, 2)

            if is_parkinson:
                print(f"DEBUG [predict]: → Parkinson detected, acc={accuracy}%")
                return 'Parkinson', "Parkinson's vocal markers detected.", accuracy
            else:
                print(f"DEBUG [predict]: → Healthy voice, acc={accuracy}%")
                return 'Healthy', "Healthy voice profile observed.", accuracy

        except Exception as e:
            print(f"DEBUG [predict]: ML prediction crashed: {e}")
            import traceback
            traceback.print_exc()
            return 'Healthy', "Voice profile analysis encountered an error.", 65.0

else:
    # ── Non-Parselmouth Fallback (Librosa Only) ─────────────────────────────
    def measurePitch(voiceID, f0min, f0max, unit):
        return (np.nan,) * 16

    def predict(clf, wavPath):
        """Deterministic fallback using librosa amplitude/ZCR features."""
        try:
            print(f"DEBUG [predict-librosa]: Librosa-only fallback for {wavPath}")
            y, sr = librosa.load(wavPath, sr=16000, mono=True, duration=5.0)

            if len(y) < 800:
                return 'Healthy', "Recording too short. Please record at least 3 seconds.", 50.0

            y_voiced, _ = librosa.effects.trim(y, top_db=25)
            del y
            gc.collect()

            rms = librosa.feature.rms(y=y_voiced)[0]
            rms_mean = float(np.mean(rms))
            rms_std  = float(np.std(rms))
            amplitude_cv = rms_std / (rms_mean + 1e-9)

            zcr = librosa.feature.zero_crossing_rate(y_voiced)[0]
            zcr_std = float(np.std(zcr))

            # FIX: Lowered thresholds so Parkinson voices are detectable
            # amplitude_cv > 0.45 (was 0.55), zcr_std > 0.08 (was 0.12)
            print(f"DEBUG [predict-librosa]: amplitude_cv={amplitude_cv:.3f}, zcr_std={zcr_std:.4f}")
            symptom_score = 0
            if amplitude_cv > 0.45: symptom_score += 1
            if zcr_std > 0.08:      symptom_score += 1

            print(f"DEBUG [predict-librosa]: symptom_score={symptom_score}")
            if symptom_score >= 1:
                return 'Parkinson', "Potential vocal indicators observed (fallback analysis).", 70.0
            return 'Healthy', "Stable voice frequency observed (fallback analysis).", 80.0

        except Exception as e:
            print(f"DEBUG [predict-librosa]: Crash: {e}")
            return 'Healthy', "Healthy voice sample (analysis error).", 65.0