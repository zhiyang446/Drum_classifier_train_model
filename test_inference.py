# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import torch
import pretty_midi
from train_phase2 import SymmetricDrumTCN

# Parameters
SR = 44100
HOP_LENGTH = 256
N_MELS = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transcribe_audio(audio_path, model_path, output_midi_path, tempo=None, time_sig=None, thresh_kick=0.50, thresh_snare=0.50, thresh_hihat=0.30):
    print(f"Loading audio file: {audio_path}")
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    n_frames = int(np.ceil(len(y) / HOP_LENGTH))
    
    # 1. Feature Extraction
    print("Extracting features...")
    # Channel 1: Log-Mel
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Channel 2: Superflux (linear domain diff -> log)
    diff_mel = np.diff(mel, axis=1)
    diff_mel = np.maximum(0, diff_mel)
    diff_mel = np.pad(diff_mel, ((0, 0), (1, 0)), mode='constant')
    log_diff_mel = np.log1p(diff_mel * 1000.0)
    
    # Independent Channel Z-Score Normalization
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_diff_mel = (log_diff_mel - log_diff_mel.mean()) / (log_diff_mel.std() + 1e-6)
    
    # Stack [1, 2, 256, Time]
    features = np.stack([log_mel, log_diff_mel], axis=0)
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    
    # 2. Load Model
    print(f"Loading model: {model_path}")
    model = SymmetricDrumTCN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Model Inference
    print("Running inference...")
    with torch.no_grad():
        onset_logits, vel_logits = model(features_tensor)
        onset_preds = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy() # [Time, 3]
        vel_preds = torch.sigmoid(vel_logits).squeeze(0).cpu().numpy() # [Time, 3]
        
    # 4. Post-processing (Peak Picking, Debounce, Parabolic Interpolation)
    print("Post-processing notes...")
    
    # 1.5 Auto-detect Tempo and Time Signature if not provided
    if tempo is None:
        try:
            # Estimate tempo using librosa
            tempo_est = librosa.feature.tempo(y=y, sr=SR, hop_length=HOP_LENGTH)[0]
            tempo = float(round(tempo_est))
            print(f"Auto-detected tempo: {tempo} BPM")
        except Exception:
            tempo = 120.0
            print("Failed to auto-detect tempo. Defaulting to 120.0 BPM")
            
    if time_sig is None:
        time_sig = '4/4'
        print("No time signature specified. Defaulting to 4/4")

    # Output midi tracks
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    try:
        num, denom = map(int, time_sig.split('/'))
        pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0.0))
    except Exception:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    pm.instruments.append(drum_instrument)
    
    # MIDI pitches: Kick=36, Snare=38, Hi-Hat=42
    inst_pitches = [36, 38, 42]
    inst_names = ['Kick', 'Snare', 'Hi-Hat']
    
    # Instrument-specific post-processing parameters:
    # Format: (threshold, peak_radius, min_dist_frames, valley_coef)
    inst_params = {
        'Kick': (thresh_kick, 2, 6, 0.60),
        'Snare': (thresh_snare, 2, 6, 0.60),
        'Hi-Hat': (thresh_hihat, 2, 6, 0.60)
    }
    
    for inst_idx in range(3):
        prob = onset_preds[:, inst_idx]
        vel = vel_preds[:, inst_idx]
        pitch = inst_pitches[inst_idx]
        name = inst_names[inst_idx]
        
        threshold, peak_radius, min_dist, valley_coef = inst_params[name]
        
        last_trigger_frame = -999
        note_count = 0
        
        for t in range(peak_radius, len(prob) - peak_radius):
            if prob[t] > threshold:
                # Local maximum peak check
                is_peak = True
                for r in range(1, peak_radius + 1):
                    if prob[t] <= prob[t-r] or prob[t] <= prob[t+r]:
                        is_peak = False
                        break
                if not is_peak:
                    continue
                    
                # Debounce check
                if t - last_trigger_frame >= min_dist:
                    # Valley check (ensure there is a distinct dip between current and last hit)
                    if last_trigger_frame != -999:
                        valley_val = np.min(prob[last_trigger_frame:t])
                        # If valley is too high (above valley_coef of current peak), it is likely a flam or rattle
                        if valley_coef is not None and valley_val > valley_coef * prob[t]:
                            continue
                                
                    # Sub-frame Parabolic Interpolation with Vertex Validity Guard
                    denom = prob[t-1] - 2 * prob[t] + prob[t+1]
                    if abs(denom) > 1e-5:
                        dt = (prob[t-1] - prob[t+1]) / (2 * denom)
                        # Clamp dt to [-0.5, 0.5]
                        dt = np.clip(dt, -0.5, 0.5)
                    else:
                        dt = 0.0
                        
                    # Precise timestamp
                    onset_time = (t + dt) * HOP_LENGTH / SR
                    
                    # 1D Max Pooling for Velocity in [t-2, t+2]
                    vel_pool = vel[max(0, t-2):min(len(vel), t+3)]
                    peak_vel = np.max(vel_pool)
                    velocity_midi = int(np.clip(peak_vel * 127.0, 1, 127))
                    
                    # Add note to midi
                    note = pretty_midi.Note(
                        velocity=velocity_midi,
                        pitch=pitch,
                        start=onset_time,
                        end=onset_time + 0.1 # fixed 100ms duration for midi events
                    )
                    drum_instrument.notes.append(note)
                    
                    last_trigger_frame = t
                    note_count += 1
                        
        print(f"  -> Detected {note_count} {inst_names[inst_idx]} notes.")
        
    # Sort notes by start time
    drum_instrument.notes.sort(key=lambda x: x.start)
    
    # Scale/Normalize velocities to standard musical range [40, 110] if max velocity is low
    if len(drum_instrument.notes) > 0:
        vels = [n.velocity for n in drum_instrument.notes]
        max_v = max(vels)
        min_v = min(vels)
        if max_v < 90:
            target_min = 40
            target_max = 110
            for n in drum_instrument.notes:
                if max_v > min_v:
                    scaled = target_min + (n.velocity - min_v) * (target_max - target_min) / (max_v - min_v)
                    n.velocity = int(round(scaled))
                else:
                    n.velocity = 80
            print(f"Normalized MIDI velocities from range [{min_v}, {max_v}] to range [{target_min}, {target_max}].")
            
    # Save MIDI
    pm.write(output_midi_path)
    print(f"Successfully exported MIDI file to: {output_midi_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test inference for Phase 3 Model")
    parser.add_argument('--input', type=str, default='test.wav', help="Input WAV drum track path")
    parser.add_argument('--output', type=str, default=None, help="Output MIDI file path")
    parser.add_argument('--model', type=str, default='best_drum_model.pth', help="Model weight path")
    parser.add_argument('--tempo', type=float, default=None, help="Tempo of the track in BPM. If None, auto-detected.")
    parser.add_argument('--time-signature', type=str, default=None, help="Time signature of the track (e.g. 5/8, 4/4). If None, defaults to 4/4.")
    parser.add_argument('--thresh-kick', type=float, default=0.50, help="Confidence threshold for Kick drum (default: 0.50)")
    parser.add_argument('--thresh-snare', type=float, default=0.50, help="Confidence threshold for Snare drum (default: 0.50)")
    parser.add_argument('--thresh-hihat', type=float, default=0.30, help="Confidence threshold for Hi-Hat (default: 0.30)")
    args = parser.parse_args()
    
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_drums.mid"
    else:
        output_path = args.output
        
    transcribe_audio(
        args.input, 
        args.model, 
        output_path, 
        tempo=args.tempo, 
        time_sig=args.time_signature,
        thresh_kick=args.thresh_kick,
        thresh_snare=args.thresh_snare,
        thresh_hihat=args.thresh_hihat
    )

