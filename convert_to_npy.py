# -*- coding: utf-8 -*-
import os
import glob
import json
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import soundfile as sf

SR = 44100
OUTPUT_DIR = 'processed_data'
NPY_DIR = os.path.join(OUTPUT_DIR, 'npy')
os.makedirs(NPY_DIR, exist_ok=True)

inst_indices = {'KD': 0, 'SD': 1, 'HH': 2}

def convert_wav_to_npy(wav_path, npy_path):
    """
    Reads a WAV file, converts it to mono float32 at SR=44100, and saves as .npy.
    """
    if os.path.exists(npy_path):
        return True
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        np.save(npy_path, y.astype(np.float32))
        return True
    except Exception as e:
        print(f"Error converting {wav_path}: {e}")
        return False

def main():
    # 1. Process GMD Dataset
    gmd_meta_json = os.path.join(OUTPUT_DIR, 'gmd_meta.json')
    if os.path.exists(gmd_meta_json):
        print(f"Loading GMD metadata from {gmd_meta_json}...")
        with open(gmd_meta_json, 'r', encoding='utf-8') as f:
            gmd_meta = json.load(f)
            
        print(f"Converting GMD audio tracks to .npy in {NPY_DIR}...")
        count = 0
        for song_key, info in gmd_meta.items():
            audio_path = info['audio_path']
            safe_song_key = song_key.replace('/', '_').replace('\\', '_')
            npy_path = os.path.join(NPY_DIR, f"{safe_song_key}_wave.npy")
            if convert_wav_to_npy(audio_path, npy_path):
                info['wave_npy_path'] = os.path.abspath(npy_path)
                count += 1
            if count % 100 == 0:
                print(f"  Converted {count}/{len(gmd_meta)} GMD songs...")
                
        # Save updated GMD metadata
        with open(gmd_meta_json, 'w', encoding='utf-8') as out_f:
            json.dump(gmd_meta, out_f, indent=2)
        print(f"Successfully updated GMD metadata with .npy paths.")
    else:
        print("GMD metadata json not found. Skipping GMD.")

    # 2. Process IDMT Dataset
    print("Processing IDMT dataset...")
    mix_files = glob.glob(os.path.join('audio', '*#MIX.wav'))
    song_prefixes = sorted(list(set([os.path.basename(f).split('#')[0] for f in mix_files])))
    
    idmt_meta = {}
    print(f"Found {len(song_prefixes)} IDMT songs. Converting WAVs and parsing XMLs...")
    
    for idx, prefix in enumerate(song_prefixes):
        xml_path = os.path.join('annotation_xml', f"{prefix}#MIX.xml")
        mix_path = os.path.join('audio', f"{prefix}#MIX.wav")
        
        # Parse XML events
        events = []
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for event in root.findall('.//event'):
                    inst = event.find('instrument').text
                    if inst in inst_indices:
                        onset_sec = float(event.find('onsetSec').text)
                        events.append({'time': onset_sec, 'inst': inst})
            except Exception as e:
                print(f"Error parsing XML for {prefix}: {e}")
                
        # Convert MIX wave
        mix_npy_path = os.path.join(NPY_DIR, f"idmt_{prefix}_MIX_wave.npy")
        convert_wav_to_npy(mix_path, mix_npy_path)
        
        # Convert Clean tracks (KD, SD, HH)
        clean_npy_paths = {}
        for inst in inst_indices.keys():
            clean_path = os.path.join('audio', f"{prefix}#{inst}#train.wav")
            clean_npy_path = os.path.join(NPY_DIR, f"idmt_{prefix}_{inst}_wave.npy")
            if os.path.exists(clean_path):
                convert_wav_to_npy(clean_path, clean_npy_path)
                clean_npy_paths[inst] = os.path.abspath(clean_npy_path)
            else:
                # Fallback to MIX
                clean_npy_paths[inst] = os.path.abspath(mix_npy_path)
                
        idmt_meta[prefix] = {
            'wave_npy_path': os.path.abspath(mix_npy_path),
            'clean_tracks_npy_paths': clean_npy_paths,
            'events': events
        }
        if (idx + 1) % 20 == 0 or (idx + 1) == len(song_prefixes):
            print(f"  Processed {idx + 1}/{len(song_prefixes)} IDMT songs...")
            
    idmt_meta_json = os.path.join(OUTPUT_DIR, 'idmt_meta.json')
    with open(idmt_meta_json, 'w', encoding='utf-8') as out_f:
        json.dump(idmt_meta, out_f, indent=2)
    print(f"Saved IDMT metadata to {idmt_meta_json}")

if __name__ == '__main__':
    main()
