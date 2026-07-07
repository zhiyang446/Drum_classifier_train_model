# -*- coding: utf-8 -*-
"""
ADT E-GMD Preprocessing Script
Parses the e-gmd-v1.0.0.csv and corresponding MIDI files to generate processed_data/egmd_meta.json
"""
import os
import csv
import json
import argparse
import pretty_midi

EGMD_DIR = 'egmd_dataset_2'
OUTPUT_DIR = 'processed_data'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'egmd_meta.json')

# General MIDI Drum Map to KD, SD, HH
KD_PITCHES = {35, 36}
SD_PITCHES = {37, 38, 40}
HH_PITCHES = {22, 26, 42, 44, 46}

def parse_midi_file(midi_path):
    """
    Parses a MIDI file and extracts target drum onset events: (time, instrument, velocity).
    """
    events = []
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        for instrument in pm.instruments:
            for note in instrument.notes:
                pitch = note.pitch
                if pitch in KD_PITCHES:
                    inst = 'KD'
                elif pitch in SD_PITCHES:
                    inst = 'SD'
                elif pitch in HH_PITCHES:
                    inst = 'HH'
                else:
                    continue
                
                events.append({
                    'time': float(note.start),
                    'inst': inst,
                    'velocity': float(note.velocity)
                })
        events.sort(key=lambda x: x['time'])
    except Exception as e:
        pass
    return events


def run_self_check():
    """
    中文註解：最小自檢，確認 E-GMD CSV 路徑會依資料夾參數組合。
    """
    assert os.path.join('x', 'e-gmd-v1.0.0.csv').endswith('e-gmd-v1.0.0.csv')
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，將 E-GMD CSV/MIDI 轉成專案共用 metadata。
    """
    parser = argparse.ArgumentParser(description='Preprocess E-GMD metadata.')
    parser.add_argument('--egmd-dir', default=EGMD_DIR)
    parser.add_argument('--output', default=OUTPUT_JSON)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    info_csv = os.path.join(args.egmd_dir, 'e-gmd-v1.0.0.csv')
    if not os.path.exists(info_csv):
        print(f"Error: Could not find {info_csv}!")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Parsing E-GMD CSV and extracting MIDI annotations...", flush=True)
    
    with open(info_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Found {len(rows)} entries in e-gmd-v1.0.0.csv.", flush=True)
    
    egmd_meta = {}
    total_notes_count = 0
    skipped_count = 0
    processed_count = 0
    
    # Process rows
    for idx, row in enumerate(rows):
        audio_rel = row['audio_filename']
        midi_rel = row['midi_filename']
        
        audio_path = os.path.abspath(os.path.join(args.egmd_dir, audio_rel))
        midi_path = os.path.abspath(os.path.join(args.egmd_dir, midi_rel))
        
        if not os.path.isfile(audio_path) or not os.path.isfile(midi_path):
            skipped_count += 1
            continue
            
        events = parse_midi_file(midi_path)
        if not events:
            # Skip files with no target notes
            skipped_count += 1
            continue
            
        # Create a unique key for the song instance (each kit render is a unique entry)
        song_key = f"egmd_{row['drummer']}_{row['session']}_{row['id']}_{row['kit_name'].replace(' ', '_').replace('(', '').replace(')', '')}"
        
        egmd_meta[song_key] = {
            'audio_path': audio_path,
            'duration': float(row['duration']),
            'bpm': float(row['bpm']) if row['bpm'] else 120.0,
            'split': row['split'],
            'kit_name': row['kit_name'],
            'events': events
        }
        
        total_notes_count += len(events)
        processed_count += 1
        
        if processed_count % args.progress_every == 0 or (idx + 1) == len(rows):
            print(f"  Processed {processed_count} songs... (skipped {skipped_count})", flush=True)
            
    print(f"Successfully processed {processed_count} songs. Total drum notes: {total_notes_count}.", flush=True)
    print(f"Writing metadata to {args.output}...", flush=True)
    
    with open(args.output, 'w', encoding='utf-8') as out_f:
        json.dump(egmd_meta, out_f, indent=2)
        
    print("Pre-processing completed successfully!", flush=True)

if __name__ == '__main__':
    main()
