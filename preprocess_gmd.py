# -*- coding: utf-8 -*-
import os
import csv
import json
import pretty_midi

# Parameters
GMD_DIR = 'gmd_dataset'
INFO_CSV = os.path.join(GMD_DIR, 'info.csv')
OUTPUT_DIR = 'processed_data'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'gmd_meta.json')

# MIDI Pitch Mapping to target classes: KD, SD, HH
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
            # GMD tracks should be marked as drum tracks, but we parse all notes just in case
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
                
                # Append event: (onset_sec, instrument_name, velocity)
                events.append({
                    'time': float(note.start),
                    'inst': inst,
                    'velocity': float(note.velocity)
                })
        # Sort events by time
        events.sort(key=lambda x: x['time'])
    except Exception as e:
        print(f"Error parsing MIDI {midi_path}: {e}")
    return events

def main():
    if not os.path.exists(INFO_CSV):
        print(f"Error: Could not find {INFO_CSV}! Please check the folder location.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Parsing info.csv and extracting GMD MIDI annotations...")
    
    gmd_meta = {}
    total_notes_count = 0
    
    with open(INFO_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Found {len(rows)} entries in info.csv.")
    
    for idx, row in enumerate(rows):
        audio_rel = row['audio_filename']
        midi_rel = row['midi_filename']
        
        # Use absolute paths so soundfile works correctly on Windows
        audio_path = os.path.abspath(os.path.join(GMD_DIR, audio_rel))
        midi_path = os.path.abspath(os.path.join(GMD_DIR, midi_rel))
        
        if not os.path.isfile(audio_path):
            continue
            
        if not os.path.isfile(midi_path):
            continue
            
        # Parse MIDI
        events = parse_midi_file(midi_path)
        if not events:
            # Skip files with no target notes
            continue
            
        # Unique prefix/key for the song
        song_key = f"gmd_{row['drummer']}_{row['session']}_{row['id']}"
        
        gmd_meta[song_key] = {
            'audio_path': audio_path,
            'duration': float(row['duration']),
            'bpm': float(row['bpm']) if row['bpm'] else 120.0,
            'split': row['split'],
            'beat_type': row['beat_type'],
            'events': events
        }
        
        total_notes_count += len(events)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(rows):
            print(f"Processed {idx + 1}/{len(rows)} songs...")

    print(f"Successfully processed {len(gmd_meta)} valid songs.")
    print(f"Total extracted drum notes: {total_notes_count}")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as out_f:
        json.dump(gmd_meta, out_f, indent=2)
        
    print(f"Saved GMD metadata to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
