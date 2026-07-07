# -*- coding: utf-8 -*-
"""
Build a readiness manifest for mixed E-GMD / STAR / IDMT-style training.

This script does not train. It only verifies which dataset manifests are
available and creates a local XML manifest from audio/annotation_xml.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter


INST_SET = {'KD', 'SD', 'HH'}


def parse_local_xml(xml_path):
    """
    中文註解：解析本地 XML 標註，轉成與 STAR/E-GMD 相同的 events 結構。
    """
    events = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for event in root.findall('.//event'):
        inst_node = event.find('instrument')
        onset_node = event.find('onsetSec')
        if inst_node is None or onset_node is None:
            continue
        inst = inst_node.text
        if inst not in INST_SET:
            continue
        events.append({
            'time': float(onset_node.text),
            'inst': inst,
            'velocity': 100.0,
        })
    events.sort(key=lambda item: item['time'])
    return events


def build_local_xml_meta(audio_dir, xml_dir):
    """
    中文註解：從本地 MIX wav 與 XML 建立 clean/local anchor metadata。
    """
    meta = {}
    if not os.path.isdir(audio_dir) or not os.path.isdir(xml_dir):
        return meta

    for name in sorted(os.listdir(audio_dir)):
        if not name.endswith('#MIX.wav'):
            continue
        prefix = name.split('#')[0]
        audio_path = os.path.abspath(os.path.join(audio_dir, name))
        xml_path = os.path.join(xml_dir, f'{prefix}#MIX.xml')
        if not os.path.isfile(xml_path):
            continue
        events = parse_local_xml(xml_path)
        if not events:
            continue
        meta[f'local_{prefix}'] = {
            'audio_path': audio_path,
            'annotation_path': os.path.abspath(xml_path),
            'duration': max(ev['time'] for ev in events),
            'bpm': 120.0,
            'split': 'train',
            'kit_name': 'local_xml',
            'events': events,
        }
    return meta


def load_json_if_exists(path):
    """
    中文註解：安全讀取 JSON metadata，不存在時回傳 None。
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize_meta(meta):
    """
    中文註解：統計 metadata 的樣本數與 KD/SD/HH event 數量。
    """
    counts = Counter()
    if not meta:
        return {'items': 0, 'events': {'KD': 0, 'SD': 0, 'HH': 0}}
    for item in meta.values():
        for event in item.get('events', []):
            if event.get('inst') in INST_SET:
                counts[event['inst']] += 1
    return {
        'items': len(meta),
        'events': {inst: counts[inst] for inst in ('KD', 'SD', 'HH')},
    }


def dataset_entry(name, path, meta, required):
    """
    中文註解：建立單一資料集 manifest entry，包含可用性與統計。
    """
    return {
        'name': name,
        'path': path,
        'required': required,
        'status': 'available' if meta else 'missing',
        **summarize_meta(meta),
    }


def run_self_check():
    """
    中文註解：最小自檢，確保統計與 ready 判斷可用。
    """
    fake = {
        'x': {'events': [
            {'inst': 'KD', 'time': 0.0},
            {'inst': 'SD', 'time': 0.5},
            {'inst': 'HH', 'time': 1.0},
        ]}
    }
    summary = summarize_meta(fake)
    assert summary['items'] == 1
    assert summary['events']['SD'] == 1
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，建立 mixed training readiness manifest。
    """
    parser = argparse.ArgumentParser(description='Build mixed dataset readiness manifest.')
    parser.add_argument('--star-meta', default='processed_data/star_meta.json')
    parser.add_argument('--egmd-meta', default='processed_data/egmd_meta.json')
    parser.add_argument('--idmt-meta', default='processed_data/idmt_meta.json')
    parser.add_argument('--audio-dir', default='audio')
    parser.add_argument('--xml-dir', default='annotation_xml')
    parser.add_argument('--local-output', default='processed_data/local_xml_meta.json')
    parser.add_argument('--output', default='processed_data/mixed_manifest.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    local_meta = build_local_xml_meta(args.audio_dir, args.xml_dir)
    with open(args.local_output, 'w', encoding='utf-8') as f:
        json.dump(local_meta, f, indent=2, ensure_ascii=False)

    star_meta = load_json_if_exists(args.star_meta)
    egmd_meta = load_json_if_exists(args.egmd_meta)
    idmt_meta = load_json_if_exists(args.idmt_meta)

    datasets = [
        dataset_entry('egmd', args.egmd_meta, egmd_meta, required=True),
        dataset_entry('star', args.star_meta, star_meta, required=True),
        dataset_entry('idmt', args.idmt_meta, idmt_meta, required=True),
        dataset_entry('local_xml', args.local_output, local_meta, required=False),
    ]
    missing_required = [item['name'] for item in datasets if item['required'] and item['status'] != 'available']
    manifest = {
        'ready_for_mixed_training': not missing_required,
        'missing_required': missing_required,
        'recommended_batch_ratio': {'egmd': 0.50, 'star': 0.30, 'idmt': 0.20},
        'fallback_available': {'local_xml_can_substitute_clean_anchor': bool(local_meta)},
        'datasets': datasets,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'Wrote local XML meta: {args.local_output} ({len(local_meta)} items)')
    print(f'Wrote mixed manifest: {args.output}')
    print(f'Ready for mixed training: {manifest["ready_for_mixed_training"]}')
    if missing_required:
        print(f'Missing required datasets: {", ".join(missing_required)}')


if __name__ == '__main__':
    main()
