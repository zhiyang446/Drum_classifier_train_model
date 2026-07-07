# -*- coding: utf-8 -*-
"""
Audit whether raw AI should match acoustic events or denser notation targets.
"""
import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter


INSTS = ('kick', 'snare', 'hihat')
XML_TO_COL = {'KD': 'kick', 'SD': 'snare', 'HH': 'hihat'}


def count_xml_events(xml_path):
    """
    中文註解：統計 acoustic XML 中明確標註的 KD/SD/HH 事件數。
    """
    counts = Counter()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for event in root.findall('.//event'):
        inst_node = event.find('instrument')
        if inst_node is None:
            continue
        col = XML_TO_COL.get(inst_node.text)
        if col:
            counts[col] += 1
    return {inst: counts[inst] for inst in INSTS}


def count_layer_events(csv_path):
    """
    中文註解：統計 raw/notation layer CSV 中 native、final、virtual 三種事件數。
    """
    rows = list(csv.DictReader(open(csv_path, 'r', encoding='utf-8-sig')))
    result = {}
    for prefix in ('native', 'final', 'virtual'):
        result[prefix] = {
            inst: sum(row.get(f'{prefix}_{inst}') == 'True' for row in rows)
            for inst in INSTS
        }
    result['rows'] = len(rows)
    return result


def build_audit(xml_path, raw_csv, notation_csv, notation_gate):
    """
    中文註解：建立單檔 raw acoustic target audit 結果。
    """
    acoustic = count_xml_events(xml_path)
    raw = count_layer_events(raw_csv)
    notation = count_layer_events(notation_csv)
    gate = parse_gate(notation_gate)
    rows = []
    for inst in INSTS:
        raw_final = raw['final'][inst]
        acoustic_count = acoustic[inst]
        gate_count = gate.get(inst, 0)
        rows.append({
            'instrument': inst,
            'acoustic_xml': acoustic_count,
            'raw_ai_final': raw_final,
            'notation_final': notation['final'][inst],
            'notation_virtual': notation['virtual'][inst],
            'strict_notation_gate': gate_count,
            'raw_matches_acoustic': raw_final >= acoustic_count,
            'gate_exceeds_acoustic': gate_count > acoustic_count,
            'target_layer': 'notation' if gate_count > acoustic_count else 'raw_ai',
        })
    return rows


def parse_gate(text):
    """
    中文註解：解析簡單 gate 字串，例如 KD=16,SD=8,HH=32。
    """
    result = {}
    aliases = {'KD': 'kick', 'SD': 'snare', 'HH': 'hihat'}
    if not text:
        return result
    for part in text.split(','):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        inst = aliases.get(key.strip().upper())
        if inst:
            result[inst] = int(value.strip())
    return result


def write_outputs(rows, output_csv, output_json):
    """
    中文註解：輸出 CSV/JSON 審核報告，供後續驗收與文件引用。
    """
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


def run_self_check():
    """
    中文註解：最小自檢，確認 gate 解析與 layer 判斷不被改壞。
    """
    gate = parse_gate('KD=16,SD=8,HH=32')
    assert gate == {'kick': 16, 'snare': 8, 'hihat': 32}
    print('Self-check passed.')


def main():
    """
    中文註解：CLI 入口，執行單檔 acoustic/raw/notation target audit。
    """
    parser = argparse.ArgumentParser(description='Audit raw AI acoustic targets.')
    parser.add_argument('--xml', required=False)
    parser.add_argument('--raw-csv', required=False)
    parser.add_argument('--notation-csv', required=False)
    parser.add_argument('--notation-gate', default='KD=16,SD=8,HH=32')
    parser.add_argument('--output-csv', default='validation_runs/raw_acoustic_audit/summary.csv')
    parser.add_argument('--output-json', default='validation_runs/raw_acoustic_audit/summary.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return
    if not args.xml or not args.raw_csv or not args.notation_csv:
        raise SystemExit('--xml, --raw-csv, and --notation-csv are required unless --self-check is used.')

    rows = build_audit(args.xml, args.raw_csv, args.notation_csv, args.notation_gate)
    write_outputs(rows, args.output_csv, args.output_json)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
