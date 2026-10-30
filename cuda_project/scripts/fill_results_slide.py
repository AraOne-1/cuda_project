#!/usr/bin/env python3
"""Populate the measured-results slide (slide 12) using genuinely measured benchmark.csv rows.
Refuses missing/incomplete measurements. Requires python-pptx.
Run after benchmark.py, never with fabricated or placeholder data.
"""
import argparse
import csv
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/benchmark.csv')
    parser.add_argument('--pptx', default='docs/CUDA_Covertype_Presentation.pptx')
    parser.add_argument('--out', default='docs/CUDA_Covertype_Presentation_MEASURED.pptx')
    args = parser.parse_args()
    with open(args.csv, newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    data = {int(r['query_rows']): r for r in rows if r['version'] == 'gpu_optimized'}
    wanted = [32, 128, 512, 2048]
    if any(q not in data for q in wanted):
        raise SystemExit('Missing optimized GPU measurements for required Q=32,128,512,2048. Run benchmark first.')
    values = {}
    for q in wanted:
        r = data[q]
        try:
            nums = [float(r[c]) for c in ('cpu_ms','gpu_kernel_ms','gpu_e2e_ms','kernel_speedup','index_mismatches')]
        except (ValueError, KeyError) as ex:
            raise SystemExit(f'Unmeasured or invalid numerical values in Q={q}: {ex}')
        if not (all(x >= 0 and x < float('inf') for x in nums) and all(x > 0 for x in nums[:4])):
            raise SystemExit(f'Invalid/placeholder timing at Q={q}; refusing to fill slide.')
        values[q] = [f'{nums[0]:,.1f}', f'{nums[1]:,.1f}', f'{nums[2]:,.1f}',
                     f'{nums[3]:.2f}x', str(int(nums[4]))]
    ppt = Presentation(args.pptx)
    matches = [s for s in ppt.slides if any('Measured results' in sh.text for sh in s.shapes if sh.has_text_frame)]
    if len(matches) != 1:
        raise SystemExit('Expected one Measured results slide; presentation changed.')
    slide = matches[0]
    # The generated slide contains five placeholder dashes in each row.
    placeholders = [sh for sh in slide.shapes if sh.has_text_frame and sh.text.strip() == '—']
    if len(placeholders) != 20:
        raise SystemExit(f'Expected 20 placeholder cells; found {len(placeholders)}.')
    placeholders.sort(key=lambda sh: (round(sh.top / Inches(0.52)), sh.left))
    for i, q in enumerate(wanted):
        for j, value in enumerate(values[q]):
            placeholders[i*5+j].text_frame.paragraphs[0].text = value
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        if 'Results must come from your own NVIDIA CUDA run' in sh.text:
            sh.text_frame.paragraphs[0].text = 'ACTUAL NVIDIA GPU MEASUREMENTS (see results/benchmark.csv)'
        if 'GPU results: NOT MEASURED.' in sh.text:
            sh.text_frame.paragraphs[0].text = 'Measurements inserted from benchmark.csv; disclose GPU and CPU models in report.'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ppt.save(out)
    print('Created:', out)

if __name__ == '__main__':
    main()
