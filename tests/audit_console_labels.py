"""Read-only label/source audit; writes a separate audit summary, never regenerates labels."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bmode_opt'))
import hisense_backend_sim as S
from hisense_loader import load_capture
import scene_family as SF


def counter(items):
    return dict(collections.Counter(items))


def summary(values):
    a = np.asarray(values, float)
    return dict(zip(['min', 'p10', 'median', 'p90', 'max'],
                    np.percentile(a, [0, 10, 50, 90, 100]).tolist())) if a.size else {}


def main():
    source = ROOT / 'data/labels_console.jsonl'
    rows, invalid = [], []
    for n, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except Exception as e:
            invalid.append([n, str(e)])
    captures = collections.defaultdict(list)
    for path in (ROOT / 'data/hisense_medical').rglob('Algo_BC0.bin'):
        captures[path.parent.name].append(path.parent)
    errors = collections.defaultdict(list)
    paths = {}
    for r in rows:
        name = r['frame_id']
        session = r['group_id'].rsplit('/', 1)[0]
        p = ROOT / 'data/hisense_medical' / session / name
        paths[name] = p
        if not (p / 'Algo_BC0.bin').exists():
            errors['missing_source'].append(name)
            continue
        c = load_capture(p)
        setting = SF.front_setting(c)
        checks = {'depth': np.isclose(r['depth_mm'], setting[1]),
                  'frequency': np.isclose(r['frequency_mhz'], setting[2]),
                  'focus': np.isclose(r['focus_mm'], setting[3]),
                  'mode': r['imaging_mode'] == ('harmonic' if setting[0] else 'fundamental'),
                  'gain': np.isclose(r['gain_db'], S.capture_gain_db(c)),
                  'tgc': np.array_equal(r['tgc_levels'], c.tgc_levels),
                  'dr': r['dr_ui'] == c.dynamic_range_level,
                  'gain_slope': np.isclose(r['gain_db_per_level'], S.gain_db_per_level(setting[0])),
                  'tgc_slope': np.isclose(r['tgc_db_per_level'], S.tgc_db_per_level(setting[0]))}
        for k, good in checks.items():
            if not good:
                errors['source_' + k].append(name)
        gd = r['optimal_gain_db'] - r['gain_db']
        td = np.array(r['optimal_tgc_levels']) - r['tgc_levels']
        if not (np.isclose(gd, r['delta_gain_db']) and
                np.isclose(gd, r['delta_gain_levels'] * r['gain_db_per_level'])):
            errors['gain_arithmetic'].append(name)
        if not (np.allclose(td, r['delta_tgc_levels']) and
                np.allclose(td * r['tgc_db_per_level'], r['delta_tgc_db'])):
            errors['tgc_arithmetic'].append(name)
        if not np.isclose(r['optimal_dr_ui'] - r['dr_ui'], r['delta_dr_ui']):
            errors['dr_arithmetic'].append(name)
        if any(not 0 <= x <= 255 for x in r['optimal_tgc_levels']):
            errors['tgc_bounds'].append(name)
        implied_gain = c.gain_level + r['delta_gain_levels']
        if not 0 <= implied_gain <= 255:
            errors['implied_gain_bounds'].append([name, implied_gain])
        for stem, names in [('depth', ('shallow', 'correct', 'deep')),
                            ('frequency', ('low', 'correct', 'high')),
                            ('focus', ('shallow', 'correct', 'deep'))]:
            field = stem + ('_mhz' if stem == 'frequency' else '_mm')
            if r[stem + '_determined']:
                delta = r['optimal_' + field] - r[field]
                ladder = r[stem + '_ladder']
                if r['optimal_' + field] not in ladder or r[field] not in ladder:
                    errors[stem + '_ladder'].append(name)
                    continue
                steps = ladder.index(r['optimal_' + field]) - ladder.index(r[field])
                direction = names[1] if steps == 0 else names[0 if steps > 0 else 2]
                if not (np.isclose(delta, r['delta_' + field]) and
                        steps == r['delta_' + stem + '_steps'] and direction == r[stem + '_direction']):
                    errors[stem + '_arithmetic_direction'].append(name)
            elif any(r[k] is not None for k in ['optimal_' + field, stem + '_direction',
                                                'delta_' + field, 'delta_' + stem + '_steps']):
                errors[stem + '_undetermined_filled'].append(name)
    axes = {}
    for stem, other in [('depth', ['frequency_mhz', 'focus_mm']),
                        ('frequency', ['depth_mm', 'focus_mm']),
                        ('focus', ['depth_mm', 'frequency_mhz'])]:
        rs = [r for r in rows if r[stem + '_determined']]
        keys = {(r['family_id'], r['imaging_mode'], *(r[k] for k in other)) for r in rs}
        strict = [r for r in rs if not any(r[k] for k in
                  [stem + '_at_edge', 'family_unbracketed', 'family_anchor_starved'])]
        field = stem + ('_mhz' if stem == 'frequency' else '_mm')
        unsupported = []
        for r in rs:
            comparable = [q for q in rows if q['family_id'] == r['family_id'] and
                          q['imaging_mode'] == r['imaging_mode'] and
                          all(q[k] == r[k] for k in other)]
            if r['optimal_' + field] not in {q[field] for q in comparable}:
                unsupported.append(r['frame_id'])
        axes[stem] = {'determined': len(rs), 'comparison_sets': len(keys),
                      'directions': counter(r[stem + '_direction'] for r in rs),
                      'edge': sum(r[stem + '_at_edge'] for r in rs),
                      'strict_family_nonedge_rows': len(strict),
                      'recommended_value_not_observed_in_comparison_set': unsupported,
                      'optimal_values': counter(str(r['optimal_' + field]) for r in rs)}
    all_ids = set(captures)
    ids = [r['frame_id'] for r in rows]
    sessions = sorted({r['group_id'].rsplit('/', 1)[0] for r in rows})
    report = {'label_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'rows': len(rows), 'invalid_json': invalid,
              'duplicate_label_ids': {k: v for k, v in counter(ids).items() if v > 1},
              'raw_capture_count': sum(map(len, captures.values())),
              'duplicate_raw_ids': {k: [str(p.relative_to(ROOT)) for p in v]
                                    for k, v in captures.items() if len(v) > 1},
              'unlabeled_captures': [str(p.relative_to(ROOT)) for k in sorted(all_ids - set(ids))
                                     for p in captures[k]],
              'modes': counter(r['imaging_mode'] for r in rows),
              'split': counter(str(r['split']) for r in rows),
              'groups': len({r['group_id'] for r in rows}),
              'families': len({r['family_id'] for r in rows}),
              'family_mode_pairs': len({(r['family_id'], r['imaging_mode']) for r in rows}),
              'session_count': len(sessions), 'errors': dict(errors), 'axes': axes,
              'flags': {k: sum(r[k] for r in rows) for k in
                        ['family_unbracketed', 'family_anchor_starved', 'calibration_borrowed',
                         'at_gain_edge', 'dr_determined']},
              'notes': counter(n for r in rows for n in r['notes']),
              'gain_directions': counter(r['gain_direction'] for r in rows),
              'slider_directions': {k: counter(r['slider_directions'][k] for r in rows)
                                    for k in ['near', 'mid', 'far']},
              'dr_directions': counter(r['dr_direction'] for r in rows),
              'distributions': {k: summary([r[k] for r in rows]) for k in
                                ['delta_gain_db', 'delta_gain_levels', 'objective', 'tolerance',
                                 'label_uncertainty', 'deadband_gain_levels', 'equivalent_count']},
              'nonzero_gain_delta_marked_correct': sum(r['gain_direction'] == 'correct' and
                            abs(r['delta_gain_levels']) > 1e-6 for r in rows),
              'optimal_tgc_boundary_rows': sum(any(x in (0, 255) for x in r['optimal_tgc_levels'])
                                               for r in rows),
              'all_front_determined': sum(all(r[k + '_determined'] for k in axes) for r in rows),
              'per_session': {s: {'rows': len(rs), **{k: sum(r[k + '_determined'] for r in rs)
                                                     for k in axes}}
                              for s in sessions for rs in
                              [[r for r in rows if r['group_id'].rsplit('/', 1)[0] == s]]}}
    out = ROOT / 'docs/console_label_audit_20260913.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
