#!/usr/bin/env python3
"""Read-only source/CPU acceptance for C1-C4. No server/pretrained/robot use.

The argument output directory must be new. Each pytest group has a fresh process;
failures/timeouts/skips are recorded, never converted into passes. Synthetic
marker tests exercise the native visual interface, not pretrained DINO behavior.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Explicit repository path for local evidence replay')
    args = parser.parse_args()
    root = args.repo.resolve()
    out = args.output.resolve()
    if out.exists():
        parser.error('refusing to overwrite acceptance evidence')
    if not (root/'clearvla/mainline/model/policy.py').is_file():
        parser.error('not a ClearVLA source checkout')
    out.mkdir(parents=True)
    import torch
    def git(*a: str) -> str:
        return subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    groups = [
        ('C1_binding', ['tests/test_causal_binding_repair.py']),
        ('C2_gradient', ['tests/test_causal_gradient_readout.py']),
        ('C3_values', ['tests/test_causal_p2_value_repair.py']),
        ('C4_boundary', ['tests/test_causal_command_boundary.py']),
        ('integration_update', ['tests/test_causal_chain_integration.py::test_real_update_owns_and_updates_binder_and_p2_value_gain']),
        ('integration_lifecycle', ['tests/test_causal_chain_integration.py::test_complete_lifecycle_keeps_one_causal_encoding_and_world_rebuild']),
        ('integration_abi', ['tests/test_causal_chain_integration.py::test_abi_rejects_semantic_relabeling']),
        ('integration_checkpoint', ['tests/test_causal_chain_integration.py::test_new_checkpoint_exact_restore_and_deployment']),
        ('integration_neutral', ['tests/test_causal_chain_integration.py::test_zero_start_full_sampling_matches_world_value_mode']),
        ('native_fp32', ['tests/test_causal_native_interface.py::test_native_causal_chain_forward[False]']),
        ('native_bf16', ['tests/test_causal_native_interface.py::test_native_causal_chain_forward[True]']),
        ('native_adapter', ['tests/test_causal_native_interface.py::test_native_causal_chain_fresh_checkpoint_adapter']),
        ('existing_views', ['tests/test_dinov3_multiview.py']),
        ('existing_task', ['tests/test_mainline_task_execution.py']),
        ('existing_values', ['tests/test_mainline_task_execution_language_values.py']),
        ('existing_gripper', ['tests/test_calvin_selective_frame_weighting.py', 'tests/test_maniskill_binary_gripper.py']),
        ('existing_action', ['tests/test_mainline_action_field.py']),
        ('existing_encoder', ['tests/test_dinov3_online.py']),
    ]
    record = {'schema': 'causal-chain-C1-C4-acceptance-v1',
              'head': git('rev-parse', 'HEAD'), 'tree': git('rev-parse','HEAD^{tree}'),
              'working_tree_before': git('status', '--porcelain'),
              'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(),
              'supported_project_python_torch': sys.version_info[:2] == (3,12) and torch.__version__.split('+')[0].startswith('2.11.'),
              'scope': 'component/compact-model CPU updates, synthetic marker visual interface, checkpoints; no pretrained/GPU/robot qualification',
              'groups': []}
    parsed = 0
    for path in sorted((root/'clearvla/mainline').rglob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed += 1
    record['source_ast_files'] = parsed
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(root)+os.pathsep+str(root/'tests'), TERM='dumb')
    inventory = set()
    for name, nodes in groups:
        command = [sys.executable, '-m', 'pytest', *nodes, '-q', '--tb=short',
                   '--junitxml='+str(out/(name+'.xml'))]
        start = time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            try:
                completed = subprocess.run(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=240)
                rc = completed.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                log.write('\nACCEPTANCE: group timeout; NOT accepted.\n')
        group = {'name': name, 'command': command, 'returncode': rc,
                 'wall_seconds': time.perf_counter()-start}
        report = out/(name+'.xml')
        if report.exists():
            cases = ET.parse(report).getroot().findall('.//testcase')
            group.update(tests=len(cases), failures=sum(len(x.findall('failure')) for x in cases),
                         errors=sum(len(x.findall('error')) for x in cases),
                         skipped=sum(len(x.findall('skipped')) for x in cases))
            for case in cases:
                key = (case.get('classname'), case.get('name'))
                if key in inventory:
                    raise RuntimeError('duplicate selected test identity: '+str(key))
                inventory.add(key)
        else:
            group['incomplete_result'] = True
        record['groups'].append(group)
        (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
        print(name, rc, group.get('tests', 'incomplete'), flush=True)
    record['distinct_completed_cases'] = len(inventory)
    record['passed'] = sum(g.get('tests',0)-g.get('failures',0)-g.get('errors',0)-g.get('skipped',0) for g in record['groups'])
    record['failed_groups'] = [g['name'] for g in record['groups'] if g['returncode'] or g.get('failures') or g.get('errors') or g.get('incomplete_result')]
    record['skips'] = sum(g.get('skipped',0) for g in record['groups'])
    record['working_tree_after'] = git('status', '--porcelain')
    (out/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    return 1 if record['failed_groups'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
