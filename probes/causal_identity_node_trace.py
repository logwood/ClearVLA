"""Observe actual outputs along the A/B graph, without recomputing readers."""
import numpy as np
import torch


def rms(value):
    return float(np.sqrt(np.square(np.asarray(value, dtype=np.float64)).mean()))


class NodeTrace:
    def __init__(self, model):
        self.saved = []
        self.events = {}
        self.active = False

        def around(owner, name, observe):
            old = getattr(owner, name)
            self.saved.append((owner, name, old))

            def wrapped(*args, **kwargs):
                out = old(*args, **kwargs)
                if self.active:
                    observe(out, args, kwargs)
                return out
            setattr(owner, name, wrapped)

        def fields(prefix, obj, names):
            for name in names:
                self.record(prefix + name, getattr(obj, name, None))

        around(model.p1, 'build_static', lambda out, a, kw: fields('p1_', out[0], ('protected_detail',)))
        around(model.p1, 'update_dynamic', lambda out, a, kw: fields('p1_', out[0], ('factual_base', 'policy_query_residual')))
        effect = model.policy_compiler.effect_reader
        around(effect, 'spatial_select', lambda out, a, kw: fields('p2_selected_', out[0], (
            'semantic_value', 'geometry_value', 'selected_s_context', 'selected_target_value',
        )))
        around(effect, 'temporal_terminal', lambda out, a, kw: fields('p2_effect_', out[0], ('semantic', 'geometry')))
        around(model.policy_compiler.consequence, 'forward', lambda out, a, kw: fields('consequence_', out[0], ('factual_base', 'protected_consequence')))
        coordinator = model.policy_compiler.plan_compiler.coordinator

        def coordinated(out, args, kwargs):
            inputs = args[0] if args else kwargs['context']
            fields('p3_input_', inputs, ('action', 'current_fact', 'policy_precision', 'semantic_effect', 'geometry_feature_effect', 'task_temporal', 'observed_change'))
            for name, value in zip(('temporal', 'state_change', 'private'), out[:3]):
                self.record('p3_' + name, value)
        around(coordinator, 'forward', coordinated)

        def bottom(out, args, kwargs):
            bank = args[1] if len(args) > 1 else kwargs['bank']
            fields('bottom_bank_', bank, ('values', 'protected_detail', 'protected_policy_precision'))
            self.record('bottom_optional_update', out[0])
            self.record('bottom_precision_update', out[1])
        around(model.execution_bottom.decoder, '_read_policy_delta_bank', bottom)

    def record(self, name, value):
        if isinstance(value, torch.Tensor):
            self.events.setdefault(name, []).append(value.detach().float().cpu().numpy().copy())

    def reset(self):
        self.events = {}
        self.active = True

    def finish(self, intent):
        for name in ('public_interval_carrier', 'typed_relevance_value', 'typed_common_value', 'typed_interval_residual_value', 'policy_interval_context'):
            self.record('s_' + name, getattr(intent, name, None))
        self.record('s_binding', intent.target_binding.mass)
        self.active = False
        return self.events

    def restore(self):
        for owner, name, old in reversed(self.saved):
            setattr(owner, name, old)


def compare_nodes(first, second):
    if first.keys() != second.keys():
        raise ValueError('natural instruction changed the traced node inventory')
    result = {}
    for name, left in first.items():
        right = second[name]
        if len(left) != len(right):
            raise ValueError('natural instruction changed call count: ' + name)
        calls = []
        for index, (a, b) in enumerate(zip(left, right)):
            if a.shape != b.shape:
                raise ValueError('natural instruction changed tensor shape: ' + name)
            scale = rms(a)
            delta = rms(b - a)
            calls.append(dict(call=index, shape=list(a.shape), reference_rms=scale,
                              changed_rms=rms(b), difference_rms=delta,
                              relative_difference=delta / max(scale, 1e-12)))
        result[name] = calls
    return result


def identity_from_masks(coverage, previous=None):
    """K distributions are conditional on real K; never call these object IDs."""
    rows = []
    for camera, view in enumerate(coverage):
        distribution = np.asarray(view['K_given_object'], dtype=float)
        pixels = np.asarray(view['visible_pixels'])
        for obj in range(len(pixels)):
            if pixels[obj] < 4 or view['object_source_mass'][obj] <= 0:
                continue
            q = distribution[obj]
            if not np.isclose(q.sum(), 1., atol=2e-5):
                raise ValueError('K given observed object must conserve conditional mass')
            row = dict(camera=camera, object_index=obj, visible_pixels=int(pixels[obj]),
                       object_source_mass=view['object_source_mass'][obj],
                       object_supported_pixels=view['object_supported_pixels'][obj],
                       K_distribution=q.tolist(), dominant_K=int(q.argmax()),
                       same_view_other_object_tv={str(j): float(np.abs(q - distribution[j]).sum() / 2)
                                                  for j in range(len(pixels)) if j != obj and pixels[j] >= 4 and view['object_source_mass'][j] > 0})
            other = coverage[1 - camera]
            if other['visible_pixels'][obj] >= 4 and other['object_source_mass'][obj] > 0:
                row['same_object_cross_view_tv'] = float(np.abs(q - np.asarray(other['K_given_object'][obj])).sum() / 2)
                row['other_object_cross_view_tv'] = {
                    str(j): float(np.abs(q - np.asarray(other['K_given_object'][j])).sum() / 2)
                    for j in range(len(pixels)) if j != obj and other['visible_pixels'][j] >= 4 and other['object_source_mass'][j] > 0
                }
            if previous is not None and previous[camera]['visible_pixels'][obj] >= 4 and previous[camera]['object_source_mass'][obj] > 0:
                row['same_object_previous_window_tv'] = float(np.abs(q - np.asarray(previous[camera]['K_given_object'][obj])).sum() / 2)
            rows.append(row)
    return rows
