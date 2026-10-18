"""The existing visibility checks and dynamic width-two, depth-three source search."""

from __future__ import annotations
from collections import Counter, defaultdict
import json
import numpy as np


def _source_visibility(value, tokenizer, source_budget, cache):
    from .syntax import source_token_spans

    if value in cache:
        return cache[value]
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("dynamic visibility requires fast original source offsets")
    encoded = tokenizer(
        value, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, verbose=False,
    )
    offsets = encoded["offset_mapping"][:source_budget]
    if len(encoded["input_ids"]) != len(encoded["offset_mapping"]):
        raise ValueError("source IDs/offset length mismatch")
    if any(not 0 <= a <= b <= len(value) for a, b in offsets):
        raise ValueError("source offsets outside Python character coordinates")
    # Merge UTF-8/BPE duplicate character spans before checking visibility.
    end = max((b for a, b in offsets if b > a), default=0)
    tokens = source_token_spans(value)
    states = [
        (0 if a >= end else 2 if b <= end else 1, min(max(end - a, 0), b - a))
        for _, a, b in tokens
    ]
    result = (encoded["input_ids"][:source_budget], states, tokens,
              len(encoded["input_ids"]) > source_budget, end)
    cache[value] = result
    return result


def visible_candidates(
    source, tokenizer, source_budget, *, proposals, reference_source=None, visibility_cache=None
):
    """Reject a view if it displaces any part of the original visible prefix."""
    from .syntax import apply_source_edits

    cache = {} if visibility_cache is None else visibility_cache
    reference = source if reference_source is None else reference_source
    original_ids, before, original_tokens, truncated, boundary = _source_visibility(
        reference, tokenizer, source_budget, cache
    )
    result = []
    reasons = Counter()
    for candidate in proposals:
        transformed = apply_source_edits(source, candidate["edits"])
        ids, after, tokens, _, end = _source_visibility(transformed, tokenizer, source_budget, cache)
        if len(tokens) != len(original_tokens):
            raise ValueError("dynamic surface edit changed lexical token cardinality")
        lost = any(
            old[0] > new[0] or old[0] == new[0] == 1 and old[1] > new[1]
            for old, new in zip(before, after)
        )
        gained = truncated and any(
            old[0] < new[0] or old[0] == new[0] == 1 and old[1] < new[1]
            for old, new in zip(before, after)
        )
        partial_rename = any(
            old[0] == 1 and a[0] != b[0] for old, a, b in zip(before, original_tokens, tokens)
        )
        if lost:
            reasons["visible_execution_removed"] += 1
        elif gained:
            reasons["extra_execution_exposed"] += 1
        elif partial_rename:
            reasons["partial_visible_identifier_changed"] += 1
        elif ids == original_ids:
            reasons["identical_encoder_input"] += 1
        else:
            result.append(
                dict(
                    candidate,
                    raw_source=transformed,
                    visible_tokens=len(ids),
                    visible_chars=end,
                    original_visible_chars=boundary,
                    original_truncated=truncated,
                    visible_execution_tokens=sum(s[0] == 2 for s in after),
                    visibility_verified=True,
                )
            )
    return result, dict(reasons)


def surface_seed(seed, epoch, sample_key, purpose):
    """Stable method RNG; independent of Python hash, dropout and batch ordering."""
    import hashlib

    return int.from_bytes(
        hashlib.sha256(json.dumps([seed, epoch, sample_key, purpose]).encode()).digest()[:8], "big"
    )


def replay_surface_units(source, units):
    from .syntax import apply_source_edits

    for unit in units:
        source = apply_source_edits(source, unit["edits"])
    return source


def search_surface_views(
    records,
    score,
    tokenizer,
    source_budget,
    materials,
    *,
    seed=42,
    epoch=0,
    query_limit=12,
    bank="train",
    purpose="train",
    random_selection=False,
    kinds=None,
    required_combination=None,
    original_logits=None,
    selection="hardest",
):
    """Width-two depth-three search; batch score queries, bounded proposals.

    score accepts function rows and returns logits without changing parameters.
    It is never called by random-selection ablation or candidate-only preparation.
    All visited candidates remain available for worst-case and multi-view metrics.
    """
    import random
    from .syntax import surface_preserving_candidates, resolve_language, SURFACE_RULES

    if query_limit not in (12, 48) or bank not in ("train", "heldout"):
        raise ValueError("unsupported surface search budget")
    if selection not in ("hardest", "boundary"):
        raise ValueError("unknown hard selection strategy")
    if selection == "boundary" and (
        purpose != "train" or score is None or random_selection or required_combination
    ):
        raise ValueError("boundary selection is only supported for scored training search")
    states = []
    for row in records:
        rng = random.Random(surface_seed(seed, epoch, row["sample_key"], purpose))
        source = row["raw_source"]
        cache = {}
        ids = tuple(_source_visibility(source, tokenizer, source_budget, cache)[0])
        states.append(
            dict(
                row=row,
                rng=rng,
                beams=[dict(raw_source=source, units=[], added_comment_tokens=0, used_bindings=[])],
                seen_sources={source},
                seen_inputs={ids},
                visited=[],
                reasons=Counter(),
                original_logit=None,
                original_queries=0,
                candidate_queries=0,
                visibility_cache=cache,
                stop_depth=None,
                selection_reason=None,
            )
        )
    per_depth = query_limit // 3
    for depth in range(3):
        pending = []
        for index, state in enumerate(states):
            row = state["row"]
            draft = []
            for parent in state["beams"]:
                allowed = (
                    [required_combination[depth]]
                    if required_combination and depth < len(required_combination)
                    else [] if required_combination else kinds
                )
                if allowed == []:
                    continue
                proposed, reasons = surface_preserving_candidates(
                    parent["raw_source"],
                    resolve_language(
                        parent["raw_source"], row.get("language", "c"), row.get("file_name", "")
                    ),
                    rng=state["rng"],
                    materials=materials,
                    tokenizer=tokenizer,
                    bank=bank,
                    kinds=allowed,
                    used_bindings=parent["used_bindings"],
                    comment_tokens=parent["added_comment_tokens"],
                    limit=per_depth // len(state["beams"]),
                    proposal_limit=24,
                )
                state["reasons"].update(reasons)
                legal, reasons = visible_candidates(
                    parent["raw_source"],
                    tokenizer,
                    source_budget,
                    proposals=proposed,
                    reference_source=row["raw_source"],
                    visibility_cache=state["visibility_cache"],
                )
                state["reasons"].update(reasons)
                for unit in legal:
                    path = parent["units"] + [
                        {k: v for k, v in unit.items() if k not in ("raw_source",)}
                    ]
                    types = Counter(u["kind"] for u in path)
                    if (
                        bank == "train"
                        and required_combination is None
                        and any(
                            not (Counter(combo) - types)
                            for combo in SURFACE_RULES["heldout_combinations"]
                        )
                    ):
                        state["reasons"]["heldout_operator_combination"] += 1
                        continue
                    changed = unit["raw_source"]
                    ids = tuple(state["visibility_cache"][changed][0])
                    if changed in state["seen_sources"] or ids in state["seen_inputs"]:
                        state["reasons"]["duplicate_source_or_input"] += 1
                        continue
                    state["seen_sources"].add(changed)
                    state["seen_inputs"].add(ids)
                    candidate = dict(
                        raw_source=changed,
                        units=path,
                        added_comment_tokens=parent["added_comment_tokens"]
                        + unit["added_comment_tokens"],
                        used_bindings=parent["used_bindings"]
                        + (
                            [unit["binding_token_index"]]
                            if unit["binding_token_index"] is not None
                            else []
                        ),
                        strict_semantic=all(u["strict_semantic"] for u in path),
                    )
                    draft.append(candidate)
            for candidate in draft[:per_depth]:
                pending.append((index, candidate))
        if not pending:
            break
        if score is not None and not random_selection:
            missing = [i for i, _ in pending if states[i]["original_logit"] is None]
            missing = list(dict.fromkeys(missing))
            if original_logits is not None:
                for i in missing:
                    z = float(original_logits[states[i]["row"]["sample_key"]])
                    if not np.isfinite(z):
                        raise ValueError("nonfinite original score")
                    states[i]["original_logit"] = z
            elif missing:
                values = score([
                    dict(states[i]["row"], _source_token_ids=states[i]["visibility_cache"][
                        states[i]["row"]["raw_source"]][0]) for i in missing
                ])
                if len(values) != len(missing):
                    raise ValueError("original score cardinality mismatch")
                for i, z in zip(missing, values):
                    if not np.isfinite(z):
                        raise ValueError("nonfinite original score")
                    states[i]["original_logit"] = float(z)
                    states[i]["original_queries"] = 1
            scored = [
                (at, i, c) for at, (i, c) in enumerate(pending)
                if selection != "boundary"
                or int(states[i]["original_logit"] >= 0) == states[i]["row"]["label"]
            ]
            logits = score([
                dict(states[i]["row"], raw_source=c["raw_source"],
                     _source_token_ids=states[i]["visibility_cache"][c["raw_source"]][0])
                for _, i, c in scored
            ]) if scored else []
            if len(logits) != len(scored):
                raise ValueError("candidate score cardinality mismatch")
            values = [None] * len(pending)
            for (at, _, _), z in zip(scored, logits):
                if z is None:
                    raise ValueError("scored candidate has no logit")
                values[at] = z
        else:
            values = [None] * len(pending)
        next_beams = defaultdict(list)
        for (i, candidate), z in zip(pending, values):
            state = states[i]
            if z is not None:
                if not np.isfinite(z):
                    raise ValueError("nonfinite candidate score")
                candidate.update(
                    logit=float(z),
                    loss=float(np.logaddexp(0, float(z)) - state["row"]["label"] * float(z)),
                )
                state["candidate_queries"] += 1
            state["visited"].append(candidate)
            next_beams[i].append(candidate)
        for i, state in enumerate(states):
            children = next_beams[i]
            if selection == "boundary" and children:
                if int(state["original_logit"] >= 0) != state["row"]["label"]:
                    state["selection_reason"] = "original_wrong_random"
                elif any(int(c["logit"] >= 0) != state["row"]["label"] for c in children):
                    state["selection_reason"] = "first_boundary_min_bce"
                if state["selection_reason"] is not None:
                    state["beams"] = []
                    state["stop_depth"] = depth + 1
                    continue
            if random_selection or score is None:
                state["rng"].shuffle(children)
                state["beams"] = children[:2]
            else:
                state["beams"] = sorted(children, key=lambda c: c["loss"], reverse=True)[:2]
    result = []
    for state in states:
        visited = state["visited"]
        if required_combination:
            visited = [c for c in visited if len(c["units"]) == len(required_combination)]
        reason = state["selection_reason"]
        if not visited:
            selected = None
        elif random_selection or score is None or reason == "original_wrong_random":
            selected = state["rng"].choice(visited)
        elif reason == "first_boundary_min_bce":
            crossed = [c for c in visited if int(c["logit"] >= 0) != state["row"]["label"]]
            selected = min(crossed, key=lambda c: c["loss"])
        else:
            selected = max(visited, key=lambda c: c["loss"])
        base = state["original_logit"]
        y = state["row"]["label"] if base is not None else None
        result.append(
            dict(
                sample_key=state["row"]["sample_key"],
                selected=selected,
                visited=state["visited"],
                candidate_queries=state["candidate_queries"],
                original_queries=state["original_queries"],
                original_logit=base,
                selection_reason=("no_legal_candidate" if selected is None else reason or (
                    "random" if random_selection or score is None else
                    "no_boundary_max_bce" if selection == "boundary" else "hardest_max_bce"
                )),
                stop_depth=state["stop_depth"],
                visibility_encodings=len(state["visibility_cache"]),
                loss_delta=(
                    selected["loss"] - float(np.logaddexp(0, base) - y * base)
                    if selected is not None and "loss" in selected and base is not None
                    else None
                ),
                reasons=dict(state["reasons"]),
            )
        )
        if len(state["visited"]) > query_limit:
            raise ValueError("surface candidate query budget exceeded")
    return result
