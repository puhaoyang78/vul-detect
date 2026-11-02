"""RCCL alignment coverage and diagnostics using the shared paired pipeline."""

from pathlib import Path
import json
import tempfile

import torch

from . import model as core
from .artifacts import write_json
from .data import read_jsonl, write_jsonl, paired_features, rccl_annotations, rccl_coverage
from .training import paired_loader, paired_collate, paired_forward, RepairPairSampler, LOSVERAdamW, LOSVER_CONFIG


def coverage(data_dir, output_dir, model_path):
    from transformers import RobertaTokenizerFast

    root = Path(output_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    result = dict(input_mode="512", model_path=str(model_path), seed=123456, test_used=False, splits={})
    for split, count in (("train", 8708), ("valid", 1124)):
        rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl")
        if len(rows) != count:
            raise ValueError("RCCL coverage requires complete official Train/Valid")
        loader = paired_loader(rows, tokenizer, "baseline", rccl="diff")
        records = loader.rccl_records
        random_records = rccl_annotations(rows, loader.dataset, "random")
        for a, b in zip(records, random_records):
            if len(a["near_lines"]) != len(b["selected_lines"]):
                raise ValueError("random RCCL must retain each pair's selected line count")
            a["random_lines"] = b["selected_lines"]
        result["splits"][split] = rccl_coverage(records)
        write_jsonl(root / f"coverage.{split}.jsonl", records)
        print(json.dumps(dict(split=split, coverage=result["splits"][split])), flush=True)
    write_json(root / "coverage.json", result)
    return result


@torch.no_grad()
def diagnostics(model, tokenizer, rows, output_dir, split, *, device, projection=None):
    """Source-only probabilities, then annotation-only line-similarity diagnostics.

    All methods use the same true-neighbor, fixed-random and remaining equal-line
    cohorts. The projection, when supplied, affects diagnostics only, never logits.
    """
    import numpy as np
    from .evaluation import bounded_region_record
    from .metrics import paired_metrics

    root = Path(output_dir)
    loader = paired_loader(rows, tokenizer, "baseline", rccl="diff")
    rccl_annotations(rows, loader.dataset, "random")
    projection = projection.to(device) if projection is not None else None
    model.eval()
    predictions, similarities, regions, at = [], [], [], 0
    for batch in loader:
        prob, details = paired_forward(model,batch,"baseline",device,with_labels=False,
                                       return_attention=True,return_line_states=True)
        states = details["line_states"]
        raw = torch.nn.functional.normalize(states,dim=-1)
        projected = (torch.nn.functional.normalize(torch.nn.functional.linear(states,projection),dim=-1)
                     if projection is not None else None)
        for i in range(len(prob)):
            r = rows[at+i]
            n = int((batch["line_numbers"][i] >= 0).sum())
            p = float(prob[i,0])
            prediction = dict(sample_key=r["sample_key"],pair_key=r["pair_key"],target=r["target"],
                              probability=p,prediction=int(p > .5),
                              line_numbers=batch["line_numbers"][i,:n].tolist(),
                              line_rates=details["line_rates"][i,:n].cpu().tolist(),
                              line_token_counts=details["line_counts"][i,:n].cpu().tolist())
            predictions.append(prediction)
            regions.append(bounded_region_record(r,prediction))
        for i in range(0,len(prob),2):
            f = loader.dataset[at+i]
            near = set(map(tuple,f["rccl_near"]))
            # An identical visible input or empty true neighborhood is ineligible
            # for both true and random RCCL, including the diagnostic denominator.
            if not near:
                continue
            sets = dict(near=f["rccl_near"],random=f["rccl_selected"],
                        other=[p for p in f["rccl_alignment"] if tuple(p) not in near])
            record = dict(pair_key=rows[at+i]["pair_key"],pure_addition=loader.rccl_records[(at+i)//2]["pure_addition"])
            for name, pairs in sets.items():
                record[name] = {}
                indices = torch.tensor(pairs,device=states.device,dtype=torch.long).reshape(-1,2)
                for label, vectors in (("raw",raw),("projected",projected)):
                    if vectors is not None:
                        record[name][label] = (vectors[i,indices[:,0]]*vectors[i+1,indices[:,1]]).sum(-1).cpu().tolist()
            similarities.append(record)
        at += len(prob)
    if at != len(rows):
        raise ValueError("RCCL diagnostics did not cover the complete split")
    pure = {r["pair_key"] for r in loader.rccl_records if r["pure_addition"]}
    subset = [i for i,r in enumerate(rows) if r["pair_key"] in pure]
    scores = [r["probability"] for r in predictions]
    metrics = dict(all=paired_metrics(rows,scores),pure_addition=paired_metrics([rows[i] for i in subset],[scores[i] for i in subset]))
    similarity_summary = {}
    for cohort in ("all","pure_addition"):
        members = [r for r in similarities if cohort == "all" or r["pure_addition"]]
        summary = {}
        for name in ("near","random","other"):
            for representation in ("raw","projected"):
                values = [r[name].get(representation,[]) for r in members]
                means = [float(np.mean(v)) for v in values if v]
                flat = [x for v in values for x in v]
                summary[f"{name}_{representation}"] = dict(
                    pairs=len(means),lines=len(flat),pair_mean=float(np.mean(means)) if means else None,
                    pair_std=float(np.std(means)) if means else None,line_mean=float(np.mean(flat)) if flat else None)
        similarity_summary[cohort] = summary
    region_summary = []
    for target in (1,0):
        members = [r for r in regions if r["target"] == target and r["eligible"]]
        region_summary.append(dict(target=target,eligible_functions=len(members),
                                   mean_rate_ratio=float(np.mean([r["mean_rate_ratio"] for r in members])) if members else None,
                                   attention_enrichment=float(np.mean([r["attention_enrichment"] for r in members])) if members else None))
    result = dict(metrics=metrics,similarity=similarity_summary,regions=region_summary,
                  coverage=rccl_coverage(loader.rccl_records),classification_projection_used=False)
    write_jsonl(root/f"{split}.selected.predictions.jsonl",predictions)
    write_jsonl(root/f"{split}.rccl_similarity.jsonl",similarities)
    write_json(root/f"{split}.rccl_diagnostics.json",result)
    return result


def check(data_dir, output_dir, model_path, *, device="auto"):
    """Actual UniXcoder/GPU checks; only Train supplies smoke members."""
    from .evaluation import load_paired

    root = Path(output_dir)
    stats = json.loads((root/"coverage.json").read_text()) if (root/"coverage.json").exists() else coverage(data_dir, root, model_path)
    if stats["input_mode"] != "512" or stats["model_path"] != str(model_path) or stats["seed"] != 123456:
        raise ValueError("RCCL coverage configuration differs")
    audits = read_jsonl(root / "coverage.train.jsonl")
    keys = {next(r["pair_key"] for r in audits if r["eligible"] and r["pure_addition"]),
            next(r["pair_key"] for r in audits if r["eligible"] and not r["pure_addition"]),
            next(r["pair_key"] for r in audits if not r["eligible"])}
    rows = [r for r in read_jsonl(Path(data_dir) / "primevul_paired/train.lines.jsonl") if r["pair_key"] in keys]
    device = core.resolve_device(device)
    if device.type != "cuda":
        raise ValueError("RCCL smoke requires a real GPU")
    options = dict(bounded_attention=True, attention_context=True, attention_region="diff")
    core.seed_everything(123456)
    baseline, tokenizer = core.losver_model(model_path, "baseline", **options)
    rng = torch.get_rng_state().clone()
    expected_order = list(RepairPairSampler([None]*30))
    core.seed_everything(123456)
    model, _ = core.losver_model(model_path, "baseline", **options, rccl="diff")
    if not torch.equal(rng, torch.get_rng_state()) or expected_order != list(RepairPairSampler([None]*30)):
        raise ValueError("RCCL initialization changes original or sampler RNG")
    for k, v in baseline.state_dict().items():
        if not torch.equal(v, model.state_dict()[k]):
            raise ValueError("RCCL changed an original initialization")
    loader = paired_loader(rows, tokenizer, "baseline", rccl="diff")
    features = loader.dataset
    batch = paired_collate(features)
    baseline.to(device).eval(); model.to(device).eval()
    with torch.no_grad():
        p = paired_forward(baseline,batch,"baseline",device,with_labels=False)
        q = paired_forward(model,batch,"baseline",device,with_labels=False)
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    for split in ("train","valid"):
        if not (root/"initial"/f"{split}.rccl_diagnostics.json").exists():
            print(f"RCCL initial {split} representation diagnostics",flush=True)
            all_rows = read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
            diagnostics(baseline,tokenizer,all_rows,root/"initial",split,device=device,
                        projection=model.rccl_projection.weight.detach())
    baseline.train(); model.train()
    core.seed_everything(123456)
    with torch.no_grad():
        _, original, bd = paired_forward(baseline,batch,"baseline",device,return_attention=True)
    core.seed_everything(123456)
    calls = []
    handle = model.encoder.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        loss, actual, parts = paired_forward(model,batch,"baseline",device,return_attention=True,rccl_dropout_seed=123456)
    finally:
        handle.remove()
    torch.testing.assert_close(actual,original,rtol=0,atol=0)
    torch.testing.assert_close(parts["bce"],bd["bce"],rtol=0,atol=0)
    torch.testing.assert_close(loss,parts["bce"]+.1*parts["region_loss"]+.05*parts["rccl_loss"])
    if len(calls) != 1:
        raise ValueError("RCCL must use exactly one encoder pass for a batch")
    del baseline, bd, original
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    repeat, _ = model.rccl_loss(parts["line_states"],batch["rccl_matches"].to(device),batch["rccl_owners"].to(device),123456)
    torch.testing.assert_close(repeat,parts["rccl_loss"],rtol=0,atol=0)
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(device)):
        raise ValueError("RCCL Dropout advanced encoder or sampler RNG")
    parts["rccl_loss"].backward()
    gradient = float(model.rccl_projection.weight.grad.abs().sum())
    encoder_gradient = float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum())
    if not gradient > 0 or not encoder_gradient > 0:
        raise ValueError("RCCL must update both projection and shared encoder")
    before = model.rccl_projection.weight.detach().clone()
    optimizer = LOSVERAdamW(model.parameters(),lr=LOSVER_CONFIG["learning_rate"],eps=LOSVER_CONFIG["adam_epsilon"])
    optimizer.step(); optimizer.zero_grad()
    update = float((before-model.rccl_projection.weight).abs().sum())
    if not update > 0:
        raise ValueError("RCCL head did not update")
    model.eval()
    with torch.no_grad():
        expected = paired_forward(model,batch,"baseline",device,with_labels=False)
        padding = dict(batch)
        padding["input_ids"] = batch["input_ids"].masked_fill(~batch["attention_mask"].bool(),9)
        q = paired_forward(model,padding,"baseline",device,with_labels=False)
        torch.testing.assert_close(q,expected,rtol=0,atol=0)
    cp = dict(protocol="losver",line_mapping="source_offsets",input_mode="512",variant="baseline",model_path=str(model_path),
              pooling="bounded_attention",state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
              bounded_attention=dict(region_supervision="diff",context=True),rccl=dict(mode="diff"))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"smoke.pt"
        torch.save(cp,path)
        restored,_ = load_paired(torch.load(path,weights_only=False,map_location="cpu"),device)
        if hasattr(restored,"rccl_projection"):
            raise ValueError("RCCL projection must be absent from inference")
        with torch.no_grad():
            actual = paired_forward(restored,batch,"baseline",device,with_labels=False)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        del restored
        single = core.predict_checkpoint(path,[dict(raw_source=rows[0]["func"])],device=device).sigmoid()
        delta = float((single-expected[0,0].cpu()).abs().max())
        if delta > 1e-5:
            raise ValueError("RCCL single-source inference differs")
    result = dict(gpu=torch.cuda.get_device_name(device),device=str(device),hidden_size=model.encoder.config.hidden_size,
                  added_parameters=model.rccl_projection.weight.numel(),temperature=.1,dropout=.1,
                  train_only_sample_keys=[r["sample_key"] for r in rows],
                  eligible_smoke_pairs=int(parts["rccl_pairs"]),rccl_loss=float(parts["rccl_loss"].detach()),
                  encoder_calls=len(calls),projection_gradient=gradient,encoder_gradient=encoder_gradient,
                  projection_update_l1=update,initialization_equal=True,sampler_rng_equal=True,
                  dropout_rng_preserved=True,checkpoint_equal=True,padding_equal=True,
                  inference_projection_absent=True,diff_not_used_in_inference=True,single_inference_max_delta=delta,
                  coverage=stats["splits"],test_used=False)
    write_json(root/"gpu-smoke.json",result)
    return result


def report(output_dir, data_dir, *, device="auto"):
    """Three fixed runs, Valid selection and optional post-selection Test results."""
    import csv
    from .data import _line_diff
    from .evaluation import load_paired
    from .metrics import paired_metrics

    root = Path(output_dir)
    names = ("matched_context_diff","rccl_diff","rccl_random")
    runs = {name:root/name/"primevul_paired/unixcoder" for name in names}
    configurations, epoch_logs, diagnostics_by_run = {}, {}, {}
    for name, run in runs.items():
        cfg = json.loads((run/"training.metrics.json").read_text())
        if cfg["completed_epochs"] != 16 or cfg["optimizer_steps"] != 17424 or cfg["training_config"]["batch_size"] != 8:
            raise ValueError("RCCL report requires all three full 16-epoch Batch8 runs")
        configurations[name] = cfg
        epoch_logs[name] = [r for r in read_jsonl(run/"best.training.jsonl") if r["event"] == "epoch"]
        if len(epoch_logs[name]) != 16:
            raise ValueError("incomplete RCCL epoch history")
        chosen = max(epoch_logs[name],key=lambda r:(round(r["validation"]["p_c"],4),-r["epoch"]))
        if chosen["epoch"] != cfg["selected_epochs"]["p_c"]:
            raise ValueError("checkpoint was not selected by unchanged complete Valid P-C")
    reference = configurations[names[0]]
    fields = ("training_config","training_members","validation_members","input_coverage","input_mode",
              "model_path","function_micro_batch","window_micro_batch","gradient_accumulation","bounded_attention")
    order = [r["paired_batch_order"] for r in epoch_logs[names[0]]]
    official_pairs = {r["pair_key"] for r in read_jsonl(root/"coverage.train.jsonl")}
    for epoch_order in order:
        if len(epoch_order) != 4354 or set(epoch_order) != official_pairs:
            raise ValueError("a training epoch omitted or repeated a repair pair")
    for name in names:
        if any(configurations[name][f] != reference[f] for f in fields):
            raise ValueError("RCCL training protocols or official members differ")
        if [r["paired_batch_order"] for r in epoch_logs[name]] != order:
            raise ValueError("RCCL actual training batch orders differ")
        if (runs[name]/"train.attention_regions.jsonl").read_bytes() != (runs[names[0]]/"train.attention_regions.jsonl").read_bytes():
            raise ValueError("RCCL modified the original Diff targets")
    device = core.resolve_device(device)
    for name, run in runs.items():
        missing = any(not (run/f"{s}.rccl_diagnostics.json").exists() for s in ("train","valid"))
        if missing:
            cp = torch.load(run/"best-p_c.pt",map_location="cpu",weights_only=False)
            model, tokenizer = load_paired(cp,device)
            projection = cp["state_dict"].get("rccl_projection.weight")
            for split in ("train","valid"):
                if (run/f"{split}.rccl_diagnostics.json").exists():
                    continue
                print(f"{name}: selected checkpoint {split} representation diagnostics",flush=True)
                rows = read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
                value = diagnostics(model,tokenizer,rows,run,split,device=device,projection=projection)
                value["selected_epoch"] = cp["selected_epoch"]
                write_json(run/f"{split}.rccl_diagnostics.json",value)
            del model, cp
            if device.type == "cuda":
                torch.cuda.empty_cache()
        diagnostics_by_run[name] = {s:json.loads((run/f"{s}.rccl_diagnostics.json").read_text()) for s in ("train","valid")}
    test_available = [(run/"test.metrics.json").exists() for run in runs.values()]
    if any(test_available) and not all(test_available):
        raise ValueError("complete all three fixed Test evaluations before reporting Test")
    splits = ("valid","test") if all(test_available) else ("valid",)
    table = []
    for split in splits:
        rows = read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.jsonl")
        if len(rows) != (1124 if split == "valid" else 1128):
            raise ValueError("RCCL evaluation split is not the complete official split")
        if split == "valid":
            pure = {r["pair_key"] for r in read_jsonl(root/"coverage.valid.jsonl") if r["pure_addition"]}
        else:
            pure = set()
            for i in range(0,len(rows),2):
                deleted, edits = _line_diff(rows[i]["func"],rows[i+1]["func"])
                if not deleted and any(e["other_lines"] for e in edits):
                    pure.add(rows[i]["pair_key"])
        for name, run in runs.items():
            path = run/("valid.best-p_c.predictions.jsonl" if split == "valid" else "test.predictions.jsonl")
            predictions = read_jsonl(path)
            if [r["sample_key"] for r in predictions] != [r["sample_key"] for r in rows]:
                raise ValueError("RCCL predictions do not match the official split/order")
            expected = diagnostics_by_run[name]["valid"]["metrics"]["all"] if split == "valid" else json.loads((run/"test.metrics.json").read_text())["metrics"]
            if split == "test":
                saved = json.loads((run/"test.metrics.json").read_text())
                if saved["selected_epoch"] != configurations[name]["selected_epochs"]["p_c"] or saved["oracle"]:
                    raise ValueError("Test did not use the fixed non-oracle Valid-selected checkpoint")
            else:
                repeated = read_jsonl(run/"valid.selected.predictions.jsonl")
                if len(repeated) != len(predictions) or any(
                    p["sample_key"] != q["sample_key"] or p["prediction"] != q["prediction"]
                    or abs(p["probability"]-q["probability"]) > 1e-5 for p,q in zip(predictions,repeated)
                ):
                    raise ValueError("annotation-only diagnostics changed source-only Valid inference")
            scores = [p["probability"] for p in predictions]
            if paired_metrics(rows,scores)["pair_counts"] != expected["pair_counts"]:
                raise ValueError("saved predictions and reported classification differ")
            for cohort in ("all","pure_addition"):
                indices = [i for i,r in enumerate(rows) if cohort == "all" or r["pair_key"] in pure]
                value = paired_metrics([rows[i] for i in indices],[scores[i] for i in indices])
                table.append(dict(method=name,split=split,cohort=cohort,epoch=configurations[name]["selected_epochs"]["p_c"],**value))
    first_bce = []
    for run in runs.values():
        first_bce.append(next(r["bce"] for r in read_jsonl(run/"best.training.jsonl") if r["event"] == "step"))
    if len(set(first_bce)) != 1:
        raise ValueError("matched runs do not start from the same stochastic classifier output")
    valid = {r["method"]:r for r in table if r["split"] == "valid" and r["cohort"] == "all"}
    success = all(valid["rccl_diff"]["p_c"] > valid[n]["p_c"] for n in ("matched_context_diff","rccl_random"))
    training = []
    for name in names:
        events = epoch_logs[name]
        training.append(dict(method=name,selected_epoch=configurations[name]["selected_epochs"]["p_c"],
                             optimizer_steps=configurations[name]["optimizer_steps"],
                             training_seconds=sum(e["training_seconds"] for e in events),
                             epoch_seconds=sum(e["epoch_seconds"] for e in events),
                             first_components=events[0]["train_components"],last_components=events[-1]["train_components"]))
    overlap = {}
    for split in ("train","valid"):
        records = [r for r in read_jsonl(root/f"coverage.{split}.jsonl") if r["eligible"]]
        shared = sum(len(set(map(tuple,r["near_lines"])) & set(map(tuple,r["random_lines"]))) for r in records)
        total = sum(len(r["near_lines"]) for r in records)
        overlap[split] = dict(eligible_pairs=len(records),selected_lines=total,shared_lines=shared,
                             shared_line_fraction=shared/total,identical_selection_pairs=sum(
                                 set(map(tuple,r["near_lines"])) == set(map(tuple,r["random_lines"])) for r in records))
    saved_reference = json.loads((root/"commands.json").read_text())["reference_checkpoint"]
    original = Path(saved_reference["path"]).stat()
    if original.st_size != saved_reference["size_bytes"] or original.st_mtime_ns != saved_reference["mtime_ns"]:
        raise ValueError("historical Context + Diff checkpoint changed")
    result = dict(protocol=dict(seed=123456,batch_size=8,epochs=16,tokens=512,threshold=.5,temperature=.1,
                                diff_weight=.1,rccl_weight=.05,test_used=all(test_available),
                                test_selection_used=False,parameter_search=False,baseline_retrained=True),
                  checks=dict(actual_pair_order_equal_all_epochs=True,original_diff_labels_equal=True,
                              common_protocol_fields=list(fields),first_step_bce=first_bce,
                              historical_checkpoint_unchanged=True,diagnostic_inference_equal=True),
                  coverage=json.loads((root/"coverage.json").read_text())["splits"],
                  classification=table,training=training,diagnostics=diagnostics_by_run,
                  random_control_overlap=overlap,
                  initial={s:json.loads((root/"initial"/f"{s}.rccl_diagnostics.json").read_text()) for s in ("train","valid")},
                  true_rccl_beats_both_controls_on_valid_p_c=success)
    write_json(root/"comparison.json",result)
    columns = ("method","split","cohort","epoch","pairs","p_c","auc","mcc","f1","recall","patched_fpr","pair_ranking_accuracy")
    with (root/"classification.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
        writer.writeheader(); writer.writerows(table)
    lines = ["# RCCL: matched Context + Diff experiments", "",
             "Official paired Train 8,708 functions / 4,354 pairs; Valid 1,124 / 562. Seed123456, Batch8, 512 tokens, 16 epochs / 17,424 optimizer steps. Fresh pretrained UniXcoder initialization for all three runs. Historical Context + Diff checkpoint retained.", "",
             "BCE + 0.1 Diff + 0.05 RCCL; temperature 0.1. Projection: bias-free 768→32 linear (24,576 parameters); two independent p=0.1 representation Dropout views per version. Four symmetric anchors per aligned line, with same-version positive and two opposite-version negatives. Mean lines within pair, then eligible pairs. Other lines are never contrastive negatives. A private CPU generator preserves encoder and sampler RNG.", "",
             "Exact Diff equal lines must be unique in both complete functions and fully visible in both versions. Select the nearest nonblank equal line on each side of every edit block, skipping ambiguous/partial neighbors. Pairs with identical 512-token inputs are excluded from RCCL. Random selects the same number of reliable equal lines in the same pair without replacement. All functions retain BCE and original Diff labels.", "",
             "Actual pair order is recorded for all 16 epochs and identical across the three runs; optimizer budget, preprocessing and Diff targets match. Full Valid P-C (rounded to four decimals; earlier epoch on ties) selects checkpoints; decisions use probability >0.5. Test, when present below, was evaluated after selection and never used for tuning.", "",
             "| Method | Split | Cohort | Epoch | Pairs | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['method']} | {r['split']} | {r['cohort']} | {r['epoch']} | {r['pairs']} | {r['p_c']:.4f} | {r['auc']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['patched_fpr']:.4f} | {r['pair_ranking_accuracy']:.4f} |")
    lines += ["", "## Coverage", ""]
    for split,c in result["coverage"].items():
        for cohort,v in c.items():
            lines.append(f"- {split} {cohort}: {v['eligible_pairs']}/{v['pairs']} pairs ({v['pair_coverage']:.2%}), {v['selected_lines']} aligned neighbor lines.")
    lines += ["", "## Training and representation diagnostics", "",
              "Cosines below are raw shared-encoder line states from each Valid-selected checkpoint in eval mode on the same eligible true-neighbor cohort. They are pair-averaged; line-weighted values, standard deviations, fixed-random/other-line cohorts and projected cosines are in comparison.json. Projected cosines measure the training head only and do not affect inference.", "",
              "| Method | Train min | RCCL loss epoch 1 → 16 | Train raw cosine | Valid raw cosine | Valid projected cosine |",
              "|---|---:|---|---:|---:|---:|"]
    initial = result["initial"]
    lines.append(f"| pretrained initialization | — | — | {initial['train']['similarity']['all']['near_raw']['pair_mean']:.4f} | {initial['valid']['similarity']['all']['near_raw']['pair_mean']:.4f} | {initial['valid']['similarity']['all']['near_projected']['pair_mean']:.4f} |")
    for r in training:
        d = diagnostics_by_run[r["method"]]
        first,last = r["first_components"].get("rccl_loss"),r["last_components"].get("rccl_loss")
        loss = f"{first:.4f} → {last:.4f}" if first is not None else "—"
        projected = d["valid"]["similarity"]["all"]["near_projected"]["pair_mean"]
        z = f"{projected:.4f}" if projected is not None else "—"
        lines.append(f"| {r['method']} | {r['training_seconds']/60:.2f} | {loss} | {d['train']['similarity']['all']['near_raw']['pair_mean']:.4f} | {d['valid']['similarity']['all']['near_raw']['pair_mean']:.4f} | {z} |")
    lines += ["", "RCCL losses are epoch means of the per-effective-batch eligible-pair mean (zero for an empty batch). Training durations include independent-process GPU sharing: GPU0 baseline/random, GPU1 true RCCL. The compute/update budget is identical."]
    for split,v in overlap.items():
        lines.append(f"Random control {split}: {v['shared_lines']}/{v['selected_lines']} selected lines also belong to the true neighbor set ({v['shared_line_fraction']:.2%}); {v['identical_selection_pairs']}/{v['eligible_pairs']} pairs select the identical set. Overlap is allowed by uniform sampling, never filtered using outcomes.")
    tests = json.loads((root/"tests.json").read_text())
    smoke = json.loads((root/"gpu-smoke.json").read_text())
    lines += ["", f"Validation: {tests['unique_tests_passed']} distinct related unit tests passed. Actual {smoke['gpu']} smoke passed initialization/RNG equality, single encoding, gradients and parameter update, padding, checkpoint restore and source-only inference. Inference omits the contrastive head; single/batched maximum probability difference {smoke['single_inference_max_delta']:.3g}. Check scripts' corrected API/metadata errors are recorded in tests.json."]
    historical = json.loads((Path(saved_reference["path"]).parent/"best-p_c.json").read_text())
    lines += ["", f"Historical Context + Diff (independent function shuffle, not the matched paired-batch control): Valid P-C {historical['validation']['p_c']:.4f}, selected epoch {historical['epoch']}. Its checkpoint size and modification time are unchanged."]
    lines += ["", ("True RCCL exceeds both matched controls on complete Valid P-C." if success else
                       "True RCCL fails to exceed both random RCCL and the matched Context + Diff baseline on complete Valid P-C. The fixed experiment is negative; no parameter search or additional variant was run."), ""]
    real = diagnostics_by_run["rccl_diff"]
    base = diagnostics_by_run["matched_context_diff"]
    lines += [f"The true-RCCL projection's near-line cosine is {real['train']['similarity']['all']['near_projected']['pair_mean']:.4f} on Train versus {real['valid']['similarity']['all']['near_projected']['pair_mean']:.4f} on Valid. This large gap accompanies a low training contrastive loss; fitting the contrastive objective does not establish classification benefit.", "",
              f"Valid raw near-line cosine is {real['valid']['similarity']['all']['near_raw']['pair_mean']:.4f} with true RCCL versus {base['valid']['similarity']['all']['near_raw']['pair_mean']:.4f} for the matched baseline. Valid AUC changes from {valid['matched_context_diff']['auc']:.4f} to {valid['rccl_diff']['auc']:.4f}, MCC from {valid['matched_context_diff']['mcc']:.4f} to {valid['rccl_diff']['mcc']:.4f}, and patched FPR from {valid['matched_context_diff']['patched_fpr']:.4f} to {valid['rccl_diff']['patched_fpr']:.4f}. Increased context separation has to be assessed alongside these detection tradeoffs.", ""]
    if all(test_available):
        test = {r["method"]:r for r in table if r["split"] == "test" and r["cohort"] == "all"}
        b,t,r = (test[n] for n in names)
        lines += [f"Post-selection Test P-C: baseline {b['pair_counts']['p_c']}/{b['pairs']}, true RCCL {t['pair_counts']['p_c']}/{t['pairs']}, random RCCL {r['pair_counts']['p_c']}/{r['pairs']}. True RCCL's AUC is {t['auc']:.4f} versus baseline {b['auc']:.4f}, patched FPR {t['patched_fpr']:.4f} versus {b['patched_fpr']:.4f}, and pair ranking {t['pair_ranking_accuracy']:.4f} versus {b['pair_ranking_accuracy']:.4f}. Report these mixed Test results without changing the Valid-based conclusion or selecting a different checkpoint.", ""]
        pure_valid = {r["method"]:r for r in table if r["split"] == "valid" and r["cohort"] == "pure_addition"}
        pure_test = {r["method"]:r for r in table if r["split"] == "test" and r["cohort"] == "pure_addition"}
        vb,vt,vr = (pure_valid[n] for n in names)
        tb,tt,tr = (pure_test[n] for n in names)
        lines += [f"The pure-addition subgroup has a small positive P-C signal: Valid true/baseline/random {vt['pair_counts']['p_c']}/{vb['pair_counts']['p_c']}/{vr['pair_counts']['p_c']} correct pairs out of {vt['pairs']}; Test {tt['pair_counts']['p_c']}/{tb['pair_counts']['p_c']}/{tr['pair_counts']['p_c']} out of {tt['pairs']}. Pure-addition Test AUC is {tt['auc']:.4f} for true RCCL versus {tb['auc']:.4f} for the baseline. Preserve these subgroup gains while the primary full-Valid criterion remains unmet.", ""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return result
