"""Pinned author implementations with shared PrimeVul evaluation and resume hooks.

The optimizer/forward/backward/sampler loops remain in the upstream scripts.
Patches only connect data, fixed Valid P-C selection and existing checkpoint I/O.
"""

import csv
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from .artifacts import write_json
from .data import read_jsonl, write_jsonl
from .metrics import paired_metrics
from .progress import TrainingProgress
from .training import save_training_state, load_training_state, rng_state, restore_rng, LOSVERAdamW

SEEDS = (42, 1234, 123456)
UPSTREAM = {
    "linevul": dict(repo="https://github.com/awsm-research/LineVul.git",
        commit="9401ec6e60b762a9308645961268203244bee048", directory="LineVul",
        script="linevul/linevul_main.py", model="codebert-base", epochs=10, batch_size=16,
        block_size=512, learning_rate=2e-5, warmup_fraction=.2, optimizer="torch.optim.AdamW",
        probability_column=1, sampler="RandomSampler"),
    "svuld": dict(repo="https://github.com/jacknichao/SVulD.git",
        commit="6ebf229e92d030f68752cd2d361f56ff97dffcbd", directory="SVulD",
        script="Source_Code/SVulD/run.py", model="unixcoder-base-nine", epochs=20, batch_size=32,
        block_size=400, learning_rate=2e-5, warmup_fraction=.1, optimizer="torch.optim.AdamW",
        probability_column=1, sampler="SequentialSampler", r_drop=True, simcse=False, simct=False,
        early_stopping=dict(patience=5, delta=0)),
    "regvd": dict(repo="https://github.com/daiquocnguyen/GNN-ReGVD.git",
        commit="b5e6657b003ada704b7c75e4bd237c1eee1d8138", directory="ReGVD",
        script="code/run.py", model="graphcodebert-base", epochs=100, batch_size=128,
        block_size=400, learning_rate=5e-4, warmup_fraction=.1, optimizer="Transformers AdamW rule",
        probability_column=0, sampler="RandomSampler", gnn="ReGCN", hidden_size=128,
        num_GNN_layers=2, format="uni", window_size=5, att_op="mul", num_classes=2),
}
PAIRS = dict(train=4354, valid=562, test=564)
METRICS = ("p_c", "mcc", "f1", "auc", "auprc", "recall", "patched_fpr")
LOSVER_REFERENCE = Path("results/losver_512/primevul_paired")


def official_rows(data_dir, split):
    rows = read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.jsonl")
    if (len(rows) != 2*PAIRS[split] or len({r["sample_key"] for r in rows}) != len(rows)
        or len({r["pair_key"] for r in rows}) != PAIRS[split]
        or [r["target"] for r in rows] != [1, 0]*PAIRS[split]
        or any(rows[i]["pair_key"] != rows[i+1]["pair_key"] for i in range(0,len(rows),2))):
        raise ValueError(f"{split}: complete original PrimeVul repair pairs required")
    return rows


def prepare(data_dir="data"):
    """Idempotent pinned checkout plus minimal compatibility patches and exports."""
    for method, spec in UPSTREAM.items():
        repo = Path("external")/spec["directory"]
        sparse = ("Source_Code/SVulD" if method=="svuld" else "linevul" if method=="linevul" else "code")
        if not repo.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repo"], str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", sparse], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", spec["commit"]], check=True)
        actual = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        if actual != spec["commit"]:
            raise ValueError(f"{repo}: expected author commit {spec['commit']}, got {actual}")
        patch = Path(__file__).with_name("upstream")/f"{method}.patch"
        applied = subprocess.run(["git","-C",str(repo),"apply","--reverse","--check",str(patch.resolve())], capture_output=True)
        if applied.returncode:
            subprocess.run(["git","-C",str(repo),"apply","--check",str(patch.resolve())],check=True)
            subprocess.run(["git","-C",str(repo),"apply",str(patch.resolve())],check=True)
    export_data(data_dir)
    return {m:dict(repo=s["repo"],commit=s["commit"]) for m,s in UPSTREAM.items()}


def export_data(data_dir):
    """Format adaptation only; originals, labels, member order and source stay intact."""
    for split in PAIRS:
        rows = official_rows(data_dir, split)
        for method in UPSTREAM:
            folder = Path(data_dir)/"official_baselines"/method
            folder.mkdir(parents=True, exist_ok=True)
            if method == "linevul":
                with (folder/f"{split}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, ("processed_func","target","sample_key","pair_key"))
                    writer.writeheader()
                    writer.writerows(dict(processed_func=r["func"], **{k:r[k] for k in ("target","sample_key","pair_key")}) for r in rows)
            else:
                exported = []
                for i, row in enumerate(rows):
                    common = {k:row[k] for k in ("sample_key","pair_key")}
                    if method == "svuld":
                        # Author R-Drop uses two Dropout views of the SAME function.
                        # Train retains its real paired contrast field; eval needs no counterpart.
                        contrast = rows[i ^ 1]["func"] if split=="train" else row["func"]
                        exported.append(dict(common, code=row["func"], contrast=contrast,
                                             label=row["target"], index=i))
                    else:
                        exported.append(dict(common, func=row["func"], target=row["target"], idx=i))
                write_jsonl(folder/f"{split}.jsonl", exported)


def _model(model):
    return model.module if hasattr(model, "module") else model


def native_begin(args, model, optimizer, scheduler):
    """Use the repository's atomic optimizer/RNG/log recovery without replacing train()."""
    args.reproduction_metadata = json.loads(Path(args.reproduction_config).read_text())
    args.reproduction_cohorts = {}
    args.reproduction_run = Path(args.output_dir)
    args.reproduction_split = "valid"
    run = args.reproduction_run
    continuing = args.reproduction_resume and (run/"last.pt").exists()
    if not continuing and ((run/"last.pt").exists() or (run/"best.training.jsonl").exists()):
        raise FileExistsError(f"{run}: started run requires --resume")
    args.reproduction_rng = None
    if continuing:
        loop, args.reproduction_rng = load_training_state(run, _model(model), optimizer,
            args.reproduction_metadata, scheduler=scheduler)
    else:
        loop = dict(epoch=0, step=0, global_step=0, epoch_rng=None, loss_sum=0., loss_steps=0,
                    best_pc=-1., best_epoch=None, history=[], complete=False, early_stopping=None)
    args.reproduction_loop = loop
    args.reproduction_progress = TrainingProgress(run/"best.pt", resume=continuing)
    args.reproduction_started = time.monotonic()
    if args.reproduction_rng is not None:
        restore_rng(args.reproduction_rng)
    save_training_state(run, _model(model), optimizer, args.reproduction_metadata, loop, scheduler=scheduler)
    return loop


def native_batches(args, loader, epoch):
    loop = args.reproduction_loop
    skip = loop["step"] if loop["epoch"]==epoch else 0
    if args.reproduction_rng is not None:
        if skip:
            restore_rng(loop["epoch_rng"])
            iterator = iter(loader)
            for _ in range(skip):
                next(iterator)
            restore_rng(args.reproduction_rng)
        else:
            restore_rng(args.reproduction_rng)
            loop["epoch_rng"] = rng_state()
            iterator = iter(loader)
        args.reproduction_rng = None
    else:
        loop["epoch_rng"] = rng_state()
        iterator = iter(loader)
    args.reproduction_epoch = epoch+1
    args.reproduction_started = time.monotonic()
    yield from enumerate(iterator, start=skip)


def native_step(args, model, optimizer, scheduler, epoch, step, loss):
    loop = args.reproduction_loop
    loop.update(epoch=epoch, step=step+1, global_step=loop["global_step"]+1,
                loss_sum=loop["loss_sum"]+float(loss.detach()), loss_steps=loop["loss_steps"]+1)
    if loop["global_step"] % args.reproduction_metadata["checkpoint_interval"] == 0:
        save_training_state(args.reproduction_run, _model(model), optimizer,
                            args.reproduction_metadata, loop, scheduler=scheduler)


def native_predictions(args, probabilities, labels):
    """Called after author forward; no Test labels or counterpart enter prediction."""
    meta = args.reproduction_metadata
    split = getattr(args, "reproduction_split", "valid")
    if split not in args.reproduction_cohorts:
        args.reproduction_cohorts[split] = [{k:r[k] for k in ("sample_key","pair_key","target")}
                                           for r in official_rows(meta["data_dir"], split)]
    rows = args.reproduction_cohorts[split]
    if list(map(int, labels)) != [r["target"] for r in rows]:
        raise ValueError("author predictions are missing, reordered or relabeled")
    values = list(map(float, probabilities))
    result = paired_metrics(rows, values)
    pred = [dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                 probability=p, prediction=int(p>.5)) for r,p in zip(rows, values)]
    args.reproduction_validation, args.reproduction_predictions = result, pred
    if getattr(args, "reproduction_evaluate", False):
        root = Path(args.output_dir)
        write_jsonl(root/f"{split}.predictions.jsonl", pred)
        write_json(root/f"{split}.metrics.json", dict(selected_epoch=args.reproduction_epoch,
            selection="full Valid P-C", metrics=result, source_only=True))
    return result


def native_epoch(args, model, optimizer, scheduler, epoch, early_stopping=None):
    loop, run = args.reproduction_loop, args.reproduction_run
    validation = args.reproduction_validation
    if validation["p_c"] > loop["best_pc"]:
        loop.update(best_pc=validation["p_c"], best_epoch=epoch+1)
        cp = dict(args.reproduction_metadata, selected_epoch=epoch+1, validation=validation,
                  state_dict=_model(model).state_dict())
        temporary = run/"best-p_c.pt.tmp"
        torch.save(cp, temporary)
        temporary.replace(run/"best-p_c.pt")
        write_json(run/"best-p_c.json", dict(epoch=epoch+1, validation=validation))
        write_jsonl(run/"valid.best-p_c.predictions.jsonl", args.reproduction_predictions)
    record = dict(event="epoch", epoch=epoch+1, method=args.reproduction_metadata["method"],
        train_loss=loop["loss_sum"]/loop["loss_steps"], optimizer_steps=loop["global_step"],
        validation_threshold=.5, validation=validation,
        training_seconds=time.monotonic()-args.reproduction_started)
    args.reproduction_progress.add(record)
    loop["history"].append(record)
    if early_stopping is not None:
        loop["early_stopping"] = {k:getattr(early_stopping,k) for k in ("counter","best_score","early_stop")}
    loop.update(epoch=epoch+1, step=0, epoch_rng=None, loss_sum=0., loss_steps=0)
    save_training_state(run, _model(model), optimizer, args.reproduction_metadata, loop, scheduler=scheduler)


def native_finish(args, model, optimizer, scheduler):
    loop = args.reproduction_loop
    loop["complete"] = True
    save_training_state(args.reproduction_run, _model(model), optimizer,
                        args.reproduction_metadata, loop, scheduler=scheduler)
    write_json(args.reproduction_run/"training.metrics.json", dict(args.reproduction_metadata,
        completed_epochs=loop["epoch"], optimizer_steps=loop["global_step"],
        selected_epochs=dict(p_c=loop["best_epoch"]),
        early_stopping=loop["early_stopping"], complete=True))


def _forward(args, model, batch):
    method = json.loads(Path(args.reproduction_config).read_text())["method"]
    b = [t.to(args.device) for t in batch]
    if method == "linevul":
        return model(input_ids=b[0], labels=b[1])
    if method == "svuld":
        return model(b[0], b[1], b[2])
    return model(b[0], b[1])


def native_smoke(args, model, tokenizer, dataset):
    """Actual author model, requested full micro-batch, gradient and state recovery."""
    meta = json.loads(Path(args.reproduction_config).read_text())
    run = Path(args.output_dir)
    model.to(args.device)
    dataset = Subset(dataset, range(meta["training_config"]["batch_size"]))
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    optimizer_class = LOSVERAdamW if meta["method"]=="regvd" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=meta["training_config"]["learning_rate"], weight_decay=0.)
    model.train()
    loss, _ = _forward(args, model, batch)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    if not torch.isfinite(loss) or norm<=0:
        raise ValueError("author model smoke has a nonfinite loss or zero gradient")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        _, before = _forward(args, model, batch)
    save_training_state(run, _model(model), optimizer, meta, dict(complete=True))
    parameter = next(p for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        parameter.add_(1.)
    _, state = load_training_state(run, _model(model), optimizer, meta)
    restore_rng(state)
    with torch.no_grad():
        _, after = _forward(args, model, batch)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    labels = [1,0]*(len(batch[0])//2)
    fake_rows = [dict(target=y,pair_key=f"smoke:{i//2}") for i,y in enumerate(labels)]
    metrics = paired_metrics(fake_rows, after[:,UPSTREAM[meta["method"]]["probability_column"]].cpu().tolist())
    write_json(run/"smoke.json",dict(method=meta["method"], upstream=meta["upstream"],
        functions=len(batch[0]), loss=float(loss.detach()), gradient_norm=norm,
        save_reload_equal=True, source_split="Train", metrics=metrics,
        maximum_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))


def native_evaluate(args, model, tokenizer, dataset_class, evaluate_function):
    args.reproduction_metadata = json.loads(Path(args.reproduction_config).read_text())
    args.reproduction_cohorts = {}
    cp = torch.load(Path(args.output_dir)/"best-p_c.pt",map_location="cpu",weights_only=False)
    _model(model).load_state_dict(cp["state_dict"],strict=True)
    args.reproduction_epoch = cp["selected_epoch"]
    model.to(args.device).eval()
    for split in ("valid","test"):
        args.reproduction_split = split
        method = cp["method"]
        if method == "linevul":
            dataset = dataset_class(tokenizer,args,file_type="eval" if split=="valid" else "test")
            evaluate_function(args,model,tokenizer,dataset)
        elif method == "svuld":
            evaluate_function(args,model,tokenizer,args.eval_data_file if split=="valid" else args.test_data_file)
        else:
            args.eval_data_file = str(Path(cp["export_dir"])/f"{split}.jsonl")
            evaluate_function(args,model,tokenizer)


def native_command(method, seed, output_dir, data_dir, *, resume=False, smoke=False, evaluate=False):
    spec = UPSTREAM[method]
    run = Path(output_dir)/("smoke" if smoke else f"seed{seed}")/"primevul_paired"/method
    run.mkdir(parents=True,exist_ok=True)
    export = (Path(data_dir)/"official_baselines"/method).resolve()
    model = Path("/home/PublicData/PHY-data/resource")/spec["model"]
    settings = {k:v for k,v in spec.items() if k not in ("repo","commit","directory","script","model")}
    settings.update(seed=seed, weight_decay=0., adam_epsilon=1e-8, gradient_clip=1., gradient_accumulation=1, fp16=False)
    meta = dict(protocol="author_official_primevul",method=method,model_path=str(model),
        upstream={k:spec[k] for k in ("repo","commit","script")}, training_config=settings,
        data_dir=str(Path(data_dir).resolve()), export_dir=str(export), checkpoint_interval=500,
        train_samples=8708,valid_samples=1124,test_samples=1128,
        training_members=[r["sample_key"] for r in official_rows(data_dir,"train")],
        validation_members=[r["sample_key"] for r in official_rows(data_dir,"valid")],
        selection="full Valid P-C; first strict maximum",decision_threshold=.5,comparison=">",
        environment=dict(torch=torch.__version__,python=sys.version.split()[0],cuda=torch.version.cuda))
    config = run/"config.json"
    if config.exists() and json.loads(config.read_text()) != meta:
        raise ValueError(f"{run}: existing configuration differs")
    write_json(config,meta)
    extension = "csv" if method=="linevul" else "jsonl"
    command=[sys.executable,"-u",str(Path("external")/spec["directory"]/spec["script"]),
        "--output_dir",str(run.resolve()),"--model_name_or_path",str(model),
        "--train_data_file",str(export/f"train.{extension}"),
        "--eval_data_file",str(export/f"valid.{extension}"),
        "--test_data_file",str(export/f"test.{extension}"),
        "--block_size",str(spec["block_size"]),"--train_batch_size",str(spec["batch_size"]),
        "--eval_batch_size",str(spec["batch_size"]),"--learning_rate",str(spec["learning_rate"]),
        "--max_grad_norm","1.0","--seed",str(seed),"--reproduction-config",str(config.resolve())]
    if method=="linevul":
        command += ["--model_type","roberta","--tokenizer_name",str(model),"--epochs",str(spec["epochs"]),"--evaluate_during_training"]
    elif method=="svuld":
        command += ["--num_train_epochs",str(spec["epochs"]),"--r_drop"]
    else:
        command += ["--model_type","roberta","--tokenizer_name",str(model),"--epoch",str(spec["epochs"]),
            "--gnn","ReGCN","--hidden_size","128","--num_GNN_layers","2","--format","uni","--window_size","5",
            "--evaluate_during_training"]
    if evaluate:
        command += ["--reproduction-evaluate"]
    else:
        command += ["--do_train"]
    if smoke:
        command += ["--reproduction-smoke"]
    if resume:
        command += ["--reproduction-resume"]
    return run, command


def run(method, seed, output_dir, data_dir, device, *, resume=False, smoke=False):
    if method not in UPSTREAM or seed not in SEEDS or device not in ("cuda:0","cuda:1"):
        raise ValueError("choose an official method, one of the three seeds and cuda:0/1")
    folder = Path(output_dir)/("smoke" if smoke else f"seed{seed}")/"primevul_paired"/method
    folder.mkdir(parents=True,exist_ok=True)
    # Multiple tmux queues can meet at a seed: wait for its owner, never double-train.
    with (folder/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        folder, command = native_command(method,seed,output_dir,data_dir,resume=resume,smoke=smoke)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=device.split(":")[1],OMP_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false",PYTHONPATH=str(Path.cwd()))
        if smoke or not (folder/"training.metrics.json").exists():
            with (folder/"native.local.log").open("a" if resume else "w") as handle:
                subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        if not smoke and not (folder/"test.metrics.json").exists():
            _, command = native_command(method,seed,output_dir,data_dir,evaluate=True)
            with (folder/"evaluate.local.log").open("w") as handle:
                subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        return json.loads((folder/("smoke.json" if smoke else "test.metrics.json")).read_text())


def report(output_dir, data_dir="data"):
    root=Path(output_dir)
    losver_reference=json.loads((LOSVER_REFERENCE/"losver"/"training.metrics.json").read_text())
    cohorts={s:[{k:r[k] for k in ("sample_key","pair_key","target")}
                for r in official_rows(data_dir,s)] for s in PAIRS}
    records=[]
    for method in (*UPSTREAM,"losver"):
        for seed in SEEDS:
            path = LOSVER_REFERENCE/"losver" if method=="losver" and seed==123456 else root/f"seed{seed}"/"primevul_paired"/method
            test_file=path/"test.metrics.json"
            if not test_file.is_file():
                records.append(dict(method=method,seed=seed,status="incomplete",run=str(path)))
                continue
            rows=cohorts["test"]
            pred=read_jsonl(path/"test.predictions.jsonl")
            if [(r["sample_key"],r["pair_key"],r["target"]) for r in rows] != [
                    (r["sample_key"],r["pair_key"],r["target"]) for r in pred]:
                raise ValueError(f"{path}: complete Test IDs/pairs/labels/order differ")
            if any(p["prediction"]!=int(p["probability"]>.5) for p in pred):
                raise ValueError(f"{path}: Test decisions are not strict probability > 0.5")
            saved=json.loads(test_file.read_text())
            value=paired_metrics(rows,[p["probability"] for p in pred])
            required = set(METRICS)-{"patched_fpr"}
            if not required.issubset(saved["metrics"]):
                raise ValueError(f"{path}: missing original classification metrics")
            if any(value[k]!=saved["metrics"][k] for k in METRICS if k in saved["metrics"]):
                raise ValueError(f"{path}: Test metrics do not reproduce")
            training=json.loads((path/"training.metrics.json").read_text())
            best=json.loads((path/"best-p_c.json").read_text())
            history=[r for r in read_jsonl(path/"best.training.jsonl") if r.get("event")=="epoch"]
            epochs=training["completed_epochs"]
            if [r["epoch"] for r in history]!=list(range(1,epochs+1)):
                raise ValueError(f"{path}: incomplete epoch history")
            if method in UPSTREAM:
                spec=UPSTREAM[method]
                if (training.get("complete") is not True or training["training_config"]["seed"]!=seed
                    or training["upstream"]["commit"]!=spec["commit"]
                    or training["training_members"]!=[r["sample_key"] for r in cohorts["train"]]
                    or training["validation_members"]!=[r["sample_key"] for r in cohorts["valid"]]
                    or training["optimizer_steps"]!=epochs*math.ceil(8708/spec["batch_size"])):
                    raise ValueError(f"{path}: original members, steps or provenance differ")
                if epochs!=spec["epochs"] and not (method=="svuld" and epochs<20
                        and training.get("early_stopping",{}).get("early_stop")):
                    raise ValueError(f"{path}: incomplete author training budget")
            elif (epochs!=16 or training["optimizer_steps"]!=17424
                  or training["training_config"]["seed"]!=seed or training["input_mode"]!="512"
                  or training["line_mapping"]!="source_offsets"):
                raise ValueError(f"{path}: not the complete formal LOSVER 512 protocol")
            if max(history,key=lambda r:r["validation"]["p_c"])["epoch"]!=best["epoch"] or saved["selected_epoch"]!=best["epoch"]:
                raise ValueError(f"{path}: not the full Valid P-C selection")
            valid_rows=cohorts["valid"]
            valid_pred=read_jsonl(path/"valid.best-p_c.predictions.jsonl")
            if [(r["sample_key"],r["pair_key"],r["target"]) for r in valid_rows] != [
                    (r["sample_key"],r["pair_key"],r["target"]) for r in valid_pred]:
                raise ValueError(f"{path}: selected Valid predictions do not cover all original pairs")
            valid=paired_metrics(valid_rows,[p["probability"] for p in valid_pred])
            if any(valid[k]!=best["validation"][k] for k in METRICS if k in best["validation"]):
                raise ValueError(f"{path}: selected Valid metrics do not reproduce")
            reload_difference=None
            if method in UPSTREAM:
                reloaded=read_jsonl(path/"valid.predictions.jsonl")
                if [(r["sample_key"],r["pair_key"],r["target"]) for r in valid_rows] != [
                        (r["sample_key"],r["pair_key"],r["target"]) for r in reloaded]:
                    raise ValueError(f"{path}: frozen Valid inference changes the cohort")
                reload_difference=max(abs(a["probability"]-b["probability"])
                                      for a,b in zip(valid_pred,reloaded))
                if reload_difference>1e-6 or any(a["prediction"]!=b["prediction"]
                                               for a,b in zip(valid_pred,reloaded)):
                    raise ValueError(f"{path}: frozen Valid inference differs from checkpoint selection")
            records.append(dict(method=method,seed=seed,status="complete",run=str(path),
                completed_epochs=training["completed_epochs"],selected_epoch=best["epoch"],
                validation={k:valid[k] for k in METRICS},test={k:value[k] for k in METRICS},
                valid_reload_max_probability_difference=reload_difference,
                metrics_recomputed_from_predictions=[k for k in METRICS if k not in saved["metrics"]]))
    aggregate={split:{m:{k:dict(mean=statistics.mean(r[split][k] for r in records if r["method"]==m),
        std=statistics.stdev(r[split][k] for r in records if r["method"]==m)) for k in METRICS}
        for m in (*UPSTREAM,"losver") if sum(r["method"]==m and r["status"]=="complete" for r in records)==3}
        for split in ("validation","test")}
    result=dict(protocol=dict(dataset="PrimeVul official paired",pairs=PAIRS,seeds=list(SEEDS),
        selection="full Valid P-C",threshold=.5,comparison=">",std="sample SD, ddof=1, n=3 seeds"),
        upstream=dict(UPSTREAM,losver=dict(repo="https://github.com/waroad/losver",
            commit=losver_reference["upstream_commit"],
            implementation="existing formal 512 source_offsets with linear function classifier",
            training_config=losver_reference["training_config"])),runs=records,aggregate=aggregate,
        environment=dict(python=sys.version.split()[0],cuda=torch.version.cuda,
            packages={p:version(p) for p in ("torch","transformers","numpy","scipy","pandas","scikit-learn")}))
    write_json(root/"comparison.json",result)
    lines=["# 官方漏洞检测基线：PrimeVul 三种子复现","",
        "LineVul、SVulD、ReGVD保持作者模型、Tokenizer、损失、采样、优化器和原始最大训练预算；"
        "LOSVER复用本仓库已有正式512实现。统一按完整Valid P-C选模，严格prob>0.5。",
        "每组Test必须含564对/1128函数；Test不参与训练、早停或选择。","",
        "表内指标均为0–1数值；P-C乘100为百分比。PR-AUC沿用现有average_precision_score口径。"
        "历史LOSVER未保存FPR字段时，从全部564个修复函数的既有概率按严格>0.5重算，不修改历史文件。","",
        "| 方法 | Seed | 实际Epoch | 选中Epoch | Test P-C | MCC | F1 | AUC | PR-AUC | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if len(aggregate["test"])==4:
        ordered=sorted(aggregate["test"],key=lambda m:aggregate["test"][m]["p_c"]["mean"],reverse=True)
        mcc_best=max(aggregate["test"],key=lambda m:aggregate["test"][m]["mcc"]["mean"])
        f1_best=max(aggregate["test"],key=lambda m:aggregate["test"][m]["f1"]["mean"])
        svuld_epochs="/".join(str(r["completed_epochs"]) for r in records if r["method"]=="svuld")
        pc_text="；".join(f"{m} {100*aggregate['test'][m]['p_c']['mean']:.2f} ± "
                           f"{100*aggregate['test'][m]['p_c']['std']:.2f}%" for m in ordered)
        lines[2:2]=["四种方法×三个种子的12组正式评价全部完成：11组新跑，LOSVER Seed123456复用。",
            "Test P-C三种子均值±样本标准差："+pc_text+"。",
            f"P-C均值最高为{ordered[0]}，MCC均值最高为{mcc_best}。F1均值最高为{f1_best}，"
            f"其修复侧FPR为{100*aggregate['test'][f1_best]['patched_fpr']['mean']:.2f}%；"
            "各指标反映了不同的检出与误报取舍。",
            f"SVulD实际完成Epoch（Seed42/1234/123456）为{svuld_epochs}，采用作者的Valid损失早停，"
            "最大预算20轮；最优权重仍按完整Valid P-C选择。",""]
    for r in records:
        if r["status"]=="complete":
            lines.append(f"| {r['method']} | {r['seed']} | {r['completed_epochs']} | {r['selected_epoch']} | "+" | ".join(f"{r['test'][k]:.6f}" for k in METRICS)+" |")
        else:
            lines.append(f"| {r['method']} | {r['seed']} | 未完成 | — | — | — | — | — | — | — | — |")
    lines += ["","三种子均值±样本标准差：","","| 方法 | P-C | MCC | F1 | AUC | PR-AUC | Recall | FPR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m,a in aggregate["test"].items():
        lines.append(f"| {m} | "+" | ".join(f"{a[k]['mean']:.6f} ± {a[k]['std']:.6f}" for k in METRICS)+" |")
    lines += ["","## 所选checkpoint的完整Valid结果","",
        "| 方法 | Seed | P-C | MCC | F1 | AUC | PR-AUC | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["status"]=="complete":
            lines.append(f"| {r['method']} | {r['seed']} | "+" | ".join(f"{r['validation'][k]:.6f}" for k in METRICS)+" |")
    lines += ["","Valid三种子均值±样本标准差：","",
        "| 方法 | P-C | MCC | F1 | AUC | PR-AUC | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m,a in aggregate["validation"].items():
        lines.append(f"| {m} | "+" | ".join(f"{a[k]['mean']:.6f} ± {a[k]['std']:.6f}" for k in METRICS)+" |")
    lines += ["","SVulD沿用作者README及论文RQ1的R-Drop：同函数两次Dropout视图，0.5×双CE+0.1×对称KL求和，"
        "顺序采样，最多20 Epoch、Valid损失早停patience5；未改为普通分类器，也未擅自开启SimCSE/SimCT消融。",
        "ReGVD保留400 Token完整ID序列构图（包括特殊Token/Padding）、固定词嵌入、双输出及第0列概率等作者行为，未修正算法。",
        "LOSVER Seed123456复用results/losver_512的source_offsets正式512实现；早期results/primevul_paired缺少该协议，未混用。",
        "LOSVER沿用40轮定位器（Valid行F1选模）和16轮预测行检测器（完整Valid P-C选模），不使用Oracle。",
        "原始源码、导出的CSV/JSONL、上游checkout、含Token日志与全部权重留在本地。","",
        "## 上游版本与配置",""]
    for m,s in UPSTREAM.items():
        lines += [f"- {m}: [{s['repo']}]({s['repo'].removesuffix('.git')}), commit `{s['commit']}`; "
                  f"{s['epochs']} Epoch最大预算，Batch{s['batch_size']}，Token{s['block_size']}，lr={s['learning_rate']}。"]
    lines += [f"- losver: [作者源码](https://github.com/waroad/losver), 历史元数据源版本 "
              f"`{losver_reference['upstream_commit']}`；沿用已确认的source_offsets/512输入和线性函数分类头。"
              "这些已有实现修正与历史Seed123456一致，本轮未改动；不称为未经适配的论文数值复现。"]
    lines += ["","必要修改见vulnmechanism/upstream/*.patch：数据/固定评价/已有断点续训接口，"
        "可选解释和日志依赖延后加载，SciPy及NumPy符号更新，ReGVD旧Transformers AdamW使用仓库同一更新规则。",
        "核心依赖未降级。正式训练由tmux管理，原作者前向、反向和优化步顺序保留。",""]
    lines += ["环境：Python "+result["environment"]["python"]+"，CUDA "+str(torch.version.cuda)+"；"
              +"，".join(f"{p} {v}" for p,v in result["environment"]["packages"].items())+"。",""]
    lines += ["## 复现与验证","","```bash",
        "python -m vulnmechanism.cli official-baselines prepare",
        "python -m vulnmechanism.cli official-baselines smoke --method linevul --seed 42 --device cuda:0",
        "python -m vulnmechanism.cli official-baselines run --method linevul --seed 42 --device cuda:0 --resume",
        "python -m vulnmechanism.cli official-baselines report","```","",
        "method分别选择linevul、svuld、regvd，seed分别选择42、1234、123456。每个作者训练循环使用相同的正式接口，"
        "组合级文件锁防止多个tmux队列重复训练；恢复复用原项目模型/优化器/调度器/RNG原子保存与日志截断。",
        "测试包括实际数据成员/标签/源码导出对齐、作者R-Drop原始损失、随机采样+Dropout的中断恢复、"
        "P-C为0仍保存第一个checkpoint且同分保留更早Epoch；记录见official-tests.log和losver-tests.log。",
        "真实GPU smoke分别记录在smoke/primevul_paired/<method>/smoke.json，均为Train诊断，不是正式性能。",
        "SVulD的同函数双Dropout及对称KL求和按公开代码执行；contrast字段仅在Train保留真实配对，"
        "正式R-Drop分支不使用该字段，Valid/Test将其设为函数自身。未补入额外InfoNCE或跨版本损失。",""]
    lines += ["真实GPU检查（均来自Train，只用于检查计算和恢复）：","",
        "| 方法/阶段 | 函数数 | Loss | 裁剪前梯度范数 | 峰值分配显存/GiB | 恢复输出一致 |",
        "|---|---:|---:|---:|---:|---|"]
    for method in UPSTREAM:
        p=root/"smoke"/"primevul_paired"/method/"smoke.json"
        if p.is_file():
            s=json.loads(p.read_text())
            lines.append(f"| {method} | {s['functions']} | {s['loss']:.6f} | {s['gradient_norm']:.6f} | "
                         f"{s['maximum_memory_bytes']/2**30:.3f} | {s['save_reload_equal']} |")
    p=root/"losver-smoke.json"
    if p.is_file():
        for phase,s in json.loads(p.read_text()).items():
            lines.append(f"| losver/{phase} | {s['functions']} | {s['loss']:.6f} | "
                         f"{s['gradient_norm']:.6f} | — | {s['save_reload_equal']} |")
    lines += ["","6项新增基线测试与34项既有LOSVER/恢复测试通过。"
        "恢复测试覆盖作者使用的0/4 worker、随机/顺序采样、Dropout、最终参数及RNG一致性。",
        "实际检查中，最初的stdlib csv读取触发131072字段上限，改用作者的pandas读取器后全量源码对齐通过；"
        "沙箱的本地多进程通信权限不足，相关检查改在同一Conda环境的宿主执行并通过。",
        "I/O使用周期/轮末原子last.pt和Valid改善时的最优权重覆盖保存；评价元数据首次读取后缓存，"
        "不会逐轮累积大权重文件。续跑入口为同一run命令的--resume。",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    with (root/"comparison.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("method","seed","status",*METRICS))
        for r in records:
            writer.writerow((r["method"],r["seed"],r["status"],*[r.get("test",{}).get(k) for k in METRICS]))
    return result
