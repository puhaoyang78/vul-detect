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
import tempfile
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
    "vulberta": dict(repo="https://github.com/ICL-ml4csec/VulBERTa.git",
        commit="75ab355bf8be124751084c0ff3563df24f48fe38", directory="VulBERTa",
        script="Finetuning_VulBERTa-MLP.ipynb", model="VulBERTa", epochs=10, batch_size=4,
        block_size=1024, learning_rate=3e-5, warmup_fraction=0., optimizer="Transformers AdamW rule",
        probability_column=1, sampler="RandomSampler", fp16=True, weight_decay=0., gradient_clip=1.),
    "vulmixer": dict(repo="https://github.com/mlavd/vulmixer.git",
        commit="090c8fadda530027b6805b1931967b34d2015ed0", directory="VulMixer",
        script="run.py", model="graphcodebert-base", epochs=20, batch_size=128,
        block_size=1024, learning_rate=5e-4, warmup_fraction=0., optimizer="torch.optim.AdamW",
        probability_column=1, sampler="RandomSampler", fp16=True, weight_decay=.01, gradient_clip=None),
}
PAIRS = dict(train=4354, valid=562, test=564)
METRICS = ("p_c", "mcc", "f1", "auc", "auprc", "recall", "patched_fpr")
LOSVER_REFERENCE = Path("results/losver_512/primevul_paired")
VULBERTA_MODEL = Path("/home/PublicData/PHY-data/resource/VulBERTa")


def _local_lock_path(path):
    """Single-host tmux jobs use local locks; GlusterFS flock can block readers."""
    root=Path(tempfile.gettempdir())/f"vul-detect-official-{os.getuid()}"
    root.mkdir(mode=0o700,exist_ok=True)
    if root.is_symlink() or root.stat().st_uid!=os.getuid():
        raise PermissionError(f"{root}: local lock directory must belong to this user")
    local=root.joinpath(*Path(path).resolve().parts[1:])
    local.parent.mkdir(parents=True,exist_ok=True)
    return local


def official_rows(data_dir, split):
    rows = read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.jsonl")
    if (len(rows) != 2*PAIRS[split] or len({r["sample_key"] for r in rows}) != len(rows)
        or len({r["pair_key"] for r in rows}) != PAIRS[split]
        or [r["target"] for r in rows] != [1, 0]*PAIRS[split]
        or any(rows[i]["pair_key"] != rows[i+1]["pair_key"] for i in range(0,len(rows),2))):
        raise ValueError(f"{split}: complete original PrimeVul repair pairs required")
    return rows


def prepare(data_dir="data", methods=None):
    """Idempotent pinned checkout plus minimal compatibility patches and exports."""
    methods = tuple(UPSTREAM) if methods is None else tuple(methods)
    for method in methods:
        spec = UPSTREAM[method]
        repo = Path("external")/spec["directory"]
        sparse = ("Source_Code/SVulD" if method=="svuld" else "linevul" if method=="linevul" else "code")
        if not repo.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repo"], str(repo)], check=True)
            if method not in ("vulberta", "vulmixer"):
                subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", sparse], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", spec["commit"]], check=True)
        actual = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        if actual != spec["commit"]:
            raise ValueError(f"{repo}: expected author commit {spec['commit']}, got {actual}")
        patch = Path(__file__).with_name("upstream")/f"{method}.patch"
        if not patch.exists() and method == "vulberta":
            continue  # The original notebook cells are loaded directly, without a copied trainer.
        applied = subprocess.run(["git","-C",str(repo),"apply","--reverse","--check",str(patch.resolve())], capture_output=True)
        if applied.returncode:
            subprocess.run(["git","-C",str(repo),"apply","--check",str(patch.resolve())],check=True)
            subprocess.run(["git","-C",str(repo),"apply",str(patch.resolve())],check=True)
    export_data(data_dir, methods)
    return {m:dict(repo=UPSTREAM[m]["repo"],commit=UPSTREAM[m]["commit"]) for m in methods}


def export_data(data_dir, methods=None):
    """Format adaptation only; originals, labels, member order and source stay intact."""
    for split in PAIRS:
        rows = official_rows(data_dir, split)
        for method in (UPSTREAM if methods is None else methods):
            folder = Path(data_dir)/"official_baselines"/method
            folder.mkdir(parents=True, exist_ok=True)
            if method=="vulberta":
                continue  # The notebook adapter reads the original members and caches encodings.
            output = folder/f"{split}.{'csv' if method=='linevul' else 'jsonl'}"
            if output.exists():
                continue
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
                    elif method == "vulmixer":
                        # Author preprocessing: clang-format with the supplied configuration.
                        formatted = subprocess.run([str(Path(sys.executable).with_name("clang-format")),
                            "-style=file:external/VulMixer/config/clang-format.yml"],
                            input=row["func"], text=True, capture_output=True, check=True, timeout=15).stdout
                        exported.append(dict(common, func=formatted, target=row["target"], idx=i))
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


def native_epoch(args, model, optimizer, scheduler, epoch, early_stopping=None, *, save_state=True):
    loop, run = args.reproduction_loop, args.reproduction_run
    validation = args.reproduction_validation
    if validation["p_c"] > loop["best_pc"]:
        loop.update(best_pc=validation["p_c"], best_epoch=epoch+1)
        cp = dict(args.reproduction_metadata, selected_epoch=epoch+1, validation=validation,
                  state_dict=_model(model).state_dict())
        if hasattr(args, "reproduction_validation_rng"):
            cp["validation_rng"] = args.reproduction_validation_rng
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
    if save_state:
        save_training_state(run, _model(model), optimizer, args.reproduction_metadata, loop, scheduler=scheduler)


def native_finish(args, model, optimizer, scheduler, *, save_state=True):
    loop = args.reproduction_loop
    loop["complete"] = True
    if save_state:
        save_training_state(args.reproduction_run, _model(model), optimizer,
                            args.reproduction_metadata, loop, scheduler=scheduler)
    write_json(args.reproduction_run/"training.metrics.json", dict(args.reproduction_metadata,
        completed_epochs=loop["epoch"], optimizer_steps=loop["global_step"],
        **({"optimizer_updates":loop["optimizer_updates"],
            "amp_skipped_updates":loop["global_step"]-loop["optimizer_updates"]} if "optimizer_updates" in loop else {}),
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


def native_smoke(args, model, tokenizer, dataset, *, forward=None, optimizer=None):
    """Actual author model, requested full micro-batch, gradient and state recovery."""
    meta = json.loads(Path(args.reproduction_config).read_text())
    run = Path(args.output_dir)
    model.to(args.device)
    dataset = Subset(dataset, range(meta["training_config"]["batch_size"]))
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    optimizer_class = LOSVERAdamW if meta["method"] in ("regvd","vulberta") else torch.optim.AdamW
    optimizer = optimizer or optimizer_class(model.parameters(), lr=meta["training_config"]["learning_rate"], weight_decay=0.)
    forward = forward or (lambda batch: _forward(args, model, batch))
    mixed = meta["training_config"].get("fp16",False)
    scaler = torch.amp.GradScaler("cuda",enabled=mixed)
    model.train()
    attempts=[]
    for attempt in range(16 if mixed else 1):
        optimizer.zero_grad()
        scale=scaler.get_scale()
        with torch.autocast("cuda",dtype=torch.float16,enabled=mixed):
            loss, _ = forward(batch)
        if not torch.isfinite(loss):
            raise ValueError("author model smoke has a nonfinite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if not finite:
            if not mixed: raise ValueError("author model smoke has nonfinite gradients without AMP")
            # Same automatic overflow handling as author mixed-precision training:
            # GradScaler skips the update and lowers its scale. Never clip Inf/NaN.
            scaler.step(optimizer);scaler.update()
            attempts.append(dict(loss=float(loss.detach()),scale=scale,finite_gradient=False,
                                 next_scale=scaler.get_scale(),optimizer_updated=False))
            continue
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
            meta["training_config"].get("gradient_clip",1.) or float("inf"),error_if_nonfinite=True))
        if norm<=0: raise ValueError("author model smoke has zero gradient")
        parameter = next(p for p in model.parameters() if p.grad is not None and bool(p.grad.abs().any()))
        initial = parameter.detach().clone()
        scaler.step(optimizer);scaler.update()
        attempts.append(dict(loss=float(loss.detach()),scale=scale,finite_gradient=True,optimizer_updated=True))
        break
    else:
        write_json(run/"smoke.failure.json",dict(amp_attempts=attempts))
        raise ValueError("No finite gradient/update after 16 native AMP scale adjustments")
    update = float((parameter.detach()-initial).abs().max())
    if update == 0:
        raise ValueError("author model smoke optimizer did not change parameters")
    model.eval()
    eval_rng = rng_state()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16,enabled=mixed):
        _, before = forward(batch)
    save_training_state(run, _model(model), optimizer, meta, dict(complete=True))
    with torch.no_grad():
        parameter.add_(1.)
    _, state = load_training_state(run, _model(model), optimizer, meta)
    restore_rng(eval_rng)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16,enabled=mixed):
        _, after = forward(batch)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    count = len(batch["target"] if "target" in batch else batch["labels"]) if isinstance(batch,dict) else len(batch[0])
    labels = [1,0]*(count//2)
    fake_rows = [dict(target=y,pair_key=f"smoke:{i//2}") for i,y in enumerate(labels)]
    metrics = paired_metrics(fake_rows, after[:,UPSTREAM[meta["method"]]["probability_column"]].cpu().tolist())
    write_json(run/"smoke.json",dict(method=meta["method"], upstream=meta["upstream"],
        functions=count, loss=float(loss.detach()), gradient_norm=norm,maximum_parameter_update=update,
        amp_attempts=attempts,
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
    model = VULBERTA_MODEL if method=="vulberta" else Path("/home/PublicData/PHY-data/resource")/spec["model"]
    if method=="vulberta" and (not (model/"config.json").is_file()
            or not any((model/f).is_file() for f in ("pytorch_model.bin","model.safetensors"))):
        raise FileNotFoundError(f"Author VulBERTa pretrained weights unavailable at {model}; official OneDrive link required, no substitute allowed")
    settings = {k:v for k,v in spec.items() if k not in ("repo","commit","directory","script","model")}
    settings.update(seed=seed, weight_decay=0., adam_epsilon=1e-8, gradient_clip=1., gradient_accumulation=1, fp16=False)
    if method in ("vulberta","vulmixer"):
        settings.update({k:spec[k] for k in ("fp16","weight_decay","gradient_clip")})
    meta = dict(protocol="author_official_primevul",method=method,model_path=str(model),
        upstream={k:spec[k] for k in ("repo","commit","script")}, training_config=settings,
        data_dir=str(Path(data_dir).resolve()), export_dir=str(export), checkpoint_interval=500,
        train_samples=8708,valid_samples=1124,test_samples=1128,
        training_members=[r["sample_key"] for r in official_rows(data_dir,"train")],
        validation_members=[r["sample_key"] for r in official_rows(data_dir,"valid")],
        selection="full Valid P-C; first strict maximum",decision_threshold=.5,comparison=">",
        environment=dict(torch=torch.__version__,python=sys.version.split()[0],cuda=torch.version.cuda))
    if method=="vulmixer":
        meta["embedding_cache"] = str((Path(data_dir)/"official_baselines/vulmixer/graphcodebert-base.embeddings").resolve())
        meta["preprocessing"] = "author clang-format config; GraphCodeBERT static word embeddings; zero embedding padding"
        meta["framework"] = "lightning.pytorch; author CurriculumLearning and configure_optimizers; mixed FP16"
        meta["checkpoint_interval"] = "epoch; native Lightning optimizer/scaler/loop plus RNG and log positions"
    elif method=="vulberta":
        meta["preprocessing"] = "author cleaner and Clang+BPE tokenizer; notebook cells"
        meta["framework"] = "author MyTrainer/RobertaForSequenceClassification; original HF AdamW rule; mixed FP16"
        meta["checkpoint_interval"] = "500 steps; native Trainer model/optimizer/scheduler/scaler/RNG and log positions"
        meta["pretrained_source"] = "https://1drv.ms/u/s!AueKnGqzBuIVkq4CynZHsF8Mv-en1g?e=3gg60p"
        meta["pretrained_configuration"] = json.loads((model/"config.json").read_text())
        meta["notebook_cells"] = [6,10,11,12,15,27,29,30,32,33,34]
        meta["training_config"].update(eval_batch_size=8,scheduler="linear",class_weight="N/(2*Nc), from Train")
    config = run/"config.json"
    if config.exists() and json.loads(config.read_text()) != meta:
        raise ValueError(f"{run}: existing configuration differs")
    write_json(config,meta)
    if method=="vulberta":
        command=[sys.executable,"-u","-c",
            "from vulnmechanism.official_baselines import vulberta_native; vulberta_native()",
            "--reproduction-config",str(config.resolve())]
        for enabled,flag in ((resume,"resume"),(smoke,"smoke"),(evaluate,"evaluate")):
            if enabled: command.append(f"--reproduction-{flag}")
        return run,command
    if method=="vulmixer":
        command = [sys.executable,"-u",str((Path("external/VulMixer/run.py")).resolve()),
            "--data","primevul","--model","vulmixer","--log","None",
            "--reproduction-config",str(config.resolve())]
        if smoke: command.append("--reproduction-smoke")
        elif evaluate: command.append("--reproduction-evaluate")
        else: command.append("--do_train")
        if resume: command.append("--reproduction-resume")
        return run,command
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
    with _local_lock_path(folder/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        folder, command = native_command(method,seed,output_dir,data_dir,resume=resume,smoke=smoke)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=device.split(":")[1],OMP_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false",PYTHONPATH=str(Path.cwd()))
        cwd = str(Path("external/VulMixer").resolve()) if method=="vulmixer" else None
        if method=="vulmixer": env["VULMIXER_EMBEDDING_CACHE"]=json.loads((folder/"config.json").read_text())["embedding_cache"]
        if method=="vulberta": env.update(NCCL_P2P_DISABLE="1",NCCL_IB_DISABLE="1")
        if smoke or not (folder/"training.metrics.json").exists():
            with (folder/"native.local.log").open("a" if resume else "w") as handle:
                subprocess.run(command,env=env,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        if not smoke and not (folder/"test.metrics.json").exists():
            _, command = native_command(method,seed,output_dir,data_dir,evaluate=True)
            with (folder/"evaluate.local.log").open("w") as handle:
                subprocess.run(command,env=env,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        return json.loads((folder/("smoke.json" if smoke else "test.metrics.json")).read_text())


def mixer_embedding_cache(config):
    """Run the author's cache script once; the cache is a static embedding table."""
    cache=Path(config["embedding_cache"])
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        with _local_lock_path(cache.with_suffix(".lock")).open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not cache.exists():
                subprocess.run([sys.executable,"scripts/cache_embeddings.py",
                    "--checkpoint",config["model_path"],"--cache_dir",str(cache.parent)],check=True)
    embedding=torch.load(cache,map_location="cpu",weights_only=False)
    if tuple(embedding.weight.shape)!=(50265,768):
        raise ValueError("Vul-Mixer needs the author's GraphCodeBERT word embedding table")
    return dict(file=str(cache),shape=list(embedding.weight.shape),
                bytes=cache.stat().st_size,contextual_encoder_forward=False)


def mixer_smoke(args, module, data_module):
    """Reuse the original shared_step, loss and optimizer, including mixed FP16."""
    def forward(batch):
        result=module.shared_step(batch,[])
        return result["loss"],result["all"].softmax(-1)
    args.output_dir=args.reproduction_run
    args.device=module.on_device
    module.calculate_metrics=False
    native_smoke(args,module,None,data_module.train_dataset,
                 forward=forward,optimizer=module.configure_optimizers())


def mixer_callback(args):
    """Attach evaluation/resume bookkeeping to the unchanged Lightning train loop."""
    from lightning.pytorch import Callback

    class OfficialMixerCallback(Callback):
        def __init__(self):
            args.reproduction_cohorts={}
            args.reproduction_loop=dict(epoch=0,step=0,global_step=0,epoch_rng=None,
                loss_sum=0.,loss_steps=0,best_pc=-1.,best_epoch=None,history=[],
                complete=False,early_stopping=None)
            self.saved_rng=None
            self.probabilities=[]
            self.labels=[]
            self.indices=[]

        def on_load_checkpoint(self,trainer,module,checkpoint):
            saved=checkpoint["official_reproduction"]
            if saved["metadata"]!=args.reproduction_metadata:
                raise ValueError("Vul-Mixer resume configuration or original members differ")
            args.reproduction_loop=saved["loop"]
            self.saved_rng=saved["rng"]
            log=args.reproduction_run/"best.training.jsonl"
            if log.exists():
                with log.open("r+b") as handle:
                    handle.truncate(saved["log_position"])

        def on_save_checkpoint(self,trainer,module,checkpoint):
            checkpoint["official_reproduction"]=dict(metadata=args.reproduction_metadata,
                loop=args.reproduction_loop,rng=rng_state(),
                log_position=(args.reproduction_run/"best.training.jsonl").stat().st_size)

        def on_fit_start(self,trainer,module):
            # Lightning creates/prefetches the shuffled iterator before on_train_start.
            # Restore earlier, before iterator creation; sanity validation isolates RNG.
            if self.saved_rng is not None: restore_rng(self.saved_rng)

        def on_train_start(self,trainer,module):
            args.reproduction_loop.setdefault("optimizer_updates",0)
            def updated(optimizer,step_args,step_kwargs):
                args.reproduction_loop["optimizer_updates"]+=1
            trainer.optimizers[0].register_step_post_hook(updated)

        def on_train_epoch_start(self,trainer,module):
            args.reproduction_started=time.monotonic()

        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            loop=args.reproduction_loop
            loop["loss_sum"]+=float(outputs["loss"].detach())
            loop["loss_steps"]+=1
            loop["global_step"]=trainer.global_step

        def start_evaluation(self,trainer):
            self.probabilities=[];self.labels=[];self.indices=[]
            if args.reproduction_evaluate and hasattr(args,"reproduction_reload_rng"):
                restore_rng(args.reproduction_reload_rng)
                del args.reproduction_reload_rng
            args.reproduction_validation_rng=rng_state()

        def on_validation_epoch_start(self,trainer,module):
            self.start_evaluation(trainer)

        def on_test_epoch_start(self,trainer,module):
            self.start_evaluation(trainer)

        def collect(self,outputs,batch):
            self.probabilities.extend(outputs["all"].softmax(-1)[:,1].detach().float().cpu().tolist())
            self.labels.extend(batch["target"].tolist())
            self.indices.extend(batch["index"].tolist())

        def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
            if not trainer.sanity_checking: self.collect(outputs,batch)

        def on_test_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
            # The author test_step already applies softmax to results['all'].
            self.probabilities.extend(outputs["all"][:,1].detach().float().cpu().tolist())
            self.labels.extend(batch["target"].tolist())
            self.indices.extend(batch["index"].tolist())

        def on_validation_epoch_end(self,trainer,module):
            if trainer.sanity_checking: return
            if self.indices!=list(range(len(self.probabilities))):
                raise ValueError("Vul-Mixer Valid source indices were reordered or duplicated")
            args.reproduction_validation=native_predictions(args,self.probabilities,self.labels)
            if not args.reproduction_evaluate:
                native_epoch(args,module.model,trainer.optimizers[0],None,trainer.current_epoch,save_state=False)

        def on_test_epoch_end(self,trainer,module):
            if self.indices!=list(range(len(self.probabilities))):
                raise ValueError("Vul-Mixer Test source indices were reordered or duplicated")
            native_predictions(args,self.probabilities,self.labels)

        def on_train_epoch_end(self,trainer,module):
            checkpoint=args.reproduction_run/"framework-last.ckpt"
            temporary=checkpoint.with_suffix(".ckpt.tmp")
            trainer.save_checkpoint(temporary)
            temporary.replace(checkpoint)

    return OfficialMixerCallback()


def mixer_evaluate(args,module,trainer,data_module):
    checkpoint=torch.load(args.reproduction_run/"best-p_c.pt",map_location="cpu",weights_only=False)
    module.model.load_state_dict(checkpoint["state_dict"],strict=True)
    args.reproduction_epoch=checkpoint["selected_epoch"]
    args.reproduction_reload_rng=checkpoint["validation_rng"]
    args.reproduction_split="valid"
    trainer.validate(module,data_module)
    args.reproduction_split="test"
    trainer.test(module,data_module)


def vulberta_notebook_components():
    """Load author definitions from their notebook, with only path/API adaptation."""
    import re
    import numpy as np
    from clang import cindex
    from torch.utils.data import Dataset
    root=Path("external/VulBERTa").resolve()
    book=json.loads((root/"Finetuning_VulBERTa-MLP.ipynb").read_text())
    namespace=dict(torch=torch,re=re,cindex=cindex,Dataset=Dataset,np=np)
    for index in (6,10,11,12,15):
        source="".join(book["cells"][index]["source"])
        source=source.replace("./tokenizer/",str(root/"tokenizer")+"/")
        # Removed unused symbol in tokenizers 0.22; the actual normalizers are unchanged.
        source=source.replace("StripAccents, unicode_normalizer_from_str, Replace","StripAccents, Replace")
        exec(compile(source,f"{root.name}/Finetuning_VulBERTa-MLP.ipynb:cell{index}","exec"),namespace)
    return namespace,book


def vulberta_native():
    """Execute author notebook components through its original Hugging Face Trainer."""
    import argparse
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (RobertaForSequenceClassification,Trainer,TrainingArguments,
                              TrainerCallback,set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    parser=argparse.ArgumentParser()
    parser.add_argument("--reproduction-config",required=True)
    for flag in ("resume","smoke","evaluate"):
        parser.add_argument(f"--reproduction-{flag}",action="store_true")
    args=parser.parse_args()
    meta=args.reproduction_metadata=json.loads(Path(args.reproduction_config).read_text())
    run=args.reproduction_run=Path(args.reproduction_config).parent
    args.output_dir=str(run);args.device=torch.device("cuda")
    args.reproduction_cohorts={}
    args.reproduction_split="valid"
    args.reproduction_loop=dict(epoch=0,step=0,global_step=0,epoch_rng=None,loss_sum=0.,
        loss_steps=0,best_pc=-1.,best_epoch=None,history=[],complete=False,early_stopping=None)
    seed=meta["training_config"]["seed"]
    set_seed(seed)
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    namespace,book=vulberta_notebook_components()
    datasets={}
    for split in (("valid","test") if args.reproduction_evaluate else ("train","valid")):
        cache=Path(meta["export_dir"])/f"{split}.encodings.pt"
        cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():
            encoded=torch.load(cache,weights_only=False)
        else:
            with _local_lock_path(cache.with_suffix(".lock")).open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if cache.exists():
                    encoded=torch.load(cache,weights_only=False)
                else:
                    rows=official_rows(meta["data_dir"],split)
                    encodings=namespace["my_tokenizer"].encode_batch([namespace["cleaner"](r["func"]) for r in rows])
                    encoded=dict(encodings=namespace["process_encodings"](encodings),labels=[r["target"] for r in rows],
                        members=[r["sample_key"] for r in rows],upstream_commit=meta["upstream"]["commit"],block_size=1024)
                    temporary=cache.with_suffix(".pt.tmp");torch.save(encoded,temporary);temporary.replace(cache)
        if encoded["upstream_commit"]!=meta["upstream"]["commit"] or encoded["block_size"]!=1024:
            raise ValueError("VulBERTa encoding cache version differs")
        if encoded["members"]!=[r["sample_key"] for r in official_rows(meta["data_dir"],split)]:
            raise ValueError("VulBERTa encoding cache changes original PrimeVul members")
        datasets[split]=namespace["MyCustomDataset"](encoded["encodings"],encoded["labels"])
    model=RobertaForSequenceClassification.from_pretrained(meta["model_path"])
    labels=[r["target"] for r in official_rows(meta["data_dir"],"train")]
    weights=compute_class_weight(class_weight="balanced",classes=np.array([0,1]),y=labels)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32)).to(args.device)
    namespace.update(Trainer=Trainer,criterion=criterion,device=args.device,
                     c_weights=torch.tensor(weights,dtype=torch.float32))
    source="".join(book["cells"][30]["source"]).replace("def compute_loss(self, model, inputs, return_outputs=False):",
        "def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):")
    exec(compile(source,"VulBERTa/Finetuning_VulBERTa-MLP.ipynb:cell30","exec"),namespace)

    class RecordedTrainer(namespace["MyTrainer"]):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            result=super().compute_loss(model,inputs,return_outputs=return_outputs)
            if model.training:
                loss=result[0] if return_outputs else result
                args.reproduction_loop["loss_sum"]+=float(loss.detach())
                args.reproduction_loop["loss_steps"]+=1
            return result

    class RecordedCallback(TrainerCallback):
        def on_epoch_begin(self,hf_args,state,control,**kwargs):
            args.reproduction_started=time.monotonic()
        def on_evaluate(self,hf_args,state,control,model=None,optimizer=None,lr_scheduler=None,**kwargs):
            args.reproduction_loop["global_step"]=state.global_step
            native_epoch(args,model,optimizer,lr_scheduler,round(state.epoch)-1,save_state=False)
        def on_save(self,hf_args,state,control,**kwargs):
            checkpoint=Path(hf_args.output_dir)/f"checkpoint-{state.global_step}"
            torch.save(dict(metadata=meta,loop=args.reproduction_loop,
                log_position=(run/"best.training.jsonl").stat().st_size),checkpoint/"official_reproduction.pt")

    def compute_metrics(prediction):
        probability=torch.as_tensor(prediction.predictions).softmax(-1)[:,1].tolist()
        return native_predictions(args,probability,prediction.label_ids.tolist())
    optimizer=LOSVERAdamW(model.parameters(),lr=3e-5,weight_decay=0.,eps=1e-8)
    framework=run/"framework"
    training_args=TrainingArguments(output_dir=str(framework),per_device_train_batch_size=4,
        per_device_eval_batch_size=8,num_train_epochs=10,eval_strategy="epoch",save_strategy="steps",
        save_steps=500,save_total_limit=1,seed=seed,learning_rate=3e-5,fp16=True,report_to=[],
        load_best_model_at_end=False,disable_tqdm=True)
    if args.reproduction_smoke:
        training_args.max_steps=2
        training_args.eval_strategy="no"
        training_args.save_strategy="no"
    trainer=RecordedTrainer(model=model,args=training_args,train_dataset=datasets.get("train"),
        eval_dataset=datasets["valid"],compute_metrics=compute_metrics,
        optimizers=(optimizer,None),callbacks=[RecordedCallback()])
    if args.reproduction_smoke:
        parameter=next(model.parameters())
        initial=parameter.detach().clone()
        updates=[]
        hook=optimizer.register_step_post_hook(lambda *unused: updates.append(1))
        try:
            trainer.train()
        finally:
            hook.remove()
        trainer_update=float((parameter.detach()-initial.to(parameter.device)).abs().max())
        if not updates or trainer_update==0:
            raise ValueError("VulBERTa native Trainer did not update parameters")
        # The two-step diagnostic scheduler has reached zero. The independent
        # one-batch gradient/reload check below uses the configured learning rate.
        for group in optimizer.param_groups:
            group["lr"]=meta["training_config"]["learning_rate"]
        def forward(batch):
            inputs={k:v.to(args.device) for k,v in batch.items()}
            loss,outputs=trainer.compute_loss(model,inputs,return_outputs=True)
            return loss,outputs["logits"].softmax(-1)
        native_smoke(args,model,None,datasets["train"],forward=forward,optimizer=optimizer)
        smoke=json.loads((run/"smoke.json").read_text())
        smoke.update(native_trainer_steps=trainer.state.global_step,native_trainer_updates=len(updates),
                     native_trainer_maximum_parameter_update=trainer_update)
        write_json(run/"smoke.json",smoke)
        return
    if args.reproduction_evaluate:
        checkpoint=torch.load(run/"best-p_c.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        args.reproduction_epoch=checkpoint["selected_epoch"]
        for split in ("valid","test"):
            args.reproduction_split=split;trainer.predict(datasets[split])
        return
    checkpoint=get_last_checkpoint(str(framework)) if framework.exists() else None
    if checkpoint and not args.reproduction_resume:
        raise FileExistsError("Started VulBERTa training requires --resume")
    if checkpoint:
        saved=torch.load(Path(checkpoint)/"official_reproduction.pt",map_location="cpu",weights_only=False)
        if saved["metadata"]!=meta: raise ValueError("VulBERTa resume configuration differs")
        args.reproduction_loop=saved["loop"]
        with (run/"best.training.jsonl").open("r+b") as handle: handle.truncate(saved["log_position"])
    args.reproduction_progress=TrainingProgress(run/"best.pt",resume=bool(checkpoint))
    args.reproduction_started=time.monotonic()
    trainer.train(resume_from_checkpoint=checkpoint)
    native_finish(args,model,trainer.optimizer,trainer.lr_scheduler,save_state=False)


def report(output_dir, data_dir="data"):
    from .rgcl_study import run_path, inspect_run, BACKBONE_MODELS
    root=Path(output_dir)
    methods=(*UPSTREAM,"losver","rgcl")
    losver_reference=json.loads((LOSVER_REFERENCE/"losver"/"training.metrics.json").read_text())
    cohorts={s:[{k:r[k] for k in ("sample_key","pair_key","target")}
                for r in official_rows(data_dir,s)] for s in PAIRS}
    records=[]
    for method in methods:
        for seed in SEEDS:
            path = LOSVER_REFERENCE/"losver" if method=="losver" and seed==123456 else root/f"seed{seed}"/"primevul_paired"/method
            if method=="rgcl":
                path=run_path("results/rgcl_multiseed_512","primevul_paired",seed,"rgcl")
            test_file=path/"test.metrics.json"
            if not test_file.is_file():
                access=root/f"{method}-weight-access.json"
                detail=json.loads(access.read_text()) if access.exists() else {}
                reason=detail.get("reason") if detail.get("status")=="blocked" else None
                records.append(dict(method=method,seed=seed,status="blocked" if reason else "incomplete",run=str(path),reason=reason))
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
            elif method=="rgcl":
                inspect_run(path,data_dir,"primevul_paired",seed,"rgcl",BACKBONE_MODELS["unixcoder"],checkpoint=False)
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
                optimizer_updates=training.get("optimizer_updates"),amp_skipped_updates=training.get("amp_skipped_updates"),
                validation_zero_pc_epochs=sum(r["validation"]["p_c"]==0 for r in history),
                last_train_loss=history[-1]["train_loss"],test_pair_counts=value["pair_counts"],
                validation={k:valid[k] for k in METRICS},test={k:value[k] for k in METRICS},
                valid_reload_max_probability_difference=reload_difference,
                metrics_recomputed_from_predictions=[k for k in METRICS if k not in saved["metrics"]]))
    aggregate={split:{m:{k:dict(mean=statistics.mean(r[split][k] for r in records if r["method"]==m),
        std=statistics.stdev(r[split][k] for r in records if r["method"]==m)) for k in METRICS}
        for m in methods if sum(r["method"]==m and r["status"]=="complete" for r in records)==3}
        for split in ("validation","test")}
    result=dict(protocol=dict(dataset="PrimeVul official paired",pairs=PAIRS,seeds=list(SEEDS),
        selection="full Valid P-C",threshold=.5,comparison=">",std="sample SD, ddof=1, n=3 seeds"),
        upstream=dict(UPSTREAM,losver=dict(repo="https://github.com/waroad/losver",
            commit=losver_reference["upstream_commit"],
            implementation="existing formal 512 source_offsets with linear function classifier",
            training_config=losver_reference["training_config"])),runs=records,aggregate=aggregate,
        environment=dict(python=sys.version.split()[0],cuda=torch.version.cuda,
            packages={p:version(p) for p in ("torch","transformers","numpy","scipy","pandas","scikit-learn",
                "tokenizers","lightning","accelerate","einops","easydict","lizard")}))
    write_json(root/"comparison.json",result)
    lines=["# 官方漏洞检测基线：PrimeVul 三种子复现","",
        "LineVul、SVulD、ReGVD、VulBERTa-MLP、Vul-Mixer沿用作者模型、Tokenizer、损失、采样、优化器和原始最大训练预算；"
        "LOSVER复用本仓库已有正式512实现。统一按完整Valid P-C选模，严格prob>0.5。",
        "每组Test必须含564对/1128函数；Test不参与训练、早停或选择。","",
        "表内指标均为0–1数值；P-C乘100为百分比。PR-AUC沿用现有average_precision_score口径。"
        "历史LOSVER未保存FPR字段时，从全部564个修复函数的既有概率按严格>0.5重算，不修改历史文件。","",
        "| 方法 | Seed | 实际Epoch | 选中Epoch | Test P-C | MCC | F1 | AUC | PR-AUC | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if aggregate["test"]:
        ordered=sorted(aggregate["test"],key=lambda m:aggregate["test"][m]["p_c"]["mean"],reverse=True)
        mcc_best=max(aggregate["test"],key=lambda m:aggregate["test"][m]["mcc"]["mean"])
        f1_best=max(aggregate["test"],key=lambda m:aggregate["test"][m]["f1"]["mean"])
        svuld_epochs="/".join(str(r["completed_epochs"]) for r in records if r["method"]=="svuld")
        pc_text="；".join(f"{m} {100*aggregate['test'][m]['p_c']['mean']:.2f} ± "
                           f"{100*aggregate['test'][m]['p_c']['std']:.2f}%" for m in ordered)
        lines[2:2]=[f"已完成{sum(r['status']=='complete' for r in records)}组正式评价；"
            f"阻塞{sum(r['status']=='blocked' for r in records)}组，未完成{sum(r['status']=='incomplete' for r in records)}组。"
            "既有基线和RGCL只复用，未重新训练或覆盖。",
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
            lines.append(f"| {r['method']} | {r['seed']} | {'阻塞' if r['status']=='blocked' else '未完成'} | — | — | — | — | — | — | — | — |")
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
    lines += ["- rgcl: 复用results/rgcl_multiseed_512的Seed42/1234及results/attention_context_512/context_diff的Seed123456；"
              "UniXcoder、512 Token、Batch8、16轮、lr=2e-5、K=5、λ=0.1和Token加权区域损失，原权重及结果未改动。"]
    access=root/"vulberta-weight-access.json"
    if access.exists():
        detail=json.loads(access.read_text())
        lines += ["","## VulBERTa作者权重与notebook适配","",
            "原官方OneDrive链接的普通访问发生TLS错误，现有mihomo新加坡出口的命令行下载返回403；"
            "完整分享API也返回401。按照用户追加授权，临时切到现有香港节点，使用独立Firefox代理profile，"
            "成功取得同一作者分享的pretraining_model.zip；原代理选择已恢复。"
            f"压缩包{detail.get('archive_bytes')}字节，模型位于/home/PublicData/PHY-data/resource/VulBERTa。"
            "下载历史保留在vulberta-weight-access.json，没有替换成其他模型或重新预训练。",
            "从固定提交的Finetuning_VulBERTa-MLP.ipynb直接加载Clang+BPE、cleaner、Dataset和MyTrainer定义，"
            "没有复制训练循环；沿用RobertaForSequenceClassification双输出分类头和Train类别加权CE。"
            "原始cleaner的注释/换行/Tab处理不改动，全部原始函数成员及0/1标签保留。",
            "沿用1024 Token、Batch4、Eval Batch8、10轮、lr=3e-5、FP16、线性调度/零warmup、"
            "weight_decay=0和梯度裁剪1；旧HF AdamW更新规则复用LOSVERAdamW。"
            "只移除Tokenizers已删除且未使用的导入，适配compute_loss新增参数、优化器step的可选closure接口、"
            "数据路径和固定Valid P-C选择；closure=None时更新逐值不变。",
            "分词结果按划分缓存，检查上游版本及完整成员，使用文件锁避免多个种子重复提取。"
            "原生Trainer每500步保存模型、优化器、调度器、scaler、RNG与恢复位置，只保留一份恢复目录；"
            "共享日志在恢复时截断到对应位置，最优分类权重另外按完整Valid P-C保存。"]
    for r in records:
        if r["method"]=="vulberta" and r["status"]=="complete":
            counts=r["test_pair_counts"]
            lines.append(f"- Seed{r['seed']}：完成{r['completed_epochs']}轮，其中"
                f"{r['validation_zero_pc_epochs']}轮Valid P-C为0，末轮Train CE={r['last_train_loss']:.6f}；"
                f"所选权重Test P-C/P-V/P-B/P-R数量为{counts['p_c']}/{counts['p_v']}/{counts['p_b']}/{counts['p_r']}。")
    input_check=root/"vulberta-input-check.log"
    if input_check.is_file():
        checked=json.loads(input_check.read_text())
        lines.append("输入接口复核见vulberta-input-check.log：两划分各取前8条，按作者批次右Padding规则重编码，"
                     "与缓存逐值一致；全部成员保留，未发现整体退化为同一输入或特殊Token空输入。")
        for split,s in checked.items():
            lines.append(f"- {split}：{s['functions']}个函数，{s['distinct_encoded_functions']}种不同编码，"
                f"仅特殊Token输入{s['special_tokens_only']}个，非Padding输入的UNK比例"
                f"{100*s['unknown_tokens']/s['nonpadding_tokens']:.2f}%。")
    lines += ["","## Vul-Mixer适配与原始行为","",
        "直接运行固定提交的run.py、FireSVD和CurriculumLearning；模型、多粒度路径、损失及AdamW更新规则均未重写。"
        "输入先按作者config/clang-format.yml格式化，全部原始成员/标签不变。官方cache_embeddings.py只缓存"
        "GraphCodeBERT的50265×768静态词嵌入表，并非提取Transformer上下文表示；缓存仅构建一次，三种子共同读取。",
        "保留1024 Token、Batch128、20轮、lr=5e-4、AdamW默认weight_decay=0.01、无调度器和无梯度裁剪、mixed FP16。"
        "官方NoCL样本权重为1，训练类别权重N/Nc=2，两者原样保留。未额外引入Joern或编码器。",
        "GaussianBlur在eval也随机采样，保持作者行为。所选checkpoint额外记录完整Valid前的RNG，"
        "冻结重载后恢复同一RNG与批次顺序，比较逐函数概率与决策；Test随后只做一次完整564对评价。",
        "原始Lightning训练循环和两批sanity validation保留。轮末原生完整checkpoint保存优化器、AMP scaler、"
        "Lightning loop以及追加的Python/NumPy/Torch/CUDA RNG和日志位置，原子覆盖framework-last.ckpt。"
        "--resume在轮末恢复；中途异常最多重放当前轮，不逐步写大权重。",
        "新增依赖仅einops 0.8.1、easydict 1.13、lizard 1.17.31，无核心包降级；"
        "torch.load显式weights_only=False以读取作者缓存的nn.Embedding模块，本地模型路径适配同一GraphCodeBERT。"]
    for r in records:
        if r["method"]=="vulmixer" and r["status"]=="complete":
            lines.append(f"- Seed{r['seed']}：预算内步数{r['completed_epochs']*math.ceil(8708/128)}，"
                f"实际参数更新{r['optimizer_updates']}，原生AMP跳过{r['amp_skipped_updates']}次；没有补跑跳过步数。")
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
        "新增方法分别使用prepare --method vulmixer/vulberta及run/smoke --method vulmixer/vulberta；"
        "seed分别选择42、1234、123456。每个作者训练循环使用相同的正式接口，"
        "本机/tmp中的组合级文件锁防止同一服务器多个tmux队列重复训练；缓存原子完成后只读共享。"
        "恢复复用原项目模型/优化器/调度器/RNG原子保存与日志截断。",
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
    lines += ["","本轮41项相关测试通过（12项官方基线测试及既有LOSVER/优化器检查）；记录见vulmixer-tests.log。"
        "恢复测试覆盖作者使用的0/4 worker、随机/顺序采样、Dropout、最终参数及RNG一致性。",
        "实际检查中，最初的stdlib csv读取触发131072字段上限，改用作者的pandas读取器后全量源码对齐通过；"
        "沙箱的本地多进程通信权限不足，相关检查改在同一Conda环境的宿主执行并通过。",
        "I/O使用周期/轮末原子last.pt和Valid改善时的最优权重覆盖保存；评价元数据首次读取后缓存，"
        "不会逐轮累积大权重文件。续跑入口为同一run命令的--resume。",""]
    lines += ["真实错误与处理：完整Lightning恢复最初触发PyTorch的weights_only默认值变化，"
        "对本机自行保存的完整checkpoint显式weights_only=False后，连续/中断恢复的采样、Dropout、"
        "AdamW动量及参数、最终RNG全部一致。Vul-Mixer的GPU smoke首步在默认AMP scale=65536出现溢出，"
        "保留原生GradScaler跳步/自动降scale，随后取得有限梯度及真实参数更新，再验证权重恢复的输出一致性。"
        "正式训练保持原始AMP规则，不改变初始scale、精度或训练预算。",
        "共享GlusterFS上的缓存排他锁使后续只读评价等待；改为本机/tmp作业/构建锁、缓存原子发布和只读并行访问后，"
        "只恢复阻塞的评价任务，已经完成的两组20轮权重直接复用。"
        "VulBERTa原生Trainer最初在4090上触发Accelerate的P2P/IB检查，在其单卡子进程设置"
        "NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1。随后首次原生更新发现旧优化器step缺少可选closure参数，"
        "补齐与旧HF AdamW相同的接口，原无closure路径的更新逐值不变；真实Trainer两步更新、"
        "独立梯度及恢复检查通过，模型与正式优化配置不变。",
        "作者Tokenizer对超过512的原始Token序列发出警告；Vul-Mixer实际只查静态词嵌入表并截到1024，"
        "没有调用受512位置限制的Transformer编码前向，未因此删除或截成额外不同预算的样本。",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    with (root/"comparison.csv").open("w",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(("method","seed","status",*METRICS))
        for r in records:
            writer.writerow((r["method"],r["seed"],r["status"],*[r.get("test",{}).get(k) for k in METRICS]))
    return result
