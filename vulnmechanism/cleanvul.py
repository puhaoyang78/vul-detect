"""CleanVul release members and the unchanged RGCL Diff/token mapping."""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

from .artifacts import write_json
from .data import SPLITS, _line_diff, paired_features, read_jsonl, split_rgcl_groups, write_jsonl
from .reposvul import CC_LANGUAGES, _digest, _quality
from .syntax import parse_source, source_token_spans

SCORE_SOURCE = "https://arxiv.org/html/2411.17274v3#S3"
CVE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)


def load_pairs(raw_root):
    """Use every score>=3 C/C++ row; syntax checks describe, never filter it.

    The release supplies before/after snippets rather than a native 0/1 column.
    Preserve its repair-before=1 / repair-after=0 classification convention and
    same-row correspondence, exactly as supplied. Parser failures, renaming,
    repeated entries and lexical equality are not classification exclusion rules.
    """
    csv.field_size_limit(100_000_000)
    files, candidates, audit = {}, [], []
    for score in range(5):
        path = Path(raw_root) / f"vulnerability_score_{score}.csv"
        counts, languages = Counter(), Counter()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"func_before", "func_after", "commit_url", "commit_msg",
                        "file_name", "vulnerability_score", "extension", "cve_id", "is_test"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing CleanVul columns")
            for number, r in enumerate(reader, 1):
                if int(r["vulnerability_score"]) != score:
                    raise ValueError(f"{path}:{number}: score differs from filename")
                counts["rows"] += 1
                language = r["extension"].lower().lstrip(".")
                languages[language] += 1
                if language not in CC_LANGUAGES:
                    continue
                counts["cc_rows"] += 1
                counts["cc_with_cve_field"] += bool(r["cve_id"].strip())
                counts["cc_missing_before"] += not r["func_before"].strip()
                counts["cc_missing_after"] += not r["func_after"].strip()
                counts["cc_identical_source"] += r["func_before"] == r["func_after"]
                if score < 3:
                    continue
                record = dict(pair_key=f"cleanvul:{score}:{number}", score=score, source_row=number,
                    source_file=path.name, language=language, commit_url=r["commit_url"],
                    file_name=r["file_name"], before_name=None, after_name=None, reason=None)
                audit.append(record)
                url = urlsplit(r["commit_url"])
                parts = url.path.strip("/").split("/")
                before, after = r["func_before"], r["func_after"]
                reason = None
                if not before.strip() or not after.strip():
                    reason = "missing_source"
                elif r["is_test"].lower() != "false":
                    reason = "test_or_unknown_test_flag"
                elif (url.scheme not in {"http", "https"} or url.hostname != "github.com"
                      or len(parts) != 4 or parts[2] != "commit"
                      or re.fullmatch(r"[0-9a-fA-F]{4,40}", parts[3]) is None
                      or not r["file_name"].strip()):
                    reason = "missing_or_invalid_commit_file_identity"
                else:
                    a, b = parse_source(before), parse_source(after)
                    record.update(before_name=a["name"], after_name=b["name"])
                    if not a["strict"] or not b["strict"]:
                        reason = "parse_incomplete_or_not_single_function"
                    elif not a["name"] or not b["name"]:
                        reason = "unknown_function_name"
                    elif a["name"] != b["name"]:
                        reason = "function_name_mismatch"
                    elif tuple(t[0] for t in source_token_spans(before)) == tuple(t[0] for t in source_token_spans(after)):
                        reason = "comments_or_formatting_only"
                record["reason"] = reason or "retained"
                if not before.strip() or not after.strip():
                    raise ValueError(f"{path}:{number}: empty released source; cannot invent a function")
                if reason in {"test_or_unknown_test_flag", "missing_or_invalid_commit_file_identity"}:
                    raise ValueError(f"{path}:{number}: inspect changed release metadata: {reason}")
                cves = sorted({m.upper() for m in CVE.findall(r["cve_id"] + "\n" + r["commit_msg"])})
                metadata = dict(pair_key=record["pair_key"], project=f"https://github.com/{parts[0]}/{parts[1]}",
                    commit_id=parts[3].lower(), commit_url=r["commit_url"], cves=cves,
                    commit_reference_kind="full" if len(parts[3]) == 40 else "abbreviated",
                    cve_source="csv" if r["cve_id"].strip() else "commit_message" if cves else "missing",
                    file_name=r["file_name"], func_name=record["before_name"], vulnerability_score=score,
                    source_file=path.name, source_row=number, language=language)
                candidates.append((record, metadata, before, after))
        files[path.name] = dict(score=score, counts=dict(counts), languages=dict(languages))

    # Retain the earlier check as a reproducible diagnostic, not an exclusion.
    identities = defaultdict(set)
    for record, meta, before, after in candidates:
        if record["reason"] == "retained":
            identities[(meta["commit_url"], meta["file_name"], meta["func_name"])].add((before, after))
    rows, counterparts = [], {}
    for record, meta, before, after in candidates:
        if record["reason"] == "retained" and len(identities[(meta["commit_url"], meta["file_name"], meta["func_name"])]) != 1:
            record["reason"] = "ambiguous_same_commit_file_function"
        record["included_for_classification"] = True
        old, new = meta["pair_key"] + ":before", meta["pair_key"] + ":after"
        for key, partner, code, target in ((old, new, before, 1), (new, old, after, 0)):
            rows.append(dict(meta, sample_key=key, repair_key=partner, func=code, target=target,
                             normal_kind="vulnerable" if target else "repaired"))
            counterparts[key] = after if target else before
    result = dict(raw_root=str(Path(raw_root).resolve()), files=files, high_confidence_cc_pairs=len(audit), retained_pairs=len(rows)//2,
        diagnostic_reasons=dict(Counter(a["reason"] for a in audit)),
        classification_members_removed=0,
        score_definition=dict(source=SCORE_SOURCE, meaning="LLM confidence that the change fixes a vulnerability; not severity or human confirmation",
                              score_3="high likelihood", score_4="very high likelihood", minimum=3),
        association_rule="released CSV row supplies the before/after association; no parser/name/change-type classification filter",
        label_provenance="score>=3 repair-before target=1 and repair-after target=0; confidence-based repair labels, not manual ground truth",
        raw_sources_modified=False, duplicates_removed=False)
    return rows, counterparts, result, audit


def region_record(row, tokenizer, dataset, split):
    """Eligibility is precisely the existing paired_features + region loss gate."""
    f = paired_features(row, tokenizer, "baseline")
    truth, visible = set(row["lines_ground_truth"]), set(f["line_numbers"])
    complete = {n for n, ok in zip(f["line_numbers"], f["line_complete"]) if ok}
    effective = 1 in f["line_labels"] and 0 in f["line_labels"]
    if effective:
        reason = "effective"
    elif not row.get("diff_reliable", True):
        reason = "unreliable_or_missing_counterpart"
    elif not truth:
        reason = "no_changed_source_lines_on_this_side"
    elif not truth & visible:
        reason = "changed_lines_outside_visible_input"
    elif "diff_reliable" in row and truth & (visible - complete):
        reason = "partially_visible_changed_line"
    else:
        reason = "no_visible_background"
    return dict(dataset=dataset, split=split, sample_key=row["sample_key"], pair_key=row["pair_key"],
        target=row["target"], normal_kind=row.get("normal_kind", "vulnerable" if row["target"] else "repaired"),
        source_tokens=f["source_tokens"], truncated=f["source_tokens"] > 510,
        diff_lines=len(truth), visible_diff_lines=len(truth & visible),
        complete_visible_diff_lines=len(truth & complete),
        partial_visible_diff=bool(truth & (visible-complete)), effective_region=effective,
        complete_line_effective_region=effective and not bool(truth & (visible-complete)), reason=reason)


def coverage(records):
    n = len(records)
    count = sum(r["effective_region"] for r in records)
    return dict(functions=n, effective_region=count, effective_percent=100*count/n if n else None,
        complete_line_effective_region=sum(r["complete_line_effective_region"] for r in records),
        with_diff_lines=sum(r["diff_lines"] > 0 for r in records),
        with_visible_diff=sum(r["visible_diff_lines"] > 0 for r in records),
        partially_visible_diff=sum(r["partial_visible_diff"] for r in records),
        truncated=sum(r["truncated"] for r in records), reasons=dict(Counter(r["reason"] for r in records)))


def coverage_table(records):
    return {split: {side: coverage([r for r in records if (split == "all" or r["split"] == split)
                                                   and (side == "all" or (side in ("1", "0") and r["target"] == int(side))
                                                        or side == r["normal_kind"]
                                                        or (side == "paired" and r["reason"] != "unreliable_or_missing_counterpart")
                                                        or (side == "paired_before" and r["target"] == 1 and r["reason"] != "unreliable_or_missing_counterpart"))])
                    for side in ("all", "1", "0", "repaired", "ordinary", "paired", "paired_before")} for split in ("all", *SPLITS)}


def collect(raw_root, model_path):
    from transformers import RobertaTokenizerFast

    rows, others, stats, audit = load_pairs(raw_root)
    if not rows:
        raise ValueError("No high-confidence C/C++ release members")
    commits = defaultdict(set)
    for row in rows:
        commits[row["project"]].add(row["commit_id"])
    for row in rows:
        prefix = min((h for h in commits[row["project"]] if row["commit_id"].startswith(h)), key=lambda h: (len(h), h))
        row["commit_group"] = row["project"] + "@" + prefix
    # Preserve the original reference; compatible short/full references share
    # the same conservative grouping key, including ambiguous short prefixes.
    grouped = [dict(row, commit_id=row["commit_group"]) for row in rows]
    stats["split_protocol"] = split_rgcl_groups(grouped, others, seed=42)
    stats["split_protocol"]["commit_references"] = "repository-qualified shortest observed compatible hash prefix; original commit_id preserved"
    for row, assigned in zip(rows, grouped):
        row.update(split=assigned["split"], group_key=assigned["group_key"])
    stats["retained_by_score"] = {k: n//2 for k, n in Counter(str(r["vulnerability_score"]) for r in rows).items()}
    stats["retained_by_language"] = {k: n//2 for k, n in Counter(r["language"] for r in rows).items()}
    stats["commit_reference_kinds"] = {k: n//2 for k, n in Counter(r["commit_reference_kind"] for r in rows).items()}
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False, local_files_only=True)
    annotations, records = [], []
    for i, row in enumerate(rows, 1):
        annotation = dict(row, lines_ground_truth=_line_diff(row["func"], others[row["sample_key"]])[0],
                          diff_reliable=True)
        annotations.append(annotation)
        records.append(region_record(annotation, tokenizer, "cleanvul", row["split"]))
        if i % 1000 == 0:
            print(f"CleanVul: mapped {i}/{len(rows)} functions", flush=True)
    overlap = {}
    for field in ("pair_key", "commit_id", "commit_group", "cves", "exact_source", "normalized_source"):
        groups = defaultdict(set)
        for row in rows:
            values = row["cves"] if field == "cves" else [_digest(row["func"], field == "normalized_source")] if field.endswith("source") else [row[field]]
            for value in values:
                if value:
                    groups[value].add(row["split"])
        overlap[field] = sum(len(splits) > 1 for splits in groups.values())
    if any(overlap.values()):
        raise ValueError("CleanVul connected grouping crossed splits")
    stats["cross_split_overlap_groups"] = overlap
    stats.update(input_tokens=512, source_token_budget=510, model_path=str(model_path), seed=42,
        K=5, region_loss_weight=0.1, coverage=coverage_table(records),
        class_counts=dict(Counter(str(r["target"]) for r in rows)),
        cve_sources=dict(Counter(r["cve_source"] for r in rows)),
        label_quality=_quality(rows))
    for split in SPLITS:
        members = [r for r in rows if r["split"] == split]
        rc = stats["coverage"][split]["all"]
        stats.setdefault("splits", {})[split] = dict(pairs=len(members)//2, functions=len(members),
            class_counts=dict(Counter(str(r["target"]) for r in members)), fraction=len(members)/len(rows),
            normal_kinds=dict(Counter(r["normal_kind"] for r in members)),
            coverage=dict(reliable_counterpart=len(members), effective_region=rc["effective_region"],
                          no_region_function_bce_only=len(members)-rc["effective_region"], truncated=rc["truncated"]))
    train = stats["splits"]["train"]["class_counts"]
    stats["classification_positive_weight"] = train["0"] / train["1"]
    stats["classification_weight_rule"] = "Train negatives / positives; same BCE for all three methods"
    stats["official_evaluation_split"] = False
    return rows, annotations, records, stats, audit, tokenizer


def comparison(data_dir, tokenizer):
    """Read historic members without writing or substituting any label/method."""
    result, records, codes = {}, [], {}
    for dataset, root in (("primevul_paired", Path(data_dir)/"primevul_paired"),
                          ("secvuleval", Path(data_dir)/"rgcl/secvuleval")):
        dataset_records, dataset_codes, sources = [], set(), []
        for split in SPLITS:
            path = root/f"{split}.lines.jsonl"
            if not path.exists():
                path = root/f"{split}.jsonl"
            rows = read_jsonl(path)
            sources.append(str(path))
            if any("lines_ground_truth" not in r for r in rows):
                pairs = defaultdict(list)
                for row in rows:
                    pairs[row["pair_key"]].append(row)
                for pair in pairs.values():
                    if len(pair) != 2 or sorted(r["target"] for r in pair) != [0, 1]:
                        raise ValueError(f"{path}: missing original complete repair pair")
                    for row, other in ((pair[0], pair[1]), (pair[1], pair[0])):
                        row["lines_ground_truth"] = _line_diff(row["func"], other["func"])[0]
            for i, row in enumerate(rows, 1):
                dataset_records.append(region_record(row, tokenizer, dataset, split))
                dataset_codes.add(_digest(row["func"]))
                if i % 2000 == 0:
                    print(f"{dataset} {split}: mapped {i}/{len(rows)} functions", flush=True)
        result[dataset] = dict(sources=sources, coverage=coverage_table(dataset_records),
            supervision="existing paired_features on original annotations; missing PrimeVul Test annotations reconstructed from its original repair pairs only",
            history_modified=False)
        records.extend(dataset_records)
        codes[dataset] = dataset_codes
    return result, records, codes


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect(raw_root, data_dir, output_dir, model_path):
    root = Path(output_dir)
    if any((root/name).exists() for name in ("statistics.json", "report.md", "pairs.csv", "function_coverage.csv")):
        raise FileExistsError(f"{root}: existing inspection/history was not overwritten")
    rows, annotations, records, stats, audit, tokenizer = collect(raw_root, model_path)
    stats["comparison"], historical, source_sets = comparison(data_dir, tokenizer)
    stats["cross_dataset_exact_sources"] = {name: dict(
        shared_unique_sources=len({_digest(r["func"]) for r in rows} & sources),
        cleanvul_functions=sum(_digest(r["func"]) in sources for r in rows)) for name, sources in source_sets.items()}
    stats["coverage_definition"] = "effective iff unchanged paired_features emits both positive Diff and background code lines; native source inputs include CLS/SEP and first 510 source tokens; partial-line reliability rules are preserved per dataset"
    stats["raw_high_confidence_coverage"] = {
        side: dict(effective_region=stats["coverage"]["all"][side]["effective_region"],
                   denominator=stats["high_confidence_cc_pairs"],
                   effective_percent=100*stats["coverage"]["all"][side]["effective_region"]/stats["high_confidence_cc_pairs"])
        for side in ("1", "0")}
    return write_inspection(output_dir, stats, audit, records+historical)


def write_inspection(output_dir, stats, audit, records):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"statistics.json", stats)
    write_csv(root/"pairs.csv", audit)
    write_csv(root/"function_coverage.csv", records)
    lines = ["# CleanVul 成员与 Diff 监督统计", "",
        "评分表示漏洞修复置信度，不是严重性评分或人工确认。原始 CSV 与历史实验文件均未修改。",
        f"评分定义：[CleanVul 原论文]({SCORE_SOURCE})。按本地 CSV 字段统计，未套用论文中其他版本的样本总数。", "",
        f"评分 ≥3 的 C/C++：{stats['high_confidence_cc_pairs']} 条前后记录；全部保留 {stats['retained_pairs']} 对，{sum(stats['class_counts'].values())} 个函数，类别 1:1。",
        "原先解析、函数名及词法变化检查只作为可复现诊断，不再淘汰分类成员；前后版本及标签沿用发布记录。", "",
        "有效区域：沿用 RGCL 的 512 Token 编码与 Diff 映射，包含前 510 个源码 Token；必须同时有修改代码行和背景代码行。"
        "CleanVul 的部分可见修改行遵循现有 diff_reliable 门控。未获得区域监督的函数仍可参与函数分类，不用邻行补造监督。", "",
        "| 数据 | 可靠配对函数 | 修复前有效 | 修复后有效 | 配对函数有效 |",
        "|---|---:|---:|---:|---:|"]
    for name, value in [("CleanVul", stats), *stats["comparison"].items()]:
        c=value["coverage"]["all"]
        cell=lambda k: f"{c[k]['effective_region']}/{c[k]['functions']} ({c[k]['effective_percent']:.2f}%)"
        lines.append(f"| {name} | {c['paired']['functions']} | {cell('paired_before')} | {cell('repaired')} | {cell('paired')} |")
    lines.extend(["", "上表限定可靠修复对子集。SecVulEval 另有 974 个缺少修复对应的漏洞函数和 4,418 个普通负例，它们计入原实验全体成员口径，不计入上表。PrimeVul 的历史注释未使用 diff_reliable，部分可见修改行按历史实现保留；CleanVul 与 SecVulEval 沿用现有可靠性门控。complete_line_effective_region 另列统一要求可见修改行不被截断的诊断计数，不要求窗口外的修改行也可见，不改变原训练监督。详细分母及原因见 statistics.json、function_coverage.csv。", "",
        "| 划分 | 修复对 | 漏洞函数 | 修复函数 | 修复前有效 | 修复后有效 |",
        "|---|---:|---:|---:|---:|---:|"])
    for split, value in stats["splits"].items():
        c = stats["coverage"][split]
        lines.append(f"| {split} | {value['pairs']} | {value['class_counts']['1']} | {value['class_counts']['0']} | {c['1']['effective_region']} | {c['0']['effective_region']} |")
    lines.extend(["", "Seed42，目标 8:1:1，沿用现有 CVE、提交、修复关系及相同源码的传递分组；重复或异标签函数仅统计并分组，不作为淘汰条件。"
        "CSV 未提供的 CVE 仅从提交信息中的显式 CVE 编号提取，并保留来源说明。", "",
        "未启动训练，未更改 K=5、λ=0.1，也未修改 PrimeVul、SecVulEval 的数据或历史结果。"])
    lines.extend(["", "| 原始评分 | 全语言记录 | C/C++ 记录 |", "|---|---:|---:|"])
    for value in stats["files"].values():
        lines.append(f"| {value['score']} | {value['counts']['rows']} | {value['counts']['cc_rows']} |")
    lines.extend(["", "全体分类成员口径（包含没有修复关系的原生函数）：", ""])
    for name, value in [("CleanVul", stats), *stats["comparison"].items()]:
        c = value["coverage"]["all"]["all"]
        lines.append(f"- {name}: {c['effective_region']}/{c['functions']} ({c['effective_percent']:.2f}%)。")
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return stats


def prepare(raw_root, data_dir, model_path):
    dest = Path(data_dir)/"rgcl/cleanvul"
    if dest.exists():
        raise FileExistsError(f"{dest}: existing prepared data was not overwritten")
    rows, annotations, records, stats, audit, _ = collect(raw_root, model_path)
    return write_prepared(data_dir, rows, annotations, stats)


def write_prepared(data_dir, rows, annotations, stats):
    dest = Path(data_dir)/"rgcl/cleanvul"
    if dest.exists():
        raise FileExistsError(f"{dest}: existing prepared data was not overwritten")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cleanvul-", dir=dest.parent) as temporary:
        staging = Path(temporary)/"cleanvul"
        staging.mkdir()
        for split in SPLITS:
            write_jsonl(staging/f"{split}.jsonl", (r for r in rows if r["split"] == split))
            write_jsonl(staging/f"{split}.lines.jsonl", (r for r in annotations if r["split"] == split))
        write_csv(staging/"members.csv", [{k: r[k] for k in
            ("sample_key", "pair_key", "target", "split", "group_key", "normal_kind", "repair_key", "source_file", "source_row")}
            for r in rows])
        write_json(staging/"statistics.json", stats)
        staging.rename(dest)
    return stats
