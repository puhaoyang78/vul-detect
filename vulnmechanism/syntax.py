"""Binding-safe C/C++ surface transformations for the active augmentation method."""

from __future__ import annotations
import re
import difflib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp


_C = Parser(Language(tree_sitter_c.language()))


_CPP = Parser(Language(tree_sitter_cpp.language()))


def parser_for(language: str) -> Parser:
    if language == "c":
        return _C
    if language == "cpp":
        return _CPP
    raise ValueError("language must be c or cpp")


def walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode(errors="replace")


def apply_source_edits(source: str, edits) -> str:
    """Apply verified character edits; never interpret positions as UTF-8 bytes."""
    end = 0
    for a, b, value in edits:
        if not 0 <= end <= a <= b <= len(source) or not isinstance(value, str):
            raise ValueError("invalid or overlapping source edits")
        end = b
    for a, b, value in reversed(edits):
        source = source[:a] + value + source[b:]
    return source


_SOURCE_TOKEN = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[^\W\d]\w*|\d+(?:\.\d+)?|>>=|<<=|->\*|\.\.\.|::|->|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]',
    re.S,
)


@lru_cache(maxsize=16384)
def source_token_spans(source: str) -> tuple[tuple[str, int, int], ...]:
    """Lexical tokens with Python character offsets, including literals/operators."""
    return tuple(
        (m.group(), m.start(), m.end())
        for m in _SOURCE_TOKEN.finditer(source)
        if not m.group().startswith(("//", "/*"))
    )


def byte_character_map(source: str) -> dict[int, int]:
    result, byte = {0: 0}, 0
    for char, value in enumerate(source, 1):
        byte += len(value.encode("utf-8"))
        result[byte] = char
    return result


_BINDING_SCOPES = {
    "compound_statement",
    "for_statement",
    "if_statement",
    "switch_statement",
    "while_statement",
    "do_statement",
}


def declarator_identifier(node):
    while node is not None and node.type in {
        "init_declarator",
        "pointer_declarator",
        "array_declarator",
        "reference_declarator",
        "parenthesized_declarator",
    }:
        node = node.child_by_field_name("declarator")
    return node if node is not None and node.type == "identifier" else None


def function_bindings(function, source: bytes):
    """Position/scope bindings, also usable in ordinary call arguments."""
    body = function.child_by_field_name("body")
    if body is None:
        raise ValueError("missing function body")
    bindings, parameter_index = ([], 0)
    signature_lists = [
        n for n in walk(function.child_by_field_name("declarator")) if n.type == "parameter_list"
    ]
    signature_parameters = signature_lists[0] if signature_lists else None
    for node in walk(function):
        if node.type == "enumerator":
            name = node.child_by_field_name("name")
            scope = node.parent
            while scope is not None and scope.type not in _BINDING_SCOPES:
                scope = scope.parent
            if name is not None and scope is not None:
                bindings.append(
                    dict(
                        name=text(name, source),
                        declaration=name,
                        statement=node,
                        scope=scope,
                        parameter_index=None,
                        occurrences=[],
                        supported=False,
                    )
                )
            continue
        if node.type not in {
            "parameter_declaration",
            "optional_parameter_declaration",
            "declaration",
        } | {"for_range_loop"}:
            continue
        parameter = node.type in {"parameter_declaration", "optional_parameter_declaration"}
        index = parameter_index if parameter else None
        if parameter:
            parameter_index += 1
        if not parameter and (not body.start_byte <= node.start_byte < body.end_byte):
            continue
        for decl in node.children_by_field_name("declarator"):
            name = declarator_identifier(decl)
            if name is None:
                declarator = (
                    decl.child_by_field_name("declarator")
                    if decl.type == "init_declarator"
                    else decl
                )
                names = (
                    [n for n in walk(declarator) if n.type == "identifier"]
                    if declarator is not None
                    else []
                )
            else:
                names = [name]
            scope = (
                function if parameter else node if node.type == "for_range_loop" else node.parent
            )
            while (
                scope is not None
                and (not parameter)
                and (scope.type not in _BINDING_SCOPES | {"for_range_loop"})
            ):
                scope = scope.parent
            if scope is None:
                continue
            opaque = node.parent
            while (
                opaque is not None
                and opaque != function
                and (
                    opaque.type
                    not in {
                        "lambda_expression",
                        "class_specifier",
                        "struct_specifier",
                        "union_specifier",
                        "function_definition",
                    }
                )
            ):
                opaque = opaque.parent
            supported = name is not None and opaque == function
            if parameter and node.parent != signature_parameters:
                supported = False
                scope = node.parent
                while (
                    scope is not None
                    and scope != function
                    and (
                        scope.type
                        not in _BINDING_SCOPES
                        | {"catch_clause", "lambda_expression", "function_declarator"}
                    )
                ):
                    scope = scope.parent
                if scope is None:
                    continue
                if scope.type == "catch_clause" and name is not None and (opaque == function):
                    supported = True
            if opaque is not None and opaque != function and parameter:
                scope = opaque
            if any(
                (
                    n.type in {"attribute_specifier", "attribute_declaration", "gnu_asm_expression"}
                    or (
                        n.type == "storage_class_specifier"
                        and text(n, source) in {"extern", "typedef"}
                    )
                    for n in walk(node)
                )
            ):
                supported = False
            for name_node in names:
                binding = dict(
                    name=text(name_node, source),
                    declaration=name_node,
                    statement=node,
                    scope=scope,
                    parameter_index=index if node.parent == signature_parameters else None,
                    occurrences=[],
                )
                binding["supported"] = supported
                bindings.append(binding)
    return bindings


def resolve_local_binding(node, bindings, source: bytes):
    if node.type != "identifier" or node.parent.type in {
        "qualified_identifier",
        "scoped_identifier",
    }:
        return None
    matches = [
        b
        for b in bindings
        if b["name"] == text(node, source)
        and b["declaration"].end_byte <= node.start_byte
        and b["scope"].start_byte <= node.start_byte < b["scope"].end_byte
    ]
    return (
        min(
            matches,
            key=lambda b: (
                b["scope"].end_byte - b["scope"].start_byte,
                -b["declaration"].start_byte,
            ),
        )
        if matches
        else None
    )


@lru_cache(maxsize=16384)
def surface_binding_layout(source_text: str, language: str) -> dict:
    """Per-binding source spans; ordinary calls/globals do not reject a function.

    Recognizable macro uses, opaque captures and unsupported declarations reject
    the affected binding. Unresolved API/type/member names are never surface sites.
    """
    from collections import Counter

    raw = source_text.encode()
    root = parser_for(language).parse(raw).root_node
    empty = dict(
        bindings=[], function_name=[], function_references=[], comments=[], comment_points=[]
    )
    if root.has_error:
        return dict(empty, reasons={"parse_error": 1})
    nodes = list(walk(root))
    functions = [n for n in nodes if n.type == "function_definition"]
    if len(functions) != 1:
        return dict(empty, reasons={"ambiguous_function": 1})
    function = functions[0]
    body = function.child_by_field_name("body")
    if body is None:
        return dict(empty, reasons={"missing_body": 1})
    coordinates = byte_character_map(source_text)

    def span(n):
        return [coordinates[n.start_byte], coordinates[n.end_byte]]

    declarator = function.child_by_field_name("declarator")
    while declarator is not None and declarator.type != "identifier":
        if declarator.type in {"qualified_identifier", "scoped_identifier"}:
            declarator = declarator.child_by_field_name("name")
        else:
            declarator = declarator.child_by_field_name("declarator")
    function_name = [span(declarator)] if declarator is not None else []
    bindings = function_bindings(function, raw)
    by_name = {}
    for b in bindings:
        by_name.setdefault(b["name"], []).append(b)
    declarations = {(b["declaration"].start_byte, b["declaration"].end_byte): b for b in bindings}
    unsafe = {id(b): set() for b in bindings}
    for n in nodes:
        if n.type != "identifier":
            continue
        binding = declarations.get((n.start_byte, n.end_byte)) or resolve_local_binding(
            n, by_name.get(text(n, raw), []), raw
        )
        if binding is None:
            continue
        binding["occurrences"].append(n)
        parent = n.parent
        while parent is not None and parent != function:
            if parent.type.startswith("preproc_") or parent.type in {
                "lambda_expression",
                "class_specifier",
                "struct_specifier",
                "union_specifier",
                "gnu_asm_expression",
            }:
                unsafe[id(binding)].add("opaque_or_preprocessor_binding")
            if parent.type == "call_expression":
                callee = parent.child_by_field_name("function")
                if callee is not None and callee.type == "identifier":
                    name = text(callee, raw)
                    if name.upper() == name or name in {
                        "min",
                        "max",
                        "likely",
                        "unlikely",
                        "offsetof",
                        "container_of",
                    }:
                        unsafe[id(binding)].add("macro_ambiguous_binding")
            parent = parent.parent
    # Bare same-name identifiers whose declaration could not be resolved are
    # evidence of an unsupported binding, not permission to edit a subset.
    assigned = {(n.start_byte, n.end_byte) for b in bindings for n in b["occurrences"]}
    for n in nodes:
        if (
            n.type == "identifier"
            and body.start_byte <= n.start_byte < body.end_byte
            and (n.start_byte, n.end_byte) not in assigned
            and n.parent.type not in {"qualified_identifier", "scoped_identifier"}
        ):
            for b in by_name.get(text(n, raw), []):
                unsafe[id(b)].add("unresolved_same_name")
    result, reasons = [], Counter()
    column_sensitive = any(
        v in source_text for v in ("__builtin_COLUMN", "__COLUMN__", "source_location")
    )
    for b in bindings:
        issues = unsafe[id(b)]
        if not b["supported"]:
            issues.add("unsupported_binding")
        if column_sensitive:
            issues.add("column_sensitive_source")
        if len(b["occurrences"]) < 2:
            issues.add("unused_binding")
        if issues:
            reasons.update(issues)
            continue
        occurrences = sorted({tuple(span(n)) for n in b["occurrences"]})
        result.append(
            dict(
                name=b["name"],
                declaration=span(b["declaration"])[0],
                kind="parameter" if b["parameter_index"] is not None else "local",
                spans=occurrences,
            )
        )

    def safe_position(at):
        first = source_text.rfind("\n", 0, at) + 1
        last = source_text.find("\n", at)
        line = source_text[first : last if last >= 0 else len(source_text)]
        previous = source_text[: max(0, first - 1)].rsplit("\n", 1)[-1]
        return not (
            column_sensitive
            or line.lstrip().startswith("#")
            or line.rstrip().endswith("\\")
            or previous.rstrip().endswith("\\")
        )

    def opaque(n):
        while n is not None and n != function:
            if n.type.startswith("preproc_") or n.type in {
                "lambda_expression",
                "class_specifier",
                "struct_specifier",
                "gnu_asm_expression",
            }:
                return True
            if n.type == "call_expression":
                callee = n.child_by_field_name("function")
                if (
                    callee is not None
                    and callee.type == "identifier"
                    and text(callee, raw).isupper()
                ):
                    return True
            n = n.parent
        return False

    points = []
    comments = []
    for n in nodes:
        if (
            n.type == "compound_statement"
            and body.start_byte <= n.start_byte
            and n.end_byte <= body.end_byte
            and not opaque(n)
        ):
            points.append(coordinates[n.start_byte + 1])
            points.extend(
                coordinates[c.end_byte]
                for c in n.named_children
                if c.type != "comment" and not c.type.startswith("preproc_")
            )
        if n.type == "comment" and not opaque(n):
            a, b = span(n)
            directive = re.search(
                r"(?i)clang[ -]?format|NOLINT|cppcheck|IWYU|fall[ -]?through|@|pragma|coverage|lint:",
                source_text[a:b],
            )
            if not directive and all(
                safe_position(p)
                for p in range(a, b + 1)
                if p == a or source_text[p - 1 : p] == "\n"
            ):
                comments.append([a, b])
    points = [p for p in points if safe_position(p)]
    references = []
    observable = any(
        v in source_text
        for v in (
            "__func__",
            "__FUNCTION__",
            "__PRETTY_FUNCTION__",
            "__FUNCDNAME__",
            "source_location",
        )
    )
    macro = any(
        n.type.startswith("preproc_")
        or n.type == "call_expression"
        and n.child_by_field_name("function") is not None
        and text(n.child_by_field_name("function"), raw).isupper()
        for n in nodes
    )
    name = text(declarator, raw) if declarator is not None else None
    ambiguous = any(b["name"] == name for b in bindings)
    for n in nodes:
        if n.type == "identifier" and n != declarator and text(n, raw) == name:
            parent = n.parent
            if (
                parent is not None
                and parent.type == "call_expression"
                and parent.child_by_field_name("function") == n
            ):
                references.append(span(n))
            else:
                ambiguous = True
    if observable or macro or ambiguous or column_sensitive:
        function_name = []
        references = []
        reasons["function_name_observable_or_unresolved"] += 1
    return dict(
        bindings=result,
        function_name=function_name,
        function_references=references,
        comments=comments,
        comment_points=sorted(set(points)),
        reasons=dict(reasons),
    )


SURFACE_OPERATIONS = (
    "local_name",
    "function_name",
    "comment_insert",
    "comment_delete",
    "comment_replace",
)


SURFACE_RULES = dict(
    train_names=["natural", "snake_number", "camel_pair"],
    heldout_names=["pascal_number", "prefix_pair"],
    train_comments=["word_list", "numbered_phrase"],
    heldout_comments=["bracketed_list", "hyphen_phrase"],
    heldout_combinations=[
        ["function_name", "comment_replace"],
        ["local_name", "comment_delete", "comment_insert"],
    ],
)


def surface_materials(records, seed=42):
    """Label-blind Train-only names and ordinary comment words; fixed rule holdout."""
    import random

    names = set()
    words = set()
    for row in records:
        if row["split"] != "train":
            raise ValueError("surface materials accept Train functions only")
        source = row["raw_source"]
        language = resolve_language(source, row.get("language", "c"), row.get("file_name", ""))
        layout = surface_binding_layout(source, language)
        names.update(
            b["name"]
            for b in layout["bindings"]
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,47}", b["name"])
        )
        for a, b in layout["comments"]:
            content = source[a:b]
            if re.search(r"(?i)cve|cwe|vuln|security|patch|fix|bug|overflow|underflow", content):
                continue
            words.update(w.lower() for w in re.findall(r"\b[a-zA-Z]{3,16}\b", content))
    if len(names) < 6 or len(words) < 6:
        raise ValueError("insufficient Train-only surface naming/comment material")
    rng = random.Random(seed)
    names = sorted(names)
    words = sorted(words)
    rng.shuffle(names)
    rng.shuffle(words)
    # Cap only the material bank, not the eligible bindings of each function.
    names = names[:2048]
    words = words[:1024]
    ni = max(2, len(names) // 4)
    wi = max(2, len(words) // 4)
    return dict(
        source="Train safe binding names and ordinary comment words; no labels/CVE/patch metadata",
        train=dict(names=names[ni:], words=words[wi:]),
        heldout=dict(names=names[:ni], words=words[:wi]),
        rules=SURFACE_RULES,
    )


def surface_preserving_candidates(
    source: str,
    language: str,
    *,
    rng,
    materials,
    tokenizer,
    bank="train",
    kinds=None,
    used_bindings=(),
    comment_tokens=0,
    limit=4,
    proposal_limit=24,
):
    """Bounded dynamic single-edit proposals, with per-edit binding validation.

    Composition calls this on the newly edited source, never on old coordinates.
    Function renames synchronize resolved self-calls and are input-only stress:
    external project references are unavailable from a single function.
    """
    from collections import Counter

    if bank not in ("train", "heldout") or limit < 1 or proposal_limit < limit:
        raise ValueError("invalid surface proposal budget")
    layout = surface_binding_layout(source, language)
    skipped = Counter(layout["reasons"])
    result = []
    seen = set()
    if any(k in layout["reasons"] for k in ("parse_error", "ambiguous_function", "missing_body")):
        return result, dict(skipped)
    raw = source.encode()
    all_names = {
        text(n, raw)
        for n in walk(parser_for(language).parse(raw).root_node)
        if n.type.endswith("identifier")
    }
    tokens = source_token_spans(source)
    token_index = {a: i for i, (_, a, b) in enumerate(tokens)}
    categories = list(kinds or SURFACE_OPERATIONS)
    rng.shuffle(categories)
    data = materials[bank]
    rules = materials["rules"]
    name_rules = rules["train_names" if bank == "train" else "heldout_names"]
    comment_rules = rules["train_comments" if bank == "train" else "heldout_comments"]

    def name():
        rule = rng.choice(name_rules)
        first = rng.choice(data["names"])
        second = rng.choice(data["names"])
        if rule == "natural":
            value = first
        elif rule == "snake_number":
            value = f"{first}_{rng.randrange(2,1000)}"
        elif rule == "camel_pair":
            value = first + "_" + second[:1].upper() + second[1:]
        elif rule == "pascal_number":
            value = first[:1].upper() + first[1:] + str(rng.randrange(2, 1000))
        elif rule == "prefix_pair":
            value = second[:8] + "_" + first
        else:
            raise ValueError("unknown naming rule")
        return value, rule

    def comment():
        rule = rng.choice(comment_rules)
        parts = [rng.choice(data["words"]) for _ in range(rng.randrange(1, 9))]
        if rule == "word_list":
            value = " ".join(parts)
        elif rule == "numbered_phrase":
            value = str(rng.randrange(2, 1000)) + " " + " ".join(parts)
        elif rule == "bracketed_list":
            value = "[" + ", ".join(parts) + "]"
        elif rule == "hyphen_phrase":
            value = " - ".join(parts)
        else:
            raise ValueError("unknown comment rule")
        return " /* " + value + " */ ", rule

    def mapped(position, edits):
        return position + sum(
            len(v) - (b - a) for a, b, v in edits if b <= position and a < position
        )

    for attempt in range(proposal_limit):
        if len(result) >= limit:
            break
        kind = categories[attempt % len(categories)]
        binding = None
        added = 0
        rule = None
        unit_id = None
        if kind == "local_name":
            eligible = [
                b for b in layout["bindings"] if token_index[b["declaration"]] not in used_bindings
            ]
            if not eligible:
                skipped["no_unused_safe_binding"] += 1
                continue
            binding = rng.choice(eligible)
            value, rule = name()
            unit_id = token_index[binding["declaration"]]
            if value in all_names:
                skipped["name_collision"] += 1
                continue
            edits = [(a, b, value) for a, b in binding["spans"]]
        elif kind == "function_name":
            if not layout["function_name"] or "function" in used_bindings:
                skipped["no_safe_function_name"] += 1
                continue
            value, rule = name()
            unit_id = "function"
            if value in all_names:
                skipped["name_collision"] += 1
                continue
            edits = [
                (a, b, value) for a, b in layout["function_name"] + layout["function_references"]
            ]
        elif kind in ("comment_insert", "comment_delete", "comment_replace"):
            sites = layout["comment_points"] if kind == "comment_insert" else layout["comments"]
            if not sites:
                skipped["no_" + kind + "_site"] += 1
                continue
            site = rng.choice(sites)
            a, b = (site, site) if kind == "comment_insert" else site
            if kind == "comment_delete":
                value = " " + ("\n" * source[a:b].count("\n"))
                rule = "remove_ordinary_comment"
            else:
                value, rule = comment()
                added = len(tokenizer(value, add_special_tokens=False)["input_ids"])
                if comment_tokens + added > 64:
                    skipped["comment_token_budget"] += 1
                    continue
                value += "\n" * source[a:b].count("\n")
            edits = [(a, b, value)]
        else:
            raise ValueError("unknown surface edit category")
        edits = sorted(edits)
        changed = apply_source_edits(source, edits)
        if changed == source or changed in seen:
            skipped["duplicate_source"] += 1
            continue
        if parser_for(language).parse(changed.encode()).root_node.has_error:
            skipped["edited_parse_error"] += 1
            continue
        after = source_token_spans(changed)
        allowed = {
            (a, b) for a, b, value in edits if b > a and kind in ("local_name", "function_name")
        }
        if (
            source.count("\n") != changed.count("\n")
            or len(tokens) != len(after)
            or any(
                old != new and (a, b) not in allowed
                for (old, a, b), (new, _, _) in zip(tokens, after)
            )
        ):
            raise ValueError("surface edit changed execution tokens or physical line count")
        if binding is not None:
            matches = [
                b
                for b in surface_binding_layout(changed, language)["bindings"]
                if b["declaration"] == mapped(binding["declaration"], edits)
            ]
            expected = [
                (mapped(a, edits), mapped(a, edits) + len(value)) for a, b in binding["spans"]
            ]
            if len(matches) != 1 or sorted(map(tuple, matches[0]["spans"])) != sorted(expected):
                skipped["renamed_binding_unverified"] += 1
                continue
        if kind == "function_name":
            updated = surface_binding_layout(changed, language)
            expected = [
                (mapped(a, edits), mapped(a, edits) + len(value))
                for a, b in layout["function_name"] + layout["function_references"]
            ]
            if sorted(
                map(tuple, updated["function_name"] + updated["function_references"])
            ) != sorted(expected):
                skipped["renamed_function_unverified"] += 1
                continue
        result.append(
            dict(
                kind=kind,
                edits=edits,
                raw_source=changed,
                rule=rule,
                binding_token_index=unit_id,
                added_comment_tokens=added,
                strict_semantic=kind != "function_name",
                function_name_input_only=kind == "function_name",
                validation="parse, lexical tokens, physical lines and binding correspondence",
            )
        )
        seen.add(changed)
    skipped["proposals_attempted"] += attempt + 1
    return result, dict(skipped)


def resolve_language(source: str, language: str, file_name: str = "") -> str:
    suffix = Path(file_name).suffix
    if suffix == ".c":
        return "c"
    if suffix == ".C" or suffix.lower() in {
        ".cc",
        ".cpp",
        ".cxx",
        ".c++",
        ".hpp",
        ".hh",
        ".hxx",
        ".h++",
    }:
        return "cpp"
    if language in {"c", "cpp"}:
        return language
    encoded = source.encode()
    return min(
        ("c", "cpp"),
        key=lambda lang: (
            sum(
                n.is_error or n.is_missing for n in walk(parser_for(lang).parse(encoded).root_node)
            ),
            lang,
        ),
    )


STATEMENTS = {
    "declaration",
    "expression_statement",
    "return_statement",
    "throw_statement",
    "goto_statement",
    "break_statement",
    "continue_statement",
}


CONTROLS = {
    "if_statement",
    "while_statement",
    "for_statement",
    "do_statement",
    "switch_statement",
    "for_range_loop",
}


def source_lines(node):
    """Zero-based physical lines touched by a nonempty AST node."""
    last = node.end_point.row - int(node.end_point.column == 0)
    return set(range(node.start_point.row, max(node.start_point.row, last) + 1))


def parse_source(source):
    """Try the two existing grammars; errors remain errors, never repaired text."""
    raw = source.encode()
    attempts = []
    for language in ("c", "cpp"):
        tree = parser_for(language).parse(raw)
        nodes = list(walk(tree.root_node))
        errors = [n for n in nodes if n.type == "ERROR" or n.is_missing]
        attempts.append(
            (
                sum(max(1, n.end_byte - n.start_byte) for n in errors),
                len(errors),
                language,
                tree,
                nodes,
            )
        )
        if not tree.root_node.has_error:
            break
    _, errors, language, tree, nodes = min(attempts, key=lambda x: x[:2])
    functions = [n for n in nodes if n.type == "function_definition"]
    function = functions[0] if len(functions) == 1 else None
    name = None
    if function is not None:
        d = function.child_by_field_name("declarator")
        while d is not None:
            if d.type in {
                "identifier",
                "field_identifier",
                "operator_name",
                "destructor_name",
            }:
                name = text(d, raw)
                break
            d = d.child_by_field_name(
                "name"
                if d.type in {"qualified_identifier", "scoped_identifier"}
                else "declarator"
            )
    strict = not tree.root_node.has_error and function is not None
    return dict(
        raw=raw,
        tree=tree,
        nodes=nodes,
        function=function,
        name=name,
        language=language,
        errors=errors,
        strict=strict,
        function_count=len(functions),
    )


def edit_hunks(vulnerable, patched):
    """Structural alignment for the audit; never replaces official Differ labels."""
    a, b = vulnerable.splitlines(keepends=True), patched.splitlines(keepends=True)
    return [
        dict(tag=t, v_start=i, v_end=j, p_start=k, p_end=l)
        for t, i, j, k, l in difflib.SequenceMatcher(
            None, a, b, autojunk=False
        ).get_opcodes()
        if t != "equal"
    ]


def statement_units(parsed):
    """Atomic statements and control headers, never the entire enclosing body."""
    units = []
    function = parsed["function"]
    for n in walk(function):
        if n.type in STATEMENTS:
            units.append(dict(node=n, lines=source_lines(n), kind=n.type))
        elif n.type in CONTROLS:
            condition = n.child_by_field_name("condition") or n.child_by_field_name(
                "right"
            )
            if condition:
                units.append(
                    dict(
                        node=condition,
                        lines=source_lines(condition),
                        kind="control_condition",
                    )
                )
    units.sort(key=lambda u: (u["node"].start_byte, u["node"].end_byte))
    bindings = function_bindings(function, parsed["raw"])
    by_name = defaultdict(list)
    for b in bindings:
        if b["supported"]:
            b["key"] = (b["name"], b["declaration"].start_byte)
            by_name[b["name"]].append(b)
    for u in units:
        refs, writes = set(), set()
        for n in walk(u["node"]):
            if n.type != "identifier":
                continue
            choices = by_name.get(text(n, parsed["raw"]), [])
            binding = next((b for b in choices if b["declaration"] == n), None)
            if binding is None:
                binding = resolve_local_binding(n, choices, parsed["raw"])
            if binding is None:
                continue
            refs.add(binding["key"])
            parent = n.parent
            if n == binding["declaration"]:
                writes.add(binding["key"])
            elif (
                parent.type == "assignment_expression"
                and parent.child_by_field_name("left") == n
            ):
                writes.add(binding["key"])
            elif parent.type == "update_expression":
                writes.add(binding["key"])
        u["refs"], u["writes"] = refs, writes
    return units, by_name
