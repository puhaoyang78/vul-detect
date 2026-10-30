"""Shared C/C++ syntax, binding-safe transformations and static audit helpers."""

from __future__ import annotations
import re
import difflib
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
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
            if n.type == "for_statement":
                # The updater (and expression initializer) is executable code,
                # not part of the condition or body. Native DDG/CFG endpoints
                # on i++ previously had no atomic AST unit at all.
                for field in ("initializer", "update"):
                    expression = n.child_by_field_name(field)
                    if expression is not None and expression.type not in STATEMENTS:
                        units.append(dict(node=expression, lines=source_lines(expression),
                                          kind="for_" + field))
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


def _unwrap_expression(node):
    while node is not None and node.type in {"parenthesized_expression", "condition_clause", "cast_expression"}:
        children=node.named_children
        node=node.child_by_field_name("value") or (children[-1] if children else None)
    return node


def _bound_access(node, raw, names):
    """A scope-resolved binding or exact field path; never an alias inference."""
    node=_unwrap_expression(node)
    if node is None:
        return None
    if node.type=="identifier":
        choices=names.get(text(node,raw),[])
        binding=next((b for b in choices if b["declaration"]==node),None)
        binding=binding or resolve_local_binding(node,choices,raw)
        return ("binding",binding["name"],binding["declaration"].start_byte) if binding else None
    if node.type=="field_expression":
        base=_bound_access(node.child_by_field_name("argument"),raw,names)
        field=node.child_by_field_name("field")
        signature=" ".join(t[0] for t in source_token_spans(text(field,raw))) if field else ""
        return ("field",base,signature) if base and field else None
    return None


def _bound_references(node, raw, names, *, skip=()):
    found=set();stack=[node] if node is not None else []
    while stack:
        child=stack.pop()
        if child.type in {"sizeof_expression","lambda_expression"} or child.type in skip:
            continue
        if child.type in {"identifier","field_expression"}:
            key=_bound_access(child,raw,names)
            if key:
                found.add(key)
            if child.type=="field_expression":
                continue
        stack.extend(child.named_children)
    return frozenset(found)


SLICE_RULES = dict(
    max_hops=2, max_extra_statements=12,
    dependencies="native intra-method Joern REACHING_DEF and CDG; CFG checked, never used as adjacency expansion",
    directions="separate forward and backward traversals; one exact cross-version mapping, then local traversal",
    path_semantics="statement-level may-dependence; no alias/context/taint-semantic state tracking; not reachableByFlows",
    alignment="equal-line anchors plus identical non-comment statement tokens; otherwise unique identical statement tokens",
    seeds="atomic statements/control conditions intersecting real edit hunks, excluding unchanged matched statements",
    unknown="parse/method/coordinate failures, zero or skipped native dataflow, aliases, external call summaries, macros and unresolved cross-function effects",
    purpose="paired Train supervision audit only; never a single-function inference input or vulnerability truth",
)


def statement_dependence_graph(source, cpg):
    """Map native Joern edges to atomic Tree-sitter statements without text search.

    CPG columns use UTF-16 units and LF-based rows; public spans use original
    UTF-8 bytes and the project's splitlines physical rows. Unverifiable starts
    are rejected. This is a may-dependence graph, not a path/alias proof.
    """
    parsed = parse_source(source)
    details=dict(tree_sitter_errors=parsed["errors"],tree_sitter_functions=parsed["function_count"],
        tree_sitter_language=parsed["language"],joern_status=cpg.get("status"),
        header_type_absent=(parsed["function"].child_by_field_name("type") is None) if parsed["function"] is not None else None)
    result = dict(status="ok", units=[], edges=[], unknown={}, native=cpg.get("counts", {}),parse_details=details,
                  node_units={}, node_positions={}, mapping_stats={}, mapping_examples=[])
    if not parsed["strict"] or cpg.get("status") != "ok":
        reasons=Counter()
        if not parsed["strict"]:
            reasons["tree_sitter_parse_unknown"]+=1
        if cpg.get("status")!="ok":
            reasons[cpg.get("status","joern_status_unknown")]+=1
        result.update(status="parse_or_method_unknown", unknown=dict(reasons))
        return result
    raw=source.encode()
    physical=[]; at=0
    for line in source.splitlines(keepends=True):
        physical.append(at); at+=len(line.encode())
    units,bindings=statement_units(parsed)
    for i,u in enumerate(units):
        n=u["node"]
        a,b=n.start_byte,n.end_byte
        result["units"].append(dict(id=i,kind=u["kind"],start_byte=a,end_byte=b,
            lines=list(range(bisect_right(physical,a)-1,bisect_right(physical,b-1))),
            signature=[t[0] for t in source_token_spans(raw[a:b].decode())]))
    starts=[u["start_byte"] for u in result["units"]]
    max_end=[]; end=0
    for u in result["units"]:
        end=max(end,u["end_byte"]);max_end.append(end)
    lf_lines=source.split("\n")
    lf_bytes=[]; at=0
    for line in lf_lines:
        lf_bytes.append(at);at+=len((line+"\n").encode())
    columns={}
    unknown=Counter()
    if not cpg.get("counts",{}).get("REACHING_DEF"):
        unknown["native_ddg_zero_or_skipped"]+=1
    if not cpg.get("counts",{}).get("CDG"):
        unknown["native_cdg_zero_or_trivial"]+=1
    mapping={}; rejected={}; positions={}
    for key,n in cpg["nodes"].items():
        if n["kind"] in {"METHOD","METHOD_RETURN","BLOCK","CONTROL_STRUCTURE","LOCAL","METHOD_PARAMETER_IN","METHOD_PARAMETER_OUT"}:
            rejected[key]="non_atomic_boundary_or_metadata";continue
        line,col=n.get("LINE_NUMBER"),n.get("COLUMN_NUMBER")
        if not isinstance(line,int) or not isinstance(col,int) or not 1<=line<=len(lf_lines):
            unknown["node_missing_coordinate"]+=1;rejected[key]="node_missing_coordinate";continue
        if line not in columns:
            offsets={1:0}; utf16=1; byte=0
            for char in lf_lines[line-1]:
                utf16+=len(char.encode("utf-16-le"))//2;byte+=len(char.encode());offsets[utf16]=byte
            columns[line]=offsets
        if col not in columns[line]:
            unknown["node_invalid_utf16_column"]+=1;rejected[key]="node_invalid_utf16_column";continue
        pos=lf_bytes[line-1]+columns[line][col]
        code=n.get("CODE","")
        # Joern truncates display CODE at 1000 characters; require its real prefix.
        prefix=code[:-3] if len(code)==1000 and code.endswith("...") else code
        if not prefix or not raw.startswith(prefix.encode(),pos):
            unknown["node_source_start_mismatch"]+=1;rejected[key]="node_source_start_mismatch";continue
        candidates=[];j=bisect_right(starts,pos)-1
        while j>=0 and max_end[j]>pos:
            u=result["units"][j]
            if u["start_byte"]<=pos<u["end_byte"]:
                candidates.append(u)
            j-=1
        if candidates:
            mapping[key]=min(candidates,key=lambda u:u["end_byte"]-u["start_byte"])["id"]
            positions[key]=[pos,pos+len(prefix.encode())]
        else:
            unknown["node_outside_atomic_statement"]+=1
            rejected[key]="node_outside_atomic_statement"
    call_units={u for key,u in mapping.items() if cpg["nodes"][key]["kind"]=="CALL"
                and not cpg["nodes"][key].get("NAME","").startswith("<operator>")}
    memory_units={u for key,u in mapping.items() if cpg["nodes"][key].get("NAME") in {
        "<operator>.fieldAccess","<operator>.indirectFieldAccess","<operator>.indirection","<operator>.indexAccess"}}
    found={}; mapping_stats=Counter(); examples=[]
    for e in cpg["edges"]:
        if e["kind"] not in {"REACHING_DEF","CDG"}:
            continue
        a,b=mapping.get(e["source"]),mapping.get(e["target"])
        mapping_stats[e["kind"]+":native"]+=1
        if a is None or b is None:
            unknown["dependency_endpoint_unmapped"]+=1
            reasons=sorted({rejected.get(n,"unknown") for n in (e["source"],e["target"]) if n not in mapping})
            mapping_stats[e["kind"]+(":boundary_unmapped" if reasons==["non_atomic_boundary_or_metadata"] else ":internal_unmapped")]+=1
            for reason in reasons:
                mapping_stats[e["kind"]+":lost:"+reason]+=1
            if len(examples)<12 and "non_atomic_boundary_or_metadata" not in reasons:
                examples.append(dict(edge=e,reasons=reasons,nodes=[cpg["nodes"][n] for n in (e["source"],e["target"])]))
            continue
        mapping_stats[e["kind"]+":mapped_endpoints"]+=1
        if any(result["units"][i]["kind"].startswith("for_") for i in (a,b)):
            mapping_stats[e["kind"]+":for_header_recovered_endpoints"]+=1
        if a==b:
            mapping_stats[e["kind"]+":within_statement"]+=1
            continue
        variable=e.get("variable","")
        key=(a,b,e["kind"],variable)
        reasons=[]
        if e["kind"]=="REACHING_DEF":
            if re.search(r"->|\[|\*|\.",variable) or a in memory_units or b in memory_units:
                reasons.append("alias_or_memory_state_unknown")
            if a in call_units or b in call_units:
                reasons.append("external_call_summary_unknown")
            for nid in (e["source"],e["target"]):
                n=cpg["nodes"][nid]
                if n["kind"]=="IDENTIFIER" and n.get("NAME") not in bindings:
                    reasons.append("global_or_macro_binding_unknown")
        found.setdefault(key,dict(source=a,target=b,kind=e["kind"],variable=variable,
            origin="joern_native",joern_nodes=[e["source"],e["target"]],
            certainty="unknown" if reasons else "native_may_dependence",unknown=sorted(set(reasons))))
    for e in found.values():mapping_stats[e["kind"]+":distinct_statement_edges"]+=1
    result.update(edges=list(found.values()),unknown=dict(unknown),mapped_nodes=len(mapping),
                  node_units=mapping,node_positions=positions,mapping_stats=dict(mapping_stats),mapping_examples=examples)
    return result


def align_statement_units(before, after, graph_before, graph_after):
    """Exact statement correspondence, never a nearby-line or variable-name match."""
    blocks=difflib.SequenceMatcher(None,before.splitlines(),after.splitlines(),autojunk=False).get_matching_blocks()
    line_map={a+i:b+i for a,b,n in blocks for i in range(n)}
    by_sig=defaultdict(list); old_sig=Counter()
    for u in graph_after["units"]:
        by_sig[tuple(u["signature"])].append(u)
    for u in graph_before["units"]:
        old_sig[tuple(u["signature"])]+=1
    matches=[]
    for u in graph_before["units"]:
        sig=tuple(u["signature"])
        pool=[v for v in by_sig[sig] if v["kind"]==u["kind"]]
        mapped=[line_map.get(n) for n in u["lines"]]
        anchored=[v for v in pool if None not in mapped and mapped==v["lines"]]
        if len(anchored)==1:
            matches.append(dict(v=u["id"],p=anchored[0]["id"],origin="equal_line_and_tokens"))
        elif len(pool)==old_sig[sig]==1:
            matches.append(dict(v=u["id"],p=pool[0]["id"],origin="unique_identical_statement_tokens"))
    # No two source statements may claim the same counterpart.
    duplicate=Counter(m["p"] for m in matches)
    return [m for m in matches if duplicate[m["p"]]==1]


def bounded_dependence_slice(graph, seeds, *, roots=None, retained=None):
    """Union of bounded forward/backward slices, with provenance for every unit."""
    from collections import deque
    selected=dict(retained or {})
    selected.update({i:dict(seed=True,path=[],direction="seed") for i in sorted(set(seeds))})
    initial={i:selected[i] for i in sorted(set(seeds))}
    budget=SLICE_RULES["max_extra_statements"]
    clipped=False
    for i,provenance in sorted((roots or {}).items(),key=lambda t:(len(t[1]["path"]),t[0])):
        if i in selected:
            continue
        if len(selected)-len(set(seeds))>=budget:
            clipped=True;continue
        selected[i]=dict(seed=False,**provenance)
        initial[i]=selected[i]
    outgoing,incoming=defaultdict(list),defaultdict(list)
    for e in graph["edges"]:
        outgoing[e["source"]].append(e);incoming[e["target"]].append(e)
    for direction,adj,next_key in (("forward",outgoing,"target"),("backward",incoming,"source")):
        queue=deque((i,0,initial[i]["path"]) for i in sorted(initial))
        seen={i for i,_,_ in queue}
        while queue:
            i,depth,path=queue.popleft()
            if depth>=SLICE_RULES["max_hops"]:
                continue
            for e in sorted(adj[i],key=lambda e:(e[next_key],e["kind"],e.get("variable",""))):
                j=e[next_key]
                if j in seen:
                    continue
                seen.add(j)
                step=dict(e,traversal=direction)
                new_path=path+[step]
                if j not in selected:
                    if len(selected)-len(set(seeds))>=budget:
                        clipped=True;continue
                    selected[j]=dict(seed=False,path=new_path,direction=direction)
                queue.append((j,depth+1,new_path))
    return dict(selected=selected,limit_reached=clipped)


def bidirectional_pair_slice(before, after, graph_before, graph_after, *, allow_mapping=True):
    """Diff is used here only for paired supervision generation, never inference."""
    graphs=dict(v=graph_before,p=graph_after)
    changed=dict(v=set(),p=set())
    for h in edit_hunks(before,after):
        changed["v"].update(range(h["v_start"],h["v_end"]))
        changed["p"].update(range(h["p_start"],h["p_end"]))
    matches=align_statement_units(before,after,graph_before,graph_after) if allow_mapping else []
    aligned={side:{m[side]:m for m in matches} for side in ("v","p")}
    seeds={side:[u["id"] for u in g["units"] if changed[side]&set(u["lines"])
                 and aligned[side].get(u["id"],{}).get("origin")!="equal_line_and_tokens"]
           for side,g in graphs.items()}
    first={side:bounded_dependence_slice(g,seeds[side]) for side,g in graphs.items()}
    final={}
    for side,other in (("v","p"),("p","v")):
        roots={}
        for i,selection in first[other]["selected"].items():
            match=aligned[other].get(i)
            if match is not None and not selection["seed"]:
                j=match[side]
                step=dict(kind="VERSION_MAP",source=i,target=j,source_side=other,target_side=side,
                          origin=match["origin"],certainty="exact_syntax_correspondence")
                roots[j]=dict(path=[dict(e,side=other) for e in selection["path"]]+[step],direction="cross_version")
        sliced=bounded_dependence_slice(graphs[side],seeds[side],roots=roots,retained=first[side]["selected"])
        units=graphs[side]["units"]
        selected=sliced["selected"]
        final[side]=dict(status=graphs[side]["status"],seed_units=seeds[side],
            identical_token_seed_units=[i for i in seeds[side] if i in aligned[side]],
            region_lines=sorted({n for i in selected for n in units[i]["lines"]}),
            unknown_dependency_lines=sorted({n for i,choice in selected.items()
                if any(e.get("certainty")=="unknown" for e in choice["path"]) for n in units[i]["lines"]}),
            direct_lines=sorted({n for i in seeds[side] for n in units[i]["lines"]}),
            selected_units=[dict(units[i],**selected[i]) for i in sorted(selected)],
            mapped_roots=sum(bool(choice["path"]) and choice["path"][-1]["kind"]=="VERSION_MAP" for choice in selected.values()),
            mapped_root_candidates=len(roots),limit_reached=sliced["limit_reached"] or first[side]["limit_reached"],
            unknown=graphs[side]["unknown"],native=graphs[side]["native"],
            parse_details=graphs[side].get("parse_details"))
    return dict(v=final["v"],p=final["p"],alignment_count=len(matches),
                mapping_allowed=allow_mapping,changed_lines={k:sorted(v) for k,v in changed.items()})


TARGETED_SLICE_RULES = dict(
    scope="pure additions with a newly added if condition; other repairs are out of scope",
    max_hops=2, max_extra_statements=12,
    targets="bound array indices, null-checked dereferences, integer calculations with a native memory-consumer value path, explicit memory length operands",
    control="native CDG path through conditions only plus native CFG dominance; no forward sibling expansion",
    data="backward native REACHING_DEF, exact bound operand and actual write required",
    alignment="identical atomic statement, corresponding parameter/declaration bindings and unchanged control context",
    budget="complete target/control/data paths, ordered by uncertainty and path length; never partial paths",
    unknown="unresolved binding, alias, macro, call effect, ambiguous correspondence or missing native evidence",
    checked_values="array-element predicates and opaque call results stay unknown; their address indices/arguments are not the checked values",
    development_split="train", test_used=False, strong_labels_generated=False,
)


def _targeted_facts(source, graph):
    parsed=parse_source(source)
    if not parsed["strict"]:
        return None
    units,names=statement_units(parsed);raw=parsed["raw"]
    def refs(n):
        return _bound_references(n,raw,names)
    def sig(n):
        return tuple(t[0] for t in source_token_spans(text(n,raw))) if n else ()
    def containing(n):
        candidates=[i for i,u in enumerate(units) if u["node"].start_byte<=n.start_byte and n.end_byte<=u["node"].end_byte]
        return min(candidates,key=lambda i:units[i]["node"].end_byte-units[i]["node"].start_byte) if candidates else None
    def context(n, ignored=()):
        chain=[];child=n;parent=n.parent
        while parent and parent!=parsed["function"]:
            if parent.type in CONTROLS and parent.start_byte not in ignored:
                cond=parent.child_by_field_name("condition") or parent.child_by_field_name("right")
                # The condition's own if is not an enclosing control context.
                if cond is None or not (cond.start_byte<=n.start_byte and n.end_byte<=cond.end_byte):
                    role=next((f for f in ("consequence","alternative","body")
                               if parent.child_by_field_name(f)==child),"body")
                    chain.append((parent.type,sig(cond),role))
            if parent.type.startswith("preproc_") or parent.type in {"case_statement","labeled_statement"}:
                chain.append((parent.type,sig(parent.child_by_field_name("condition")),"unknown"))
            child=parent;parent=parent.parent
        return tuple(reversed(chain))
    facts=[]
    for u in units:
        node=u["node"];writes=set();spellings={};escapes=set()
        for n in walk(node):
            if n.type in {"identifier","field_expression"}:
                key=_bound_access(n,raw,names)
                if key:
                    spellings["".join(sig(n))]=key
            if n.type in {"assignment_expression","update_expression"}:
                key=_bound_access(n.child_by_field_name("left") or n.child_by_field_name("argument"),raw,names)
                if key:
                    writes.add(key)
            if n.type=="init_declarator":
                key=_bound_access(declarator_identifier(n.child_by_field_name("declarator")),raw,names)
                if key:
                    writes.add(key)
            if n.type=="pointer_expression" and text(n.child_by_field_name("operator"),raw)=="&":
                escapes.update(refs(n.child_by_field_name("argument")))
        facts.append(dict(node=node,reads=refs(node),writes=writes,spellings=spellings,escapes=escapes,
                          calls=[n for n in walk(node) if n.type=="call_expression"]))
    checks=[];operations=[]
    def null_refs(n):
        n=_unwrap_expression(n)
        if n is None:
            return frozenset()
        op=n.child_by_field_name("operator")
        op=text(op,raw) if op else ""
        if n.type=="unary_expression" and op=="!":
            key=_bound_access(n.child_by_field_name("argument"),raw,names)
            return frozenset({key}) if key else frozenset()
        if n.type=="binary_expression":
            a,b=n.child_by_field_name("left"),n.child_by_field_name("right")
            if op in {"==","!="}:
                key=_bound_access(a,raw,names) if sig(_unwrap_expression(b)) in {("0",),("NULL",),("nullptr",)} else _bound_access(b,raw,names) if sig(_unwrap_expression(a)) in {("0",),("NULL",),("nullptr",)} else None
                return frozenset({key}) if key else frozenset()
            if op in {"&&","||"}:
                return null_refs(a)|null_refs(b)
        return frozenset()
    integral=re.compile(r"(?:u?int(?:8|16|32|64)_t|size_t|ssize_t|unsigned(?:\s+(?:int|long|short|char))*|signed(?:\s+(?:int|long|short|char))*|int|long(?:\s+(?:long|int))*|short(?:\s+int)?|char)")
    def integer_destination(n):
        parent=n.parent
        while parent and parent.type not in {"assignment_expression","init_declarator"} | STATEMENTS:
            parent=parent.parent
        if parent is None:
            return False
        lhs=parent.child_by_field_name("left") if parent.type=="assignment_expression" else declarator_identifier(parent.child_by_field_name("declarator")) if parent.type=="init_declarator" else None
        key=_bound_access(lhs,raw,names)
        if key is None or key[0]!="binding":
            return False
        binding=next((b for b in names.get(key[1],[]) if b["declaration"].start_byte==key[2]),None)
        typ=binding["statement"].child_by_field_name("type") if binding else None
        return typ is not None and integral.fullmatch(text(typ,raw).strip()) is not None
    for n in walk(parsed["function"]):
        uid=containing(n)
        if n.type=="if_statement":
            cond=n.child_by_field_name("condition")
            if cond:
                checks.append(dict(node=n,condition=cond,unit=containing(cond),
                    refs=_bound_references(cond,raw,names,skip={"subscript_expression","call_expression"}),
                    null_refs=null_refs(cond)))
        kind=None;operands=frozenset();null_operands=frozenset()
        if n.type=="subscript_expression":
            kind="array_access"
            index=n.child_by_field_name("index") or next((c for c in n.named_children if c.type=="subscript_argument_list"),None)
            operands=refs(index);null_operands=refs(n.child_by_field_name("argument"))
        elif n.type=="pointer_expression" and text(n.child_by_field_name("operator"),raw)=="*":
            kind="dereference";operands=refs(n.child_by_field_name("argument"))
        elif n.type=="field_expression" and text(n.child_by_field_name("operator"),raw)=="->":
            kind="dereference";operands=refs(n.child_by_field_name("argument"))
        elif n.type=="binary_expression" and text(n.child_by_field_name("operator"),raw) in {"+","-","*","/","%","<<"} and integer_destination(n):
            kind="integer_calculation";operands=refs(n)
        elif n.type=="call_expression":
            callee="".join(sig(n.child_by_field_name("function")))
            args=n.child_by_field_name("arguments")
            args=list(args.named_children) if args else []
            # Explicit length-argument contracts, never an API-presence match.
            position={"memcpy":2,"memmove":2,"memset":2,"strncpy":2,
                      "malloc":0,"calloc":0,"realloc":1,"kmalloc":0}.get(callee)
            if position is not None and len(args)>position:
                kind="memory_length";operands=refs(args[position])
        if kind and uid is not None and (operands or null_operands):
            operations.append(dict(unit=uid,node=n,kind=kind,refs=operands,null_refs=null_operands))
    return dict(parsed=parsed,units=facts,names=names,checks=checks,operations=operations,
                context=context,signature=sig)


def targeted_pair_slice(before, after, graph_before, graph_after, cpg_before, cpg_after, *, allow_mapping=True):
    """Conservative training-time repair candidates, not strong location labels.

    Only targets constrained by an added check are transferred. The graph is
    never expanded to all statements controlled by that check.
    """
    from collections import deque
    graphs=dict(v=graph_before,p=graph_after);cpgs=dict(v=cpg_before,p=cpg_after)
    changed={s:set() for s in graphs}
    for h in edit_hunks(before,after):
        changed["v"].update(range(h["v_start"],h["v_end"]))
        changed["p"].update(range(h["p_start"],h["p_end"]))
    result=dict(status="unknown",reasons={},targets=[],mapping_allowed=allow_mapping)
    for side,g in graphs.items():
        result[side]=dict(status=g["status"],region_lines=[],selected_units=[],unknown_dependency_lines=[],limit_reached=False)
    old_changed="\n".join(line for i,line in enumerate(before.splitlines()) if i in changed["v"])
    if source_token_spans(old_changed) or not changed["p"]:
        result.update(status="out_of_scope",reasons={"not_pure_addition":1});return result
    if not allow_mapping or any(g["status"]!="ok" for g in graphs.values()):
        result["reasons"]={"pair_or_parse_unknown":1};return result
    facts={s:_targeted_facts(src,graphs[s]) for s,src in (("v",before),("p",after))}
    fp,fv=facts["p"],facts["v"]
    checks=[c for c in fp["checks"] if c["unit"] is not None and changed["p"] & set(graph_after["units"][c["unit"]]["lines"])]
    gates={c["node"].start_byte for c in checks}
    if not checks:
        result.update(status="out_of_scope",reasons={"no_added_if_condition":1});return result
    matches=align_statement_units(before,after,graph_before,graph_after)
    # Bindings must correspond independently of the target's text position.
    bindings={}
    for bs in fv["names"].values():
        for old in bs:
            pool=[b for b in fp["names"].get(old["name"],[]) if
                  b["parameter_index"]==old["parameter_index"] and
                  fv["signature"](old["statement"])==fp["signature"](b["statement"]) and
                  fv["context"](old["statement"])==fp["context"](b["statement"],gates)]
            if len(pool)==1:
                bindings[("binding",old["name"],old["declaration"].start_byte)]=("binding",pool[0]["name"],pool[0]["declaration"].start_byte)
    def translated(key):
        return bindings.get(key) if key[0]=="binding" else ("field",translated(key[1]),key[2]) if translated(key[1]) else None
    aligned={}
    for m in matches:
        a,b=fv["units"][m["v"]],fp["units"][m["p"]]
        old_refs={translated(k) for k in a["reads"]|a["writes"]}
        if None not in old_refs and old_refs==set(b["reads"]|b["writes"]) and fv["context"](a["node"])==fp["context"](b["node"],gates):
            aligned[m["p"]]=m["v"]
    reasons=Counter();selected={s:{} for s in graphs};budget=TARGETED_SLICE_RULES["max_extra_statements"]
    reasons["added_checks_without_resolved_tested_value"]=sum(not c["refs"] for c in checks)
    if not fp["operations"]:
        reasons["no_supported_bound_memory_operation"]=1
    seeds={u["id"] for u in graph_after["units"] if changed["p"]&set(u["lines"])}
    for i in seeds:
        selected["p"][i]=dict(role="direct_edit",unknown=[],paths=[])
    cdg=defaultdict(list);incoming={s:defaultdict(list) for s in graphs}
    for s,g in graphs.items():
        for e in g["edges"]:
            incoming[s][e["target"]].append(e)
            if s=="p" and e["kind"]=="CDG":
                cdg[e["source"]].append(e)
    cfg=defaultdict(list)
    for e in cpg_after["edges"]:
        if e["kind"]=="CFG":
            cfg[e["source"]].append(e["target"])
    entries=[k for k,n in cpg_after["nodes"].items() if n["kind"]=="METHOD"]
    def reachable(blocked=()):
        seen=set();pending=list(entries);blocked=set(blocked)
        while pending:
            i=pending.pop()
            if i in seen or i in blocked:
                continue
            seen.add(i);pending.extend(cfg[i])
        return seen
    full_reach=reachable();without={}
    def control_path(a,b):
        queue=deque([(a,[])]);seen={a}
        while queue:
            i,path=queue.popleft()
            if len(path)>=TARGETED_SLICE_RULES["max_hops"]:
                continue
            for e in cdg[i]:
                j=e["target"];new=path+[e]
                if j==b:
                    return new
                if j not in seen and graph_after["units"][j]["kind"]=="control_condition":
                    seen.add(j);queue.append((j,new))
        return []
    def data_sources(side, uid, needed, depth=0, visited=frozenset()):
        paths=[]
        if depth>=TARGETED_SLICE_RULES["max_hops"]:
            return paths
        for e in incoming[side][uid]:
            if e["kind"]!="REACHING_DEF" or e["source"] in visited:
                continue
            source,target=facts[side]["units"][e["source"]],facts[side]["units"][uid]
            key=target["spellings"].get("".join(t[0] for t in source_token_spans(e.get("variable",""))))
            if key not in needed or key not in source["writes"]:
                continue
            # Native call summaries and possible memory aliases are not value definitions.
            if e.get("certainty")=="unknown":
                # Statement lifting can flag an otherwise exact scalar argument
                # or index because another expression in that unit is a call/
                # memory access. Verify the actual native use endpoint instead.
                native=cpgs[side]["nodes"].get(e.get("joern_nodes",[None,None])[1],{})
                exact_scalar=(key[0]=="binding" and not source["calls"] and
                    native.get("kind")=="IDENTIFIER" and native.get("CODE")==key[1] and native.get("NAME")==key[1])
                if not exact_scalar:
                    reasons["data_definition_alias_or_call_unknown"]+=1;continue
            paths.append([e])
            paths.extend(path+[e] for path in data_sources(side,e["source"],source["reads"],depth+1,visited|{uid}))
        return paths
    proposals=[]
    for check in checks:
        ci=check["unit"]
        blocked={n for n,u in graph_after["node_units"].items() if u==ci}
        without[ci]=reachable(blocked)
        for operation in fp["operations"]:
            ti=operation["unit"]
            if ti in seeds:
                continue
            if operation["kind"]=="dereference" and not (operation["refs"]&check["null_refs"]):
                continue
            reasons["operation_candidates"]+=1
            operands=operation["refs"]|(operation["null_refs"]&check["null_refs"])
            if operation["kind"]=="integer_calculation" and not any(
                    other["kind"] in {"array_access","memory_length"} and
                    any(p[0]["source"]==ti for p in data_sources("p",other["unit"],other["refs"]))
                    for other in fp["operations"] if other["unit"]!=ti):
                reasons["integer_calculation_without_memory_value_path"]+=1;continue
            overlap=check["refs"]&operands
            derived=data_sources("p",ti,operation["refs"])
            derived=[p for p in derived if fp["units"][p[0]["source"]]["reads"]&check["refs"]]
            if not overlap and not derived:
                reasons["no_bound_operand_or_value_path"]+=1;continue
            path=control_path(ci,ti)
            if not path:
                reasons["no_target_control_dependence"]+=1;continue
            opnodes={k for k,span in graph_after["node_positions"].items()
                     if operation["node"].start_byte<=span[0] and span[1]<=operation["node"].end_byte and k in full_reach}
            if not opnodes or opnodes&without[ci]:
                reasons["cfg_dominance_not_established"]+=1;continue
            shared=overlap or fp["units"][derived[0][0]["source"]]["reads"]&check["refs"]
            between=[u for u in fp["units"] if check["condition"].end_byte<u["node"].start_byte<operation["node"].start_byte and u is not fp["units"][ti]]
            if any(shared&(u["writes"]|u["escapes"]) for u in between):
                reasons["checked_value_changed_or_escaped"]+=1;continue
            if any(k[0]=="field" for k in shared) and any(u["calls"] for u in between):
                reasons["intervening_call_field_state_unknown"]+=1;continue
            if ti not in aligned:
                reasons["target_binding_or_context_correspondence_unknown"]+=1;continue
            if any(k[0]=="field" for k in shared):
                unknown=["field_alias_state_not_proved"]
            else:
                unknown=[]
            chain=[] if overlap else min(derived,key=lambda p:(len(p),p[0]["source"]))
            proposals.append(dict(p=ti,v=aligned[ti],kind=operation["kind"],check=ci,
                control_path=path,data_path=chain,unknown=unknown,
                correspondence=dict(origin="identical_statement_bound_operands_and_control_context",
                    vulnerable_span=[graph_before["units"][aligned[ti]][k] for k in ("start_byte","end_byte")],
                    patched_span=[graph_after["units"][ti][k] for k in ("start_byte","end_byte")]),
                cfg=dict(condition_nodes=sorted(blocked),operation_nodes=sorted(opnodes),dominates=True)))
    # Rank complete relation packages, not the order of unrelated graph nodes.
    seen_targets=set()
    for proposal in sorted(proposals,key=lambda p:(len(p["unknown"]),len(p["control_path"])+len(p["data_path"]),p["kind"],p["p"])):
        identity=(proposal["check"],proposal["p"],proposal["v"])
        if identity in seen_targets:
            continue
        seen_targets.add(identity)
        core={"p":{proposal["p"]}|{e["source"] for e in proposal["control_path"]}|{e["source"] for e in proposal["data_path"]},"v":{proposal["v"]}}
        if any(len((set(selected[s])|core[s])-(seeds if s=="p" else set()))>budget for s in graphs):
            reasons["complete_target_path_exceeds_budget"]+=1
            for s in graphs: result[s]["limit_reached"]=True
            continue
        result["targets"].append(proposal)
        for side,ids in core.items():
            for i in ids:
                selected[side].setdefault(i,dict(role="target_relation",unknown=proposal["unknown"],paths=[]))
                selected[side][i]["unknown"]=sorted(set(selected[side][i]["unknown"])|set(proposal["unknown"]))
                selected[side][i]["paths"].append(dict(kind="target",target=identity,side=side))
        # Sources and necessary conditions of the chosen target only; no CDG siblings.
        for side in ("v","p"):
            uid=proposal[side];f=facts[side]["units"][uid]
            supports=data_sources(side,uid,f["reads"])
            supports += [[e] for e in incoming[side][uid] if e["kind"]=="CDG" and graphs[side]["units"][e["source"]]["kind"]=="control_condition"]
            for path in sorted(supports,key=lambda p:(len(p),p[0]["source"])):
                ids={e["source"] for e in path}|{e["target"] for e in path}
                if len((set(selected[side])|ids)-(seeds if side=="p" else set()))>budget:
                    result[side]["limit_reached"]=True;reasons["complete_support_path_exceeds_budget"]+=1;continue
                for i in ids:
                    selected[side].setdefault(i,dict(role="target_source_or_condition",unknown=[],paths=[]))
                    selected[side][i]["unknown"]=sorted(set(selected[side][i]["unknown"])|set(proposal["unknown"]))
                    selected[side][i]["paths"].append(dict(kind="support",target=uid,edges=path))
    for side,g in graphs.items():
        result[side].update(region_lines=sorted({n for i in selected[side] for n in g["units"][i]["lines"]}),
            selected_units=[dict(g["units"][i],**selected[side][i]) for i in sorted(selected[side])],
            unknown_dependency_lines=sorted({n for i,c in selected[side].items() if c["unknown"] for n in g["units"][i]["lines"]}))
    result.update(status="ok" if result["targets"] else "unknown",reasons=dict(reasons),
                  added_checks=len(checks),strict_correspondences=len(aligned))
    return result


STATIC_RELATION_RULES = dict(
    scope="one unmodified C/C++ function; strict parse and scope-resolved local/parameter bindings",
    patterns=["boundary_check_relation", "null_check_relation", "integer_range_relation",
              "memory_extent_relation", "resource_lifetime_relation", "call_argument_relation", "control_call_relation"],
    guards="enclosing if/for/while branch, or same-block prior if with unconditional return/throw; shared bound value required",
    definitions="latest enclosing lexical definition; reject intervening assignment, update, address escape, goto/label/case; not a CFG dominance proof",
    fields="exact access path, never all fields sharing an object; intervening calls invalidate field-state checks",
    memory="memcpy/memmove/memset/strncpy with destination array/allocation definition and explicit length argument",
    capacity="all nested array dimensions, in elements; sizeof(array) in bytes; symbolic lengths stay unknown",
    resources="malloc/calloc/realloc definition to free/fclose/close; or same-block free/release to dereference, with no intervening write/goto",
    unknown="no alias, global binding, macro recovery, type-width proof, call summary, CFG/DDG/CDG or vulnerability verdict",
    lines="zero-based physical lines and UTF-8 byte offsets; presentation may use one-based lines",
    development_split="train", test_used=False,
)


def static_relations(source):
    """Single-function syntactic operation/check/definition relations, not bugs.

    No patch, label or location hint is accepted. Missing bindings, ambiguous
    paths and parse errors stay unknown; no alias or interprocedural inference.
    Byte ranges and physical lines always refer to the unmodified input.
    Integer tags use local declaration hints, not resolved C++ operator types.
    """
    parsed = parse_source(source)
    failures = Counter()
    result = dict(status="ok" if parsed["strict"] else "parse_unknown",
                  language=parsed["language"], errors=parsed["errors"],
                  function_count=parsed["function_count"], name=parsed["name"],
                  relations=[], operations={}, failures={}, ast_units=[], ast_calls=[], ast_operations=[])
    if not parsed["strict"]:
        result["error_spans"] = [dict(start_byte=n.start_byte, end_byte=n.end_byte,
            lines=sorted(source_lines(n)), kind=n.type) for n in parsed["nodes"] if n.is_error or n.is_missing]
        return result
    raw, function = parsed["raw"], parsed["function"]
    physical_starts=[]
    at=0
    for line in source.splitlines(keepends=True):
        physical_starts.append(at)
        at+=len(line.encode())
    units, names = statement_units(parsed)
    bindings = [b for bs in names.values() for b in bs]

    unwrap = _unwrap_expression

    def signature(n):
        return " ".join(t[0] for t in source_token_spans(text(n, raw))) if n else ""

    def span(n):
        first=max(0,bisect_right(physical_starts,n.start_byte)-1)
        last=max(first,bisect_left(physical_starts,n.end_byte)-1)
        return dict(start_byte=n.start_byte, end_byte=n.end_byte, lines=list(range(first,last+1)),
                    code=text(n, raw)[:300], ast_type=n.type)

    def atom(n):
        return _bound_access(n,raw,names)

    reference_cache = {}
    def refs(n):
        if n in reference_cache:
            return reference_cache[n]
        reference_cache[n] = _bound_references(n,raw,names)
        return reference_cache[n]

    def block(n):
        while n and n.type != "compound_statement":
            n = n.parent
        return n

    def contains(a, b):
        return a is not None and a.start_byte <= b.start_byte and b.end_byte <= a.end_byte

    def terminates(n):
        if n is None:
            return False
        if n.type == "compound_statement":
            children = [c for c in n.named_children if c.type != "comment"]
            return bool(children) and terminates(children[-1])
        return n.type in {"return_statement", "throw_statement"}

    def zero_value(n, key):
        n = unwrap(n)
        if n is None:
            return None
        if atom(n) == key:
            return False
        op = n.child_by_field_name("operator")
        op = text(op,raw) if op else ""
        if n.type == "unary_expression" and op == "!":
            value = zero_value(n.child_by_field_name("argument"),key)
            return None if value is None else not value
        if n.type == "binary_expression":
            left, right = n.child_by_field_name("left"), n.child_by_field_name("right")
            if op in {"==", "!="} and ((atom(left)==key and signature(unwrap(right)) in {"0","NULL","nullptr"})
                    or (atom(right)==key and signature(unwrap(left)) in {"0","NULL","nullptr"})):
                return op == "=="
            a,b = zero_value(left,key),zero_value(right,key)
            if op == "&&":
                return False if a is False or b is False else True if a is True and b is True else None
            if op == "||":
                return True if a is True or b is True else False if a is False and b is False else None
        return None

    nodes = list(walk(function))
    writes, definitions, calls = [], [], []
    for n in nodes:
        if n.type == "init_declarator":
            key = atom(declarator_identifier(n.child_by_field_name("declarator")))
            value = n.child_by_field_name("value")
            if key and value:
                definitions.append((key,n,value))
        if n.type in {"assignment_expression", "update_expression"}:
            lhs = n.child_by_field_name("left") or n.child_by_field_name("argument")
            key = atom(lhs)
            if key:
                writes.append((key,n))
                op = n.child_by_field_name("operator")
                if n.type == "assignment_expression" and op and text(op,raw) == "=":
                    definitions.append((key,n,n.child_by_field_name("right")))
        if n.type == "pointer_expression" and signature(n.child_by_field_name("operator")) == "&":
            key = atom(n.child_by_field_name("argument"))
            if key:
                writes.append((key,n))  # An escaped address may invalidate a value.
        if n.type == "call_expression":
            callee = n.child_by_field_name("function")
            args = n.child_by_field_name("arguments")
            calls.append((n,signature(callee),list(args.named_children) if args else []))
    for b in bindings:
        parent = b["declaration"].parent
        if parent.type == "array_declarator":
            while parent.parent.type=="array_declarator":
                parent=parent.parent
            definitions.append((("binding",b["name"],b["declaration"].start_byte),parent,None))
    definitions_by_key, writes_by_key = defaultdict(list), defaultdict(list)
    for key,n,value in definitions:
        definitions_by_key[key].append((n,value))
    for values in definitions_by_key.values():
        values.sort(key=lambda t:t[0].end_byte)
    definition_ends = {k:[n.end_byte for n,v in values] for k,values in definitions_by_key.items()}
    for key,n in writes:
        writes_by_key[key].append(n.start_byte)
    for values in writes_by_key.values():
        values.sort()
    def written(keys, start, end):
        for key in keys:
            positions = writes_by_key.get(key,[])
            at = bisect_left(positions,start)
            if at<len(positions) and positions[at]<end:
                return True
        return False
    def latest_definition(key, operation):
        at = bisect_right(definition_ends.get(key,[]),operation.start_byte)-1
        if at<0:
            return None
        node,value = definitions_by_key[key][at]
        db = block(node)
        if not contains(db,operation) or written({key},node.end_byte,operation.start_byte):
            failures["definition_path_or_reassignment_unknown"] += 1
            return None
        if position_between(labels,node.end_byte,operation.start_byte):
            failures["definition_unstructured_control_unknown"] += 1
            return None
        # A conditional/loop-local assignment cannot define a later outer use.
        p = node.parent
        while p and p != db:
            if p.type in CONTROLS or p.type in {"conditional_expression","lambda_expression","case_statement"}:
                return None
            p = p.parent
        return node,value

    guards = []
    for n in nodes:
        if n.type not in CONTROLS:
            continue
        cond = n.child_by_field_name("condition")
        if not cond:
            continue
        targets = refs(cond)
        if not targets:
            failures["condition_without_resolved_local_or_field"] += 1
        if n.type == "if_statement":
            guards.append((n,cond,n.child_by_field_name("consequence"),True,"enclosing_branch",targets))
            alternative = n.child_by_field_name("alternative")
            if alternative:
                guards.append((n,cond,alternative,False,"enclosing_branch",targets))
            if terminates(n.child_by_field_name("consequence")) and not alternative:
                guards.append((n,cond,n.parent,False,"early_return",targets))
        elif n.type in {"for_statement","while_statement"}:
            guards.append((n,cond,n.child_by_field_name("body"),True,"loop_condition",targets))
    guard_index = defaultdict(set)
    for i,(_,_,scope,_,_,targets) in enumerate(guards):
        if scope:
            for key in targets:
                guard_index[scope.id,key].add(i)
    labels = sorted(n.start_byte for n in nodes if n.type in {"labeled_statement","goto_statement","case_statement"})
    call_positions = sorted(c.start_byte for c,_,_ in calls)
    def position_between(positions,start,end):
        at = bisect_left(positions,start)
        return at<len(positions) and positions[at]<end

    operations = []
    memory_apis = {"memcpy":(0,2), "memmove":(0,2), "memset":(0,2), "strncpy":(0,2)}
    alloc_apis = {"malloc", "calloc", "realloc"}
    releases = []
    for n,name,args in calls:
        operations.append(("call",n,set().union(*(refs(a) for a in args)) if args else set(),None))
        if name in memory_apis and len(args) == 3:
            dest,length = args[0],args[2]
            operations.append(("memory_span",n,refs(length),atom(dest)))
        if name in {"free","fclose","close"} and len(args)==1 and atom(args[0]):
            releases.append((n,name,atom(args[0])))
    binding_by_key = {("binding",b["name"],b["declaration"].start_byte):b for b in bindings}
    for n in nodes:
        if n.type == "subscript_expression":
            index = n.child_by_field_name("index") or next((c for c in n.named_children if c.type=="subscript_argument_list"),None)
            operations.append(("indexed_memory",n,refs(index),atom(n.child_by_field_name("argument"))))
        elif n.type in {"pointer_expression","field_expression"}:
            op = n.child_by_field_name("operator")
            if op and text(op,raw) in {"*","->"}:
                operations.append(("pointer_access",n,set(),atom(n.child_by_field_name("argument"))))
        elif n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op and text(op,raw) in {"+","-","*","<<"}:
                keys=refs(n)
                known = [binding_by_key[k] for k in keys if k in binding_by_key]
                if keys and len(known)==len(keys) and all(b["statement"].child_by_field_name("type") is not None
                    and b["declaration"].parent.type not in {"pointer_declarator","array_declarator","reference_declarator"}
                    and re.fullmatch(
                    r"(?:u?int(?:8|16|32|64)_t|size_t|ssize_t|unsigned(?:\s+(?:int|long|short|char))*|signed(?:\s+(?:int|long|short|char))*|int|long(?:\s+(?:long|int))*|short(?:\s+int)?|char)",
                    text(b["statement"].child_by_field_name("type"),raw).strip())
                    for b in known):
                    operations.append(("integer_arithmetic",n,keys,None))
                else:
                    failures["arithmetic_type_or_binding_unknown"] += 1

    seen = set()
    def emit(pattern, operation, evidence, role, **extra):
        key=(pattern,operation.start_byte,evidence.start_byte,role)
        if key not in seen:
            seen.add(key)
            result["relations"].append(dict(pattern=pattern, operation=span(operation),
                evidence=span(evidence), role=role, certainty="syntactic_relation",
                safety="unknown", **extra))

    for kind,operation,keys,pointer in operations:
        if not keys and pointer is None:
            failures["operation_without_resolved_operand"] += 1
        candidate_guards = set()
        parent = operation
        while parent and parent!=function:
            for key in keys | ({pointer} if pointer else set()):
                candidate_guards.update(guard_index.get((parent.id,key),()))
            parent = parent.parent
        for i in sorted(candidate_guards):
            control,cond,scope,truth,role,targets = guards[i]
            if not contains(scope,operation) or contains(cond,operation):
                continue
            if role == "early_return" and (block(operation)!=control.parent or control.end_byte>operation.start_byte):
                continue
            overlap = (keys | ({pointer} if pointer else set())) & targets
            if not overlap:
                continue
            start = control.end_byte if role=="early_return" else scope.start_byte
            if written(overlap,start,operation.start_byte):
                failures["guard_value_written_before_operation"] += 1
                continue
            if any(k[0]=="field" for k in overlap) and position_between(call_positions,start,operation.start_byte):
                failures["intervening_call_field_state_unknown"] += 1
                continue
            if position_between(labels,start,operation.start_byte):
                failures["unstructured_control_unknown"] += 1
                continue
            if pointer in targets and kind in {"pointer_access","indexed_memory"}:
                zero=zero_value(cond,pointer)
                # Integer comparisons mentioning a pointer field are not null checks.
                if zero is not None:
                    emit("null_check_relation",operation,cond,role,
                         branch_truth=truth,zero_case="excluded" if zero!=truth else "allowed")
            if keys & targets:
                pattern={"indexed_memory":"boundary_check_relation","memory_span":"boundary_check_relation",
                         "integer_arithmetic":"integer_range_relation","call":"control_call_relation"}.get(kind)
                if pattern:
                    emit(pattern,operation,cond,role,branch_truth=truth)
        if kind == "memory_span" and pointer:
            definition=latest_definition(pointer,operation)
            if definition:
                dn,value=definition
                value_node=unwrap(value)
                allocation = value_node is not None and value_node.type=="call_expression" and signature(value_node.child_by_field_name("function")) in alloc_apis
                if dn.type=="array_declarator" or allocation:
                    dimensions=[]
                    current=dn
                    while current is not None and current.type=="array_declarator":
                        dimensions.append(signature(current.child_by_field_name("size")))
                        current=current.child_by_field_name("declarator")
                    dimensions.reverse()
                    emit("memory_extent_relation",operation,dn,"destination_capacity_definition",
                         capacity=" * ".join(dimensions) if dimensions else signature(value),
                         capacity_dimensions=dimensions,capacity_unit="elements" if dimensions else "allocation bytes",
                         capacity_bytes=f"sizeof({pointer[1]})" if dimensions and pointer[0]=="binding" else "unknown",
                         access_length=signature(next(a for c,_,args in calls if c==operation for a in args[2:3])),
                         extent_comparison="unknown")
    for operation,name,args in calls:
        for position,arg in enumerate(args):
            for key in sorted(refs(arg)):
                definition=latest_definition(key,operation)
                if definition:
                    dn,value=definition
                    if value and value.type in {"binary_expression","call_expression","sizeof_expression","cast_expression","parenthesized_expression"}:
                        emit("call_argument_relation",operation,dn,"computed_definition_to_argument",argument=position)
    for release,name,key in releases:
        definition=latest_definition(key,release)
        value=unwrap(definition[1]) if definition else None
        if value is not None and value.type=="call_expression" and signature(value.child_by_field_name("function")) in alloc_apis:
            emit("resource_lifetime_relation",release,definition[0],"allocation_to_release")
        for kind,operation,keys,pointer in operations:
            if kind not in {"pointer_access","indexed_memory"} or pointer!=key or operation.start_byte<=release.end_byte or block(operation)!=block(release):
                continue
            if release.parent.type!="expression_statement" or release.parent.parent!=block(release):
                failures["conditional_release_unknown"] += 1
                continue
            if position_between(labels,release.end_byte,operation.start_byte):
                failures["lifetime_unstructured_control_unknown"] += 1
                continue
            if not written({key},release.end_byte,operation.start_byte):
                emit("resource_lifetime_relation",operation,release,"release_to_access_candidate")
    for unit in units:
        result["ast_units"].append(dict(kind=unit["kind"], signature=signature(unit["node"]), **span(unit["node"])))
    result["ast_calls"] = [dict(name=name, arguments=[signature(a) for a in args], **span(n)) for n,name,args in calls]
    result["ast_operations"] = [dict(kind=kind,signature=signature(n),**span(n))
                                for kind,n,_,_ in operations if kind!="call"]
    result["operations"] = dict(Counter(kind for kind,_,_,_ in operations))
    result["failures"] = dict(failures)
    return result
