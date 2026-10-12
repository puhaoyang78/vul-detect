from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp


_C = Parser(Language(tree_sitter_c.language()))
_CPP = Parser(Language(tree_sitter_cpp.language()))


@dataclass(frozen=True)
class ParsedFunction:
    name: str
    parameters: tuple[str, ...]


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
    return source[node.start_byte:node.end_byte].decode(errors="replace")


def identifier(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type == "parenthesized_declarator" and node.parent is not None:
        parent = node.parent
        type_node = parent.child_by_field_name("type")
        if (
            parent.type == "function_definition"
            and type_node is not None
            and type_node.type == "type_identifier"
        ):
            return text(type_node, source)
    if node.type in {"identifier", "destructor_name", "operator_name"}:
        return text(node, source)
    if node.type == "operator_cast":
        parameters = next((n for n in walk(node) if n.type == "parameter_list"), None)
        if parameters is not None:
            return source[node.start_byte:parameters.start_byte].decode().strip()
    if node.type in {"parameter_list", "template_argument_list", "attribute_specifier"}:
        return None
    name = node.child_by_field_name("name")
    if name is not None:
        found = identifier(name, source)
        if found:
            return found
    declarator = node.child_by_field_name("declarator")
    if declarator is not None and declarator is not node:
        found = identifier(declarator, source)
        if found:
            return found
    for child in node.named_children:
        found = identifier(child, source)
        if found:
            return found
    return None


def _parameters(function: Node, source: bytes) -> tuple[str, ...]:
    declarator = function.child_by_field_name("declarator")
    parameter_list = (
        next((node for node in walk(declarator) if node.type == "parameter_list"), None)
        if declarator is not None
        else None
    )
    if parameter_list is None:
        return ()
    names: list[str] = []
    for parameter in parameter_list.named_children:
        if parameter.type not in {"parameter_declaration", "optional_parameter_declaration"}:
            continue
        name = identifier(parameter.child_by_field_name("declarator"), source)
        if name:
            names.append(name)
    return tuple(names)


def parse_function(
    source_text: str,
    language: str = "c",
    function_name: str | None = None,
) -> ParsedFunction:
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    functions = [node for node in walk(tree.root_node) if node.type == "function_definition"]
    if function_name:
        functions = [
            node
            for node in functions
            if identifier(node.child_by_field_name("declarator"), source) in {function_name, function_name.rsplit("::", 1)[-1]}
        ]
    if len(functions) != 1:
        label = f" named {function_name}" if function_name else ""
        raise ValueError(f"expected exactly one function{label}, found {len(functions)}")
    node = functions[0]
    name = identifier(node.child_by_field_name("declarator"), source)
    if not name:
        raise ValueError("function name could not be parsed")
    return ParsedFunction(name=name, parameters=_parameters(node, source))


def function_readout_span(source_text: str, language: str) -> tuple[int, int]:
    """Character extent of one complete outer function, including its signature.

    An omitted return type is a common input-fragment grammar: parse a synthetic
    type, but never send it to the encoder or infer a program type from it.
    Ambiguous/incomplete boundaries are errors, not an all-source fallback.
    Errors inside a bounded body do not by themselves invalidate its extent.
    """
    raw = source_text.encode('utf-8')
    for prefix in (b'', b'int '):
        encoded = prefix + raw
        root = parser_for(language).parse(encoded).root_node
        functions = []

        def collect(node):
            if node.type == 'function_definition':
                functions.append(node)
                return  # Nested methods/lambdas belong to the outer function.
            for child in node.named_children:
                collect(child)

        collect(root)
        if len(functions) != 1:
            continue
        function = functions[0]
        declarator = function.child_by_field_name('declarator')
        body = function.child_by_field_name('body')
        if declarator is None or body is None:
            continue
        name = identifier(declarator, encoded)
        if not name or name in {'if', 'for', 'while', 'switch', 'catch', 'sizeof'}:
            continue  # Error recovery can misparse an inner control statement as a function.
        if prefix:
            kind = function.child_by_field_name('type')
            if (function.start_byte != 0 or kind is None or kind.end_byte != 3
                    or encoded[4:declarator.start_byte].strip()):
                continue
        if (not body.children or body.children[0].type != '{'
                or body.children[-1].type != '}'
                or body.children[0].is_missing or body.children[-1].is_missing):
            raise ValueError('incomplete function body boundary')
        start, end = function.start_byte, function.end_byte
        parent = function.parent
        while parent is not None and parent.type == 'template_declaration':
            start = parent.start_byte
            parent = parent.parent
        start = max(0, start - len(prefix))
        end -= len(prefix)
        return len(raw[:start].decode('utf-8')), len(raw[:end].decode('utf-8'))
    raise ValueError('cannot establish one complete function boundary')


def single_function_language(source_text: str, file_name: str = "") -> str | None:
    """Infer C vs C++ only when a standalone function can be parsed unambiguously."""
    encoded = source_text.encode("utf-8")
    suffix = Path(file_name).suffix
    languages = (
        ("c",)
        if suffix == ".c"
        else (
            ("cpp",)
            if suffix in {".C", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
            else ("c", "cpp")
        )
    )
    for language in languages:
        root = parser_for(language).parse(encoded).root_node
        if root.has_error:
            continue
        nodes = [node for node in root.named_children if node.type != "comment"]
        if len(nodes) != 1:
            continue
        top = nodes[0]
        while top.type == "template_declaration":
            children = [
                node
                for node in top.named_children
                if node.type not in {"template_parameter_list", "comment"}
            ]
            if len(children) != 1:
                break
            top = children[0]
        if top.type != "function_definition":
            continue
        if top.child_by_field_name("type") is None:
            declarator = top.child_by_field_name("declarator")
            while declarator is not None and declarator.type != "qualified_identifier":
                declarator = declarator.child_by_field_name("declarator")
            if language != "cpp" or declarator is None:
                continue
            scope = declarator.child_by_field_name("scope")
            name = declarator.child_by_field_name("name")
            if scope is None or name is None:
                continue
            while scope.type == "qualified_identifier":
                scope = scope.child_by_field_name("name")
            if scope.type == "template_type":
                scope = scope.child_by_field_name("name")
            owner = encoded[scope.start_byte:scope.end_byte].decode()
            method = encoded[name.start_byte:name.end_byte].decode()
            if name.type != "operator_cast" and method not in {owner, "~" + owner}:
                continue
        if sum(node.type == "function_definition" for node in walk(root)) != 1:
            continue
        if identifier(top.child_by_field_name("declarator"), encoded):
            return language
    return None


def local_identifiers(
    source_text: str,
    language: str,
    function_name: str | None = None,
) -> tuple[str, ...]:
    function = parse_function(source_text, language, function_name)
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    names = [function.name, *function.parameters]
    for node in walk(tree.root_node):
        if node.type != "declaration":
            continue
        for child in node.named_children:
            declarator = (
                child.child_by_field_name("declarator")
                if child.type == "init_declarator"
                else child
            )
            name = identifier(declarator, source)
            if name and name not in names:
                names.append(name)
    return tuple(names)


@lru_cache(maxsize=16384)
def scope_safe_rename(source_text: str, language: str, *, replacement: str = "local_value") -> tuple[str | None, str]:
    """Alpha-rename one lexically bound variable in a conservative C/C++ subset.

    Unknown preprocessor/call semantics, nested functions, and nonlocal bindings
    are ineligible. No text substitution is performed in literals or members.
    Declaration positions and nested block/control scopes resolve shadowing.
    Ineligible inputs remain in the classification cohort, without an invariance
    example; returning the original source would overstate transformation coverage.
    """
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', replacement):
        raise ValueError('replacement must be an ordinary C/C++ identifier')
    source = source_text.encode("utf-8")
    root = parser_for(language).parse(source).root_node
    if root.has_error:
        return None, "parse_error"
    nodes = list(walk(root))
    functions = [n for n in nodes if n.type == "function_definition"]
    if len(functions) != 1:
        return None, "ambiguous_function"
    function = functions[0]
    if any(n.type.startswith("preproc_") or n.type in {
        "call_expression", "lambda_expression", "class_specifier",
        "struct_specifier", "union_specifier", "enum_specifier",
        "type_definition", "alias_declaration", "structured_binding_declarator",
        "for_range_loop", "gnu_asm_expression", "asm_statement",
        "attribute_specifier", "attribute_declaration", "ms_declspec_modifier",
        "try_statement", "catch_clause",
        "optional_parameter_declaration",
    } for n in nodes):
        return None, "unresolved_macro_or_binding_semantics"
    if sum(n.type == "function_declarator" for n in nodes) != 1:
        return None, "unsupported_declarator"
    body = function.child_by_field_name("body")
    if body is None:
        return None, "missing_body"
    if any(n.type in {"parameter_declaration", "declaration"}
           and any(x.type in {"storage_class_specifier", "attribute_specifier"} for x in walk(n))
           for n in nodes):
        return None, "unsupported_storage_or_attribute"
    try:
        bindings = function_bindings(function, source)
    except ValueError:
        return None, "unsupported_declarator_or_scope"
    if not bindings:
        return None, "no_local_binding"
    declarations = {(b["declaration"].start_byte, b["declaration"].end_byte): b for b in bindings}
    function_name = identifier(function.child_by_field_name("declarator"), source)
    for n in nodes:
        if n.type != "identifier":
            continue
        position = (n.start_byte, n.end_byte)
        if position in declarations:
            declarations[position]["occurrences"].append(n)
            continue
        if n.start_byte < body.start_byte and text(n, source) == function_name:
            continue
        # Qualified names may refer to external symbols, not a local variable.
        if n.parent.type in {"qualified_identifier", "scoped_identifier"}:
            return None, "nonlocal_identifier"
        binding = resolve_local_binding(n, bindings, source)
        if binding is None:
            return None, "nonlocal_identifier"
        binding["occurrences"].append(n)
    # A used binding is preferable to an unused declaration. Stable order is
    # independent of labels, model scores, and dataset split.
    used = [b for b in bindings if len(b["occurrences"]) > 1]
    if not used:
        return None, "no_used_local_binding"
    binding = used[0]
    names = {text(n, source) for n in nodes if n.type.endswith("identifier")}
    while replacement in names:
        replacement += "_"
    mutated = source
    for n in sorted(binding["occurrences"], key=lambda n: n.start_byte, reverse=True):
        mutated = mutated[:n.start_byte] + replacement.encode() + mutated[n.end_byte:]
    renamed = mutated.decode("utf-8")
    if parser_for(language).parse(mutated).root_node.has_error:
        raise ValueError("scope-safe renaming produced invalid syntax")
    return renamed, "renamed"


@lru_cache(maxsize=16384)
def semantic_preserving_view(source_text: str, language: str) -> tuple[str | None, str]:
    """A safe rename where verified, otherwise a token-preserving inline comment.

    The comment covers functions whose call/macro binding prevents verified
    renaming. It is outside all call arguments, does not add a line, and is not
    used for invalid/ambiguous function syntax or column-sensitive intrinsics.
    """
    renamed, reason = scope_safe_rename(source_text, language)
    if renamed is not None:
        return renamed, "rename"
    transformed, comment_reason = semantic_comment_view(source_text, language)
    return (transformed, 'comment') if transformed is not None else (None, reason if comment_reason == 'parse_error' else comment_reason)


def semantic_comment_view(source_text: str, language: str, comment: str = 'code') -> tuple[str | None, str]:
    """Insert one unrelated, single-line comment outside macro/call arguments."""
    if not re.fullmatch(r'[A-Za-z ]+', comment):
        raise ValueError('comment must contain only ordinary words and spaces')
    encoded = source_text.encode()
    root = parser_for(language).parse(encoded).root_node
    if root.has_error:
        return None, 'parse_error'
    top = [n for n in root.named_children if n.type != "comment"]
    if len(top) != 1:
        return None, 'ambiguous_function'
    function = top[0]
    if function.type == "template_declaration":
        functions = [n for n in walk(function) if n.type == "function_definition"]
        if len(functions) != 1:
            return None, 'ambiguous_function'
        function = functions[0]
    if function.type != "function_definition":
        return None, 'ambiguous_function'
    if any(name in source_text for name in ("__builtin_COLUMN", "__COLUMN__", "source_location")):
        return None, "column_sensitive_source"
    body = function.child_by_field_name("body")
    if body is None or encoded[body.start_byte:body.start_byte+1] != b"{":
        return None, "missing_body"
    at = body.start_byte + 1
    transformed = (encoded[:at] + f" /* {comment} */ ".encode() + encoded[at:]).decode()
    from .prepare_benchmark import normalized_source
    if (normalized_source(source_text) != normalized_source(transformed)
            or source_text.count("\n") != transformed.count("\n")):
        raise ValueError("comment transformation changed tokens or line numbers")
    return transformed, "comment"


AUGMENT_NAMES = ('value', 'item', 'count', 'data')
AUGMENT_COMMENTS = ('note', 'code', 'routine', 'section')
UNSEEN_NAMES = ('probe_item', 'probe_count', 'probe_data', 'probe_value')
UNSEEN_COMMENTS = ('orchard margin', 'linen marker', 'cedar passage', 'amber entry')


def apply_source_edits(source: str, edits) -> str:
    """Apply verified character edits; never interpret positions as UTF-8 bytes."""
    end = 0
    for a, b, value in edits:
        if not 0 <= end <= a <= b <= len(source) or not isinstance(value, str):
            raise ValueError('invalid or overlapping source edits')
        end = b
    for a, b, value in reversed(edits):
        source = source[:a] + value + source[b:]
    return source


def semantic_preserving_candidates(source: str, language: str, *, unseen: bool = False):
    """At most eight label-free proposals using the existing conservative rules.

    Compact character edits preserve provenance without copying whole functions
    into the experiment cache. Unseen banks are reserved for robustness only.
    """
    from collections import Counter
    candidates, skipped, seen = [], Counter(), set()
    original_tokens = source_token_spans(source)
    names = UNSEEN_NAMES if unseen else AUGMENT_NAMES
    comments = UNSEEN_COMMENTS if unseen else AUGMENT_COMMENTS
    for kind, values in (('rename', names), ('comment', comments)):
        for value in values:
            transformed, reason = (scope_safe_rename(source, language, replacement=value) if kind == 'rename'
                                   else semantic_comment_view(source, language, value))
            if transformed is None:
                skipped[reason] += 1
                continue
            if transformed == source or transformed in seen:
                skipped['unchanged_or_duplicate'] += 1
                continue
            if kind == 'rename':
                changed_tokens = source_token_spans(transformed)
                if len(changed_tokens) != len(original_tokens):
                    raise ValueError('verified rename changed the lexical token count')
                edits = [(a, b, new) for (old, a, b), (new, _, _) in zip(original_tokens, changed_tokens) if old != new]
            else:
                at = next(i for i, (a, b) in enumerate(zip(source, transformed)) if a != b)
                edits = [(at, at, transformed[at:at + len(transformed)-len(source)])]
            if apply_source_edits(source, edits) != transformed:
                raise ValueError('candidate edits do not reconstruct the verified transformation')
            candidates.append(dict(kind=kind, value=value, edits=edits))
            seen.add(transformed)
    return candidates, dict(skipped)


# Tokens preserve literals and punctuation; comment/whitespace removal cannot turn
# a nested method or a call into the complete source definition.
_SOURCE_TOKEN = re.compile(r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[^\W\d]\w*|\d+(?:\.\d+)?|>>=|<<=|->\*|\.\.\.|::|->|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]', re.S)


def source_tokens(source: str) -> tuple[str, ...]:
    return tuple(value for value, _, _ in source_token_spans(source))


@lru_cache(maxsize=16384)
def source_token_spans(source: str) -> tuple[tuple[str, int, int], ...]:
    """Lexical tokens with Python character offsets, including literals/operators."""
    return tuple((m.group(), m.start(), m.end()) for m in _SOURCE_TOKEN.finditer(source)
                 if not m.group().startswith(('//', '/*')))


@lru_cache(maxsize=16384)
def code_line_spans(source: str) -> dict[int, tuple[tuple[int, int], ...]]:
    """Original physical lines and their code characters; never rewrite source."""
    starts = [0] + [m.end() for m in re.finditer('\n', source)]
    spans, tokens = {}, {}
    for value, left, right in source_token_spans(source):
        first = bisect_right(starts, left) - 1
        last = bisect_right(starts, max(left, right - 1)) - 1
        for line in range(first, last + 1):
            a, b = max(left, starts[line]), min(right, starts[line + 1] if line + 1 < len(starts) else len(source))
            # Offset tokenizers can trim whitespace. Non-whitespace lexical
            # characters establish coverage, including inside multiline literals.
            for part in re.finditer(r'\S+', source[a:b]):
                spans.setdefault(line + 1, []).append((a + part.start(), a + part.end()))
                tokens.setdefault(line + 1, []).append(value)
    return {line: tuple(parts) for line, parts in spans.items()
            if any(value not in {'{', '}'} for value in tokens[line])}


def byte_character_map(source: str) -> dict[int, int]:
    result, byte = {0: 0}, 0
    for char, value in enumerate(source, 1):
        byte += len(value.encode('utf-8'))
        result[byte] = char
    return result


_BINDING_SCOPES = {'compound_statement', 'for_statement', 'if_statement',
                   'switch_statement', 'while_statement', 'do_statement'}


def declarator_identifier(node):
    while node is not None and node.type in {
        'init_declarator', 'pointer_declarator', 'array_declarator',
        'reference_declarator', 'parenthesized_declarator',
    }:
        node = node.child_by_field_name('declarator')
    return node if node is not None and node.type == 'identifier' else None


def function_bindings(function, source: bytes):
    """Position/scope bindings, also usable in ordinary call arguments."""
    body = function.child_by_field_name('body')
    if body is None:
        raise ValueError('missing function body')
    bindings, parameter_index = [], 0
    for node in walk(function):
        if node.type not in {'parameter_declaration', 'optional_parameter_declaration', 'declaration'}:
            continue
        parameter = node.type != 'declaration'
        index = parameter_index if parameter else None
        if parameter:
            parameter_index += 1
        if not parameter and not body.start_byte <= node.start_byte < body.end_byte:
            continue
        for decl in node.children_by_field_name('declarator'):
            name = declarator_identifier(decl)
            if name is None:
                raise ValueError('unsupported local declarator')
            scope = function if parameter else node.parent
            while scope is not None and not parameter and scope.type not in _BINDING_SCOPES:
                scope = scope.parent
            if scope is None:
                raise ValueError('unsupported local scope')
            bindings.append(dict(name=text(name, source), declaration=name, statement=node,
                                 scope=scope, parameter_index=index, occurrences=[]))
    return bindings


def resolve_local_binding(node, bindings, source: bytes):
    if node.type != 'identifier' or node.parent.type in {'qualified_identifier', 'scoped_identifier'}:
        return None
    matches = [b for b in bindings if b['name'] == text(node, source)
               and b['declaration'].end_byte <= node.start_byte
               and b['scope'].start_byte <= node.start_byte < b['scope'].end_byte]
    return min(matches, key=lambda b: (b['scope'].end_byte-b['scope'].start_byte,
                                      -b['declaration'].start_byte)) if matches else None


def _statement_extent(node):
    if node.type == 'compound_statement':
        return None
    if node.type in {'if_statement', 'for_statement', 'while_statement', 'do_statement',
                     'switch_statement', 'for_range_loop'}:
        bodies = [node.child_by_field_name(field) for field in ('body', 'consequence')]
        ends = [body.start_byte for body in bodies if body is not None]
        return (node.start_byte, min(ends)) if ends else None
    if node.type == 'declaration' or node.type in {
        'expression_statement', 'return_statement', 'break_statement', 'continue_statement',
        'goto_statement', 'null_statement', 'throw_statement',
    }:
        return node.start_byte, node.end_byte
    return None


def repair_candidate_lines(before: str, after: str, language: str) -> dict:
    """Full-source token edits and bounded same-block repair-guided candidates."""
    old, new = source_token_spans(before), source_token_spans(after)
    opcodes = SequenceMatcher(None, [t[0] for t in old], [t[0] for t in new], autojunk=False).get_opcodes()
    equal = [dict(before_tokens=[i, j], after_tokens=[k, l],
                  before_chars=[old[i][1], old[j-1][2]], after_chars=[new[k][1], new[l-1][2]])
             for tag, i, j, k, l in opcodes if tag == 'equal' and j > i]
    effective = code_line_spans(before)
    lines, events, skipped = set(), [], []
    old_bytes, new_bytes = before.encode(), after.encode()
    old_map, new_map = byte_character_map(before), byte_character_map(after)
    old_byte = {char: byte for byte, char in old_map.items()}
    old_root = parser_for(language).parse(old_bytes).root_node
    new_root = parser_for(language).parse(new_bytes).root_node
    old_functions = [n for n in walk(old_root) if n.type == 'function_definition']
    new_functions = [n for n in walk(new_root) if n.type == 'function_definition']
    reliable = (not old_root.has_error and not new_root.has_error
                and len(old_functions) == len(new_functions) == 1
                and not any(n.type.startswith('preproc_') or n.type in {
                    'lambda_expression', 'class_specifier', 'struct_specifier',
                    'for_range_loop', 'gnu_asm_expression', 'asm_statement',
                } for root in (old_root, new_root) for n in walk(root)))

    def add_span(left, right, rule, **info):
        chosen, at = set(), bisect_right(effective_ends, left)
        while at < len(effective_parts) and effective_parts[at][0] < right:
            a, b, line = effective_parts[at]
            if b > left:
                chosen.add(line)
            at += 1
        chosen = sorted(chosen)
        lines.update(chosen)
        events.append(dict(rule=rule, lines=chosen, **info))

    def add_statement(node, rule, **info):
        extent = _statement_extent(node)
        if extent is None:
            skipped.append('no_noncompound_statement_extent')
        else:
            add_span(old_map[extent[0]], old_map[extent[1]], rule, **info)

    old_nodes = list(walk(old_root)) if reliable else []
    blocks = [n for n in old_nodes if n.type == 'compound_statement']
    effective_parts = sorted((a, b, line) for line, spans in effective.items() for a, b in spans)
    effective_ends = [b for _, b, _ in effective_parts]

    def anchor_at(char):
        byte = old_byte[char]
        candidates = [n for n in blocks if n.start_byte <= byte < n.end_byte]
        if not candidates:
            return None, None, []
        block = min(candidates, key=lambda n: n.end_byte-n.start_byte)
        statements = [n for n in block.named_children if n.type != 'comment']
        following = [n for n in statements if n.end_byte > byte]
        anchor = following[0] if following else (statements[-1] if statements else None)
        return block, anchor, statements

    if reliable:
        try:
            old_bindings = function_bindings(old_functions[0], old_bytes)
            new_bindings = function_bindings(new_functions[0], new_bytes)
        except ValueError:
            old_bindings, new_bindings = [], []
            skipped.append('unresolved_local_bindings')
    else:
        old_bindings, new_bindings = [], []
        skipped.append('unreliable_parse_direct_edits_only')

    for tag, i, j, k, l in opcodes:
        if tag == 'equal':
            continue
        for _, left, right in old[i:j]:
            add_span(left, right, 'edited_old_token', hunk=[tag, i, j, k, l])
            if reliable:
                a, b = old_byte[left], old_byte[right]
                statement = old_root.descendant_for_byte_range(a, b)
                while statement is not None and _statement_extent(statement) is None:
                    statement = statement.parent
                if statement is not None:
                    add_statement(statement, 'edited_statement')
        if not reliable:
            continue
        char = old[i][1] if i < len(old) else (old[-1][2] if old else 0)
        block, anchor, statements = anchor_at(char)
        if tag == 'insert':
            if anchor is not None:
                add_statement(anchor, 'insert_same_block_anchor')
            else:
                skipped.append('insert_without_same_block_statement')
        changed_start = new[k][1] if k < len(new) else len(after)
        changed_end = new[l-1][2] if l > k else changed_start
        for guard in (n for n in walk(new_root) if n.type == 'if_statement'):
            condition = guard.child_by_field_name('condition')
            if condition is None or not (new_map[condition.start_byte] < changed_end
                                        and new_map[condition.end_byte] > changed_start):
                continue
            mapped = []
            for node in walk(condition):
                binding = resolve_local_binding(node, new_bindings, new_bytes)
                if binding is None:
                    continue
                if binding['parameter_index'] is not None:
                    matches = [b for b in old_bindings if b['parameter_index'] == binding['parameter_index']]
                else:
                    declaration = binding['statement']
                    a, b = new_map[declaration.start_byte], new_map[declaration.end_byte]
                    declaration_tokens = [token for token in new if a <= token[1] and token[2] <= b]
                    if declaration_tokens and declaration_tokens[-1][0] == ';':
                        # An inserted statement's identical ';' can be selected
                        # by SequenceMatcher. Declaration identity is established
                        # by all type/declarator/initializer tokens, not punctuation.
                        b = declaration_tokens[-2][2] if len(declaration_tokens) > 1 else a
                    matches = []
                    for section in equal:
                        if section['after_chars'][0] <= a and b <= section['after_chars'][1]:
                            ni, nj = section['after_tokens']
                            oi, _ = section['before_tokens']
                            position = (new_map[binding['declaration'].start_byte], new_map[binding['declaration'].end_byte])
                            token_index = next((q for q in range(ni, nj) if new[q][1:] == position), None)
                            if token_index is not None:
                                old_position = old[oi+token_index-ni][1:]
                                matches.extend(x for x in old_bindings if x['parameter_index'] is None
                                    and (old_map[x['declaration'].start_byte], old_map[x['declaration'].end_byte]) == old_position)
                if len(matches) == 1 and matches[0] not in mapped:
                    mapped.append(matches[0])
            if not mapped or anchor is None:
                skipped.append('guard_without_verified_binding_or_anchor')
                continue
            at = statements.index(anchor)
            for statement in statements[at:at+3]:
                extent = _statement_extent(statement)
                if extent is None:
                    continue
                used = [resolve_local_binding(n, old_bindings, old_bytes) for n in walk(statement)
                        if extent[0] <= n.start_byte and n.end_byte <= extent[1]]
                if any(b in used for b in mapped):
                    add_statement(statement, 'guard_first_bound_use_within_three',
                                  bindings=[old_map[b['declaration'].start_byte] for b in mapped])
                    break
    return dict(lines=sorted(lines), events=events, skip_reasons=sorted(set(skipped)),
                equal_token_sections=equal, changed=any(tag != 'equal' for tag, *_ in opcodes))


def resolve_language(source: str, language: str, file_name: str = '') -> str:
    suffix = Path(file_name).suffix
    if suffix == '.c':
        return 'c'
    if suffix == '.C' or suffix.lower() in {'.cc', '.cpp', '.cxx', '.c++', '.hpp', '.hh', '.hxx', '.h++'}:
        return 'cpp'
    if language in {'c', 'cpp'}:
        return language
    encoded = source.encode()
    return min(('c', 'cpp'), key=lambda lang: (
        sum(n.is_error or n.is_missing for n in walk(parser_for(lang).parse(encoded).root_node)), lang))


def target_hint(source: str, language: str, supplied_name: str | None = None) -> ParsedFunction:
    """Advisory syntax only. Joern must independently prove the complete span."""
    try:
        parsed = parse_function(source, language, supplied_name)
    except ValueError:
        try:
            parsed = parse_function(source, language)
        except ValueError:
            return ParsedFunction(name='', parameters=())
    if parsed.name in {'if', 'for', 'while', 'switch', 'catch', 'sizeof'}:
        return ParsedFunction(name='', parameters=())
    return parsed
