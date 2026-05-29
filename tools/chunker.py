#!/usr/bin/env python3
"""
CodeWriter RAG Chunker v2 — Lexer/Parser-based syntax-aware code chunking.

Uses per-language lexers to accurately track string/comment state, then
builds a scope tree via brace-depth or indentation parsing.  Chunks are
emitted at exact syntactic boundaries — a function is one chunk, a struct
is one chunk.  Only when a block exceeds the context budget is it split,
and then at internal semantic boundaries with interrelation metadata.

New in v2
─────────
• Character-level lexer per language family (no regex line-matching)
• Scope tree with parent/child relationships
• Cross-reference extraction  (calls / type usage)
• File-level summary chunks   (table of contents for broad queries)
• Continuation-hint headers   (replaces fixed-line overlap)
• Semantic sub-splitting of oversized blocks
• Extended JSON output with chunk metadata

Input  (stdin):  [{"name": "main.rs", "content": "...", "language": "rust"}, ...]
Output (stdout): [{"source": "...", "text": "...", "kind": "...", ...}, ...]

Usage:
    echo '[...]' | python3 chunker.py [OPTIONS]

Options:
    --max-chunk-lines N   Hard limit before splitting a block (default: 150)
    --no-metadata         Skip metadata header enrichment
    --no-summaries        Skip file-level summary chunks
    --no-refs             Skip cross-reference extraction
    --json-pretty         Pretty-print output JSON
"""

from __future__ import annotations

import sys
import json
import argparse
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum, auto


# ════════════════════════════════════════════════════════════
# §1  DATA TYPES
# ════════════════════════════════════════════════════════════

@dataclass
class LineInfo:
    """Per-line metadata produced by the lexer."""
    index: int              # 0-based line number
    raw: str                # original line text
    effective: str          # code only — strings → _S_, comments stripped
    brace_delta: int        # net { minus } on this line (brace langs)
    indent: int             # leading whitespace count (spaces; tabs=4)
    is_blank: bool
    is_comment_only: bool
    is_doc_comment: bool    # /// or /** or # docstring-ish
    is_decorator: bool      # @something
    is_preprocessor: bool   # #include, #define, etc.


@dataclass
class ScopeNode:
    """A node in the parsed scope tree."""
    kind: str               # 'file', 'function', 'struct', 'impl', 'class',
                            # 'method', 'enum', 'trait', 'interface', 'module',
                            # 'namespace', 'test', 'preamble', 'gap', 'tail'
    name: str
    start: int              # first line (0-indexed, inclusive)
    end: int                # last line (0-indexed, exclusive)
    indent: int
    children: List[ScopeNode] = field(default_factory=list)
    parent: Optional[ScopeNode] = field(default=None, repr=False)
    doc_start: int = -1     # where doc-comments / decorators begin
    signature: str = ""     # one-line signature for context injection
    refs: Set[str] = field(default_factory=set)

    @property
    def effective_start(self) -> int:
        return self.doc_start if self.doc_start >= 0 else self.start

    def line_count(self) -> int:
        return self.end - self.effective_start

    def __repr__(self):
        return (f"Scope({self.kind} '{self.name}' "
                f"L{self.effective_start+1}-{self.end})")


@dataclass
class Chunk:
    """A single output chunk ready for embedding."""
    source: str             # "file:type:name:startL-endL" or "file:summary"
    text: str               # enriched text with metadata header
    kind: str               # 'block', 'block_part', 'file_summary', 'gap'
    name: str = ""
    block_type: str = ""
    file: str = ""
    refs: List[str] = field(default_factory=list)
    part: Optional[Dict] = None   # {"index": 1, "total": 3}
    parent_scope: str = ""        # "impl VecStore" etc.


# ════════════════════════════════════════════════════════════
# §2  LANGUAGE CONFIGURATION
# ════════════════════════════════════════════════════════════

LANG_BY_EXT = {
    "rs": "rust", "py": "python", "pyw": "python",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "go": "go",
    "c": "c", "h": "c",
    "cpp": "c++", "cc": "c++", "cxx": "c++", "hpp": "c++", "hh": "c++",
    "java": "java",
    "kt": "kotlin", "kts": "kotlin",
    "lua": "lua", "zig": "zig",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "sql": "sql",
    "html": "html", "htm": "html",
    "css": "css", "scss": "css", "sass": "css", "less": "css",
    "xml": "xml", "svg": "xml",
    "json": "json", "jsonc": "json",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml",
    "md": "markdown", "markdown": "markdown",
    "txt": "text",
}

LANG_DISPLAY = {
    "rust": "Rust", "python": "Python", "javascript": "JavaScript",
    "typescript": "TypeScript", "go": "Go", "c": "C", "c++": "C++",
    "java": "Java", "kotlin": "Kotlin",
    "bash": "Bash", "sql": "SQL", "html": "HTML", "css": "CSS",
    "json": "JSON", "yaml": "YAML", "toml": "TOML",
    "markdown": "Markdown", "text": "Text", "zig": "Zig", "lua": "Lua",
}

BRACE_LANGS = {
    "rust", "javascript", "typescript", "go", "c", "c++",
    "java", "kotlin", "zig", "css",
}
INDENT_LANGS = {"python"}

# Block-starting keyword patterns per language.
# Each is (regex_for_effective_line, block_type)
# "effective_line" = strings and comments already stripped.
_KW: Dict[str, List[Tuple[re.Pattern, str]]] = {}


def _build_keywords():
    # ── Rust ────────────────────────────────────────
    _KW["rust"] = [
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?(?:async\s+)?fn\s+(\w+)'), "function"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?struct\s+(\w+)'), "struct"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?enum\s+(\w+)'), "enum"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?trait\s+(\w+)'), "trait"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?type\s+(\w+)'), "type_alias"),
        (re.compile(r'^impl(?:<[^>]*>)?\s+\w+\s+for\s+(\w+)'), "impl"),
        (re.compile(r'^impl(?:<[^>]*>)?\s+(\w+)'), "impl"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?mod\s+(\w+)'), "module"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?const\s+(\w+)'), "constant"),
        (re.compile(r'^(?:pub(?:\(crate\))?\s+)?static\s+(\w+)'), "static"),
        (re.compile(r'^macro_rules!\s+(\w+)'), "macro"),
        (re.compile(r'^#\[cfg\(test\)\]'), "test_module"),
    ]
    # ── Python ──────────────────────────────────────
    _KW["python"] = [
        (re.compile(r'^class\s+(\w+)'), "class"),
        (re.compile(r'^(?:async\s+)?def\s+(\w+)'), "function"),
        (re.compile(r'^\s{2,}(?:async\s+)?def\s+(\w+)'), "method"),
        (re.compile(r'^\s{2,}class\s+(\w+)'), "inner_class"),
        (re.compile(r'^@(\w+)'), "decorator"),
    ]
    # ── JavaScript / TypeScript ─────────────────────
    for lang in ("javascript", "typescript"):
        _KW[lang] = [
            (re.compile(r'^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)'), "function"),
            (re.compile(r'^(?:export\s+(?:default\s+)?)?class\s+(\w+)'), "class"),
            (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\('), "arrow_function"),
            (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function'), "function_expr"),
            (re.compile(r'^(?:export\s+)?interface\s+(\w+)'), "interface"),
            (re.compile(r'^(?:export\s+)?type\s+(\w+)'), "type_alias"),
            (re.compile(r'^(?:export\s+)?enum\s+(\w+)'), "enum"),
            (re.compile(r'^describe\s*\('), "test_suite"),
            (re.compile(r'^(?:it|test)\s*\('), "test_case"),
        ]
    # ── Go ──────────────────────────────────────────
    _KW["go"] = [
        (re.compile(r'^func\s+(?:\([^)]*\)\s+)?(\w+)'), "function"),
        (re.compile(r'^type\s+(\w+)\s+struct\b'), "struct"),
        (re.compile(r'^type\s+(\w+)\s+interface\b'), "interface"),
        (re.compile(r'^type\s+(\w+)\s+'), "type"),
        (re.compile(r'^var\s+\('), "var_block"),
        (re.compile(r'^const\s+\('), "const_block"),
    ]
    # ── C / C++ ─────────────────────────────────────
    for lang in ("c", "c++"):
        _KW[lang] = [
            (re.compile(r'^(?:typedef\s+)?struct\s+(\w+)'), "struct"),
            (re.compile(r'^(?:typedef\s+)?enum\s+(\w+)'), "enum"),
            (re.compile(r'^(?:typedef\s+)?union\s+(\w+)'), "union"),
            (re.compile(r'^class\s+(\w+)'), "class"),
            (re.compile(r'^namespace\s+(\w+)'), "namespace"),
            (re.compile(r'^template\s*<'), "template"),
            # Function: return_type name( — must not end with ;
            (re.compile(r'^(?:static\s+|inline\s+|extern\s+)*(?:[\w:*&<>,\s]+)\s+(\w+)\s*\([^;]*$'), "function"),
        ]
    # ── Java ────────────────────────────────────────
    _KW["java"] = [
        (re.compile(r'(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*class\s+(\w+)'), "class"),
        (re.compile(r'(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*interface\s+(\w+)'), "interface"),
        (re.compile(r'(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*enum\s+(\w+)'), "enum"),
        (re.compile(r'(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*[\w<>\[\],]+\s+(\w+)\s*\('), "method"),
    ]
    # ── Kotlin ──────────────────────────────────────
    _KW["kotlin"] = [
        (re.compile(r'(?:(?:open|abstract|sealed|data|inner|private|public|internal|protected)\s+)*class\s+(\w+)'), "class"),
        (re.compile(r'(?:(?:open|abstract|sealed|data|inner|private|public|internal|protected)\s+)*interface\s+(\w+)'), "interface"),
        (re.compile(r'(?:(?:open|abstract|sealed|data|inner|private|public|internal|protected)\s+)*object\s+(\w+)'), "object"),
        (re.compile(r'(?:(?:suspend|inline|private|public|internal|protected)\s+)*fun\s+(\w+)'), "function"),
    ]
    # ── Lua ─────────────────────────────────────────
    _KW["lua"] = [
        (re.compile(r'^(?:local\s+)?function\s+(\w[\w.]*)'), "function"),
    ]
    # ── Zig ─────────────────────────────────────────
    _KW["zig"] = [
        (re.compile(r'^(?:pub\s+)?fn\s+(\w+)'), "function"),
        (re.compile(r'^(?:pub\s+)?const\s+(\w+)\s*=\s*struct'), "struct"),
        (re.compile(r'^(?:pub\s+)?const\s+(\w+)\s*=\s*enum'), "enum"),
        (re.compile(r'^(?:pub\s+)?const\s+(\w+)\s*=\s*union'), "union"),
    ]
    # ── Bash ────────────────────────────────────────
    _KW["bash"] = [
        (re.compile(r'^(\w+)\s*\(\)\s*\{?'), "function"),
        (re.compile(r'^function\s+(\w+)'), "function"),
    ]

_build_keywords()


def detect_language(filename: str, given: str = "") -> str:
    if given and given.lower() not in ("", "auto", "unknown"):
        return given.lower()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return LANG_BY_EXT.get(ext, "text")


# ════════════════════════════════════════════════════════════
# §3  LEXER — Character-level scanner
# ════════════════════════════════════════════════════════════

@dataclass
class ScanState:
    """Cross-line scanner state."""
    in_block_comment: bool = False
    in_multiline_string: bool = False
    ml_string_delim: str = ""       # closing delimiter for the multiline string
    # Rust raw strings: r#"..."# — we track the number of #
    rust_raw_hashes: int = 0


def _measure_indent(line: str) -> int:
    n = 0
    for ch in line:
        if ch == ' ':
            n += 1
        elif ch == '\t':
            n += 4
        else:
            break
    return n


def scan_source(source: str, lang: str) -> List[LineInfo]:
    """
    Lex the entire source file.  For each line, produce a LineInfo with
    the effective code content (strings→_S_, comments stripped),
    brace delta, indent, and classification flags.
    """
    lines = source.split("\n")
    state = ScanState()
    result: List[LineInfo] = []

    is_brace = lang in BRACE_LANGS
    # Determine comment syntax
    line_comment_tokens = _line_comment_tokens(lang)
    has_block_comments = lang not in ("python", "bash", "yaml",
                                      "toml", "text", "markdown")
    block_open = "/*"
    block_close = "*/"
    if lang == "html" or lang == "xml":
        block_open = "<!--"
        block_close = "-->"
    elif lang == "lua":
        # Lua block comments handled specially: --[[ ... ]]
        pass

    for idx, raw in enumerate(lines):
        indent = _measure_indent(raw)
        stripped = raw.strip()

        # Fast-path: entirely blank
        if not stripped:
            result.append(LineInfo(
                index=idx, raw=raw, effective="",
                brace_delta=0, indent=indent,
                is_blank=True, is_comment_only=False,
                is_doc_comment=False, is_decorator=False,
                is_preprocessor=False,
            ))
            continue

        # Character-level scan
        effective_chars: List[str] = []
        brace_delta = 0
        i = 0
        n = len(raw)
        in_line_string = False
        string_delim = ""
        saw_code = False

        while i < n:
            ch = raw[i]

            # ── Inside block comment ─────────────────────
            if state.in_block_comment:
                if lang == "lua":
                    if raw[i:i+2] == "]]":
                        state.in_block_comment = False
                        i += 2
                        continue
                elif lang in ("html", "xml"):
                    if raw[i:i+3] == "-->":
                        state.in_block_comment = False
                        i += 3
                        continue
                else:
                    if raw[i:i+2] == "*/":
                        state.in_block_comment = False
                        i += 2
                        continue
                i += 1
                continue

            # ── Inside multiline string ──────────────────
            if state.in_multiline_string:
                if lang == "rust" and state.rust_raw_hashes > 0:
                    # r#"..."#  — look for "# with the right number of hashes
                    close = '"' + '#' * state.rust_raw_hashes
                    if raw[i:i+len(close)] == close:
                        state.in_multiline_string = False
                        i += len(close)
                        effective_chars.append("_S_")
                        continue
                elif state.ml_string_delim:
                    pos = raw.find(state.ml_string_delim, i)
                    if pos >= 0:
                        state.in_multiline_string = False
                        i = pos + len(state.ml_string_delim)
                        effective_chars.append("_S_")
                        continue
                i += 1
                continue

            # ── Inside single-line string ────────────────
            if in_line_string:
                if ch == '\\':
                    i += 2  # skip escaped char
                    continue
                if ch == string_delim:
                    in_line_string = False
                    effective_chars.append("_S_")
                    i += 1
                    continue
                i += 1
                continue

            # ── Check for line comments ──────────────────
            hit_line_comment = False
            for tok in line_comment_tokens:
                if raw[i:i+len(tok)] == tok:
                    hit_line_comment = True
                    break
            if hit_line_comment:
                break  # rest of line is comment

            # ── Check for block comment open ─────────────
            if has_block_comments:
                if lang == "lua" and raw[i:i+4] == "--[[":
                    state.in_block_comment = True
                    i += 4
                    continue
                elif lang in ("html", "xml") and raw[i:i+4] == "<!--":
                    state.in_block_comment = True
                    i += 4
                    continue
                elif lang not in ("lua", "html", "xml") and raw[i:i+2] == "/*":
                    state.in_block_comment = True
                    i += 2
                    continue

            # ── Check for string literals ────────────────
            string_started = False

            # Python triple-quotes
            if lang == "python" and raw[i:i+3] in ('"""', "'''"):
                delim3 = raw[i:i+3]
                # Check if closes on the same line
                close_pos = raw.find(delim3, i + 3)
                if close_pos >= 0:
                    effective_chars.append("_S_")
                    i = close_pos + 3
                    string_started = True
                else:
                    state.in_multiline_string = True
                    state.ml_string_delim = delim3
                    i += 3
                    string_started = True

            # Rust raw strings: r#"..."#, r##"..."##, etc.
            if not string_started and lang == "rust" and ch == 'r':
                hashes = 0
                j = i + 1
                while j < n and raw[j] == '#':
                    hashes += 1
                    j += 1
                if j < n and raw[j] == '"' and hashes > 0:
                    close = '"' + '#' * hashes
                    close_pos = raw.find(close, j + 1)
                    if close_pos >= 0:
                        effective_chars.append("_S_")
                        i = close_pos + len(close)
                        string_started = True
                    else:
                        state.in_multiline_string = True
                        state.rust_raw_hashes = hashes
                        i = j + 1
                        string_started = True

            # Go/JS backtick strings (multiline)
            if not string_started and lang in ("go", "javascript", "typescript") and ch == '`':
                close_pos = raw.find('`', i + 1)
                if close_pos >= 0:
                    effective_chars.append("_S_")
                    i = close_pos + 1
                    string_started = True
                else:
                    state.in_multiline_string = True
                    state.ml_string_delim = "`"
                    i += 1
                    string_started = True

            # Regular single/double quotes
            if not string_started and ch in ('"', "'"):
                # Rust: single-quote might be a lifetime 'a, not a char literal
                if ch == "'" and lang == "rust":
                    # If next char is alphanumeric and char after isn't ',
                    # it's a lifetime, not a char
                    if i + 1 < n and raw[i+1].isalpha():
                        if i + 2 >= n or raw[i+2] != "'":
                            effective_chars.append(ch)
                            saw_code = True
                            i += 1
                            continue
                in_line_string = True
                string_delim = ch
                i += 1
                continue

            if string_started:
                continue

            # ── Regular code character ───────────────────
            effective_chars.append(ch)
            if not ch.isspace():
                saw_code = True

            if is_brace:
                if ch == '{':
                    brace_delta += 1
                elif ch == '}':
                    brace_delta -= 1
            i += 1

        effective = "".join(effective_chars).strip()
        is_comment = not saw_code and not state.in_block_comment and bool(stripped)
        is_doc = False
        if is_comment:
            is_doc = (stripped.startswith("///") or stripped.startswith("//!")
                      or stripped.startswith("/**") or stripped.startswith("* ")
                      or stripped.startswith('"""') or stripped.startswith("'''")
                      or stripped.startswith("##"))
        is_deco = (not state.in_block_comment and not state.in_multiline_string
                   and stripped.startswith("@") and lang in ("python", "java",
                   "kotlin", "typescript", "javascript"))
        is_preproc = stripped.startswith("#include") or stripped.startswith("#define")

        result.append(LineInfo(
            index=idx, raw=raw, effective=effective,
            brace_delta=brace_delta, indent=indent,
            is_blank=False, is_comment_only=is_comment,
            is_doc_comment=is_doc, is_decorator=is_deco,
            is_preprocessor=is_preproc,
        ))

    return result


def _line_comment_tokens(lang: str) -> List[str]:
    if lang in ("python", "bash", "yaml", "toml"):
        return ["#"]
    if lang in ("sql",):
        return ["--"]
    if lang in ("lua",):
        return ["--"]
    if lang in ("html", "xml", "css"):
        return []  # only block comments
    return ["//"]


# ════════════════════════════════════════════════════════════
# §4  SCOPE PARSER — Build tree from lexed lines
# ════════════════════════════════════════════════════════════

def parse_scope_tree(line_infos: List[LineInfo], lines: List[str],
                     lang: str) -> ScopeNode:
    """Build a scope tree from the scanned line data."""
    root = ScopeNode(
        kind="file", name="<file>",
        start=0, end=len(lines), indent=0,
    )

    if lang in BRACE_LANGS:
        _parse_brace_scopes(root, line_infos, lines, lang)
    elif lang in INDENT_LANGS:
        _parse_indent_scopes(root, line_infos, lines, lang)
    # else: no scope detection for markup / data languages

    return root


def _parse_brace_scopes(root: ScopeNode, infos: List[LineInfo],
                         lines: List[str], lang: str):
    """
    Parse brace-scoped language into a scope tree.

    Walks forward line by line tracking global brace depth.  Each scope
    entry on the stack carries `has_opened` — True once we've seen the
    opening `{`.  A scope can only *close* (depth returns to its
    pre-open level) after `has_opened` is True.  This prevents multi-line
    signatures like `fn foo(\n  ...\n) {` from closing prematurely when
    the parameter lines still sit at the original depth.
    """
    patterns = _KW.get(lang, [])
    if not patterns:
        return

    n = len(infos)
    depth = 0
    # Stack entries: (ScopeNode, open_depth, has_opened)
    #   open_depth  = brace depth BEFORE this scope's `{`
    #   has_opened  = True once we've actually seen a `{`
    scope_stack: List[Tuple[ScopeNode, int, bool]] = [(root, -1, True)]

    i = 0
    while i < n:
        info = infos[i]

        old_depth = depth
        depth += info.brace_delta

        # When a positive brace delta appears, mark the top-of-stack
        # scope as "opened" (the `{` finally arrived).
        if info.brace_delta > 0 and len(scope_stack) > 1:
            sn, od, ho = scope_stack[-1]
            if not ho:
                scope_stack[-1] = (sn, od, True)

        # Pop any scopes whose closing `}` just appeared.
        # Only eligible if has_opened is True.
        while len(scope_stack) > 1:
            top_node, open_depth, has_opened = scope_stack[-1]
            if has_opened and depth <= open_depth:
                top_node.end = i + 1
                scope_stack.pop()
            else:
                break

        # Skip non-code lines for keyword matching
        if info.is_blank or info.is_comment_only or info.is_doc_comment:
            i += 1
            continue

        # Try to match a block-starting keyword
        matched_type = None
        matched_name = ""
        for pat, btype in patterns:
            m = pat.search(info.effective)
            if m:
                matched_type = btype
                matched_name = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                break

        if matched_type is None:
            i += 1
            continue

        # ── Doc comments / decorators above the keyword ──────────
        doc_start = i
        j = i - 1
        while j >= 0:
            prev = infos[j]
            if prev.is_doc_comment or prev.is_decorator:
                doc_start = j
                j -= 1
            elif prev.is_blank:
                if j > 0 and (infos[j-1].is_doc_comment or infos[j-1].is_decorator):
                    doc_start = j
                    j -= 1
                else:
                    break
            else:
                break

        # ── Extract signature (up to the opening brace) ──────────
        sig_line = lines[i].strip()
        sig = sig_line
        if '{' not in sig_line:
            for k in range(i + 1, min(i + 5, n)):
                sig += " " + lines[k].strip()
                if '{' in lines[k]:
                    break
        brace_pos = sig.find('{')
        if brace_pos > 0:
            sig = sig[:brace_pos].strip()

        # Classify the keyword line
        has_brace = info.brace_delta > 0 or ('{' in info.effective)
        is_single_line = (';' in info.effective and info.brace_delta == 0
                          and '{' not in info.effective)

        node = ScopeNode(
            kind=matched_type, name=matched_name,
            start=i, end=n,
            indent=info.indent,
            doc_start=doc_start if doc_start < i else -1,
            signature=sig,
        )

        parent = scope_stack[-1][0]
        node.parent = parent
        parent.children.append(node)

        if is_single_line:
            node.end = i + 1
        elif has_brace:
            # Brace on the keyword line — already opened
            scope_stack.append((node, old_depth, True))
            if depth <= old_depth:
                # One-liner: `fn f() { x }`
                node.end = i + 1
                scope_stack.pop()
        else:
            # Multi-line signature — `{` hasn't appeared yet
            scope_stack.append((node, old_depth, False))

        i += 1

    # Close any remaining open scopes
    while len(scope_stack) > 1:
        top_node, _, _ = scope_stack.pop()
        top_node.end = n


def _parse_indent_scopes(root: ScopeNode, infos: List[LineInfo],
                          lines: List[str], lang: str):
    """Parse indent-scoped language (Python, Ruby) into a scope tree."""
    patterns = _KW.get(lang, [])
    if not patterns:
        return

    n = len(infos)
    i = 0
    scope_stack: List[ScopeNode] = [root]

    while i < n:
        info = infos[i]
        if info.is_blank:
            i += 1
            continue

        # Try to match a block keyword
        matched_type = None
        matched_name = ""

        # For decorators: mark the start, then skip to the def/class
        if info.is_decorator:
            doc_start = i
            # Scan forward to find the actual def/class
            j = i + 1
            while j < n:
                if infos[j].is_blank or infos[j].is_decorator or infos[j].is_comment_only:
                    j += 1
                    continue
                # This should be the def/class line
                for pat, btype in patterns:
                    m = pat.search(infos[j].effective)
                    if m:
                        matched_type = btype
                        matched_name = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                        break
                if matched_type:
                    i = j  # jump to the def/class line
                    info = infos[i]
                break
            if not matched_type:
                i += 1
                continue
            # doc_start stays at the decorator line
        else:
            doc_start = i
            # Check for doc comments above
            j = i - 1
            while j >= 0 and (infos[j].is_doc_comment or infos[j].is_comment_only
                              or infos[j].is_blank):
                if infos[j].is_blank:
                    if j > 0 and infos[j-1].is_comment_only:
                        doc_start = j
                    else:
                        break
                else:
                    doc_start = j
                j -= 1

            for pat, btype in patterns:
                if btype == "decorator":
                    continue
                m = pat.search(info.effective if not info.is_comment_only else "")
                if not m:
                    m = pat.search(info.raw)
                if m:
                    matched_type = btype
                    matched_name = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                    break

        if matched_type is None:
            i += 1
            continue

        base_indent = info.indent
        sig = lines[i].strip()

        # Find end: next non-blank line at same or lower indent
        block_end = i + 1
        for k in range(i + 1, n):
            if infos[k].is_blank:
                # Peek ahead: if next non-blank is still indented, continue
                peek = k + 1
                while peek < n and infos[peek].is_blank:
                    peek += 1
                if peek < n and infos[peek].indent > base_indent:
                    block_end = k + 1
                    continue
                else:
                    block_end = k
                    break
            if infos[k].indent <= base_indent:
                block_end = k
                break
            block_end = k + 1
        else:
            block_end = n

        # Pop scope stack to find correct parent
        while len(scope_stack) > 1 and scope_stack[-1].indent >= base_indent:
            scope_stack.pop()

        parent = scope_stack[-1]
        node = ScopeNode(
            kind=matched_type, name=matched_name,
            start=i, end=block_end,
            indent=base_indent,
            doc_start=doc_start if doc_start < i else -1,
            signature=sig,
        )
        node.parent = parent
        parent.children.append(node)
        scope_stack.append(node)

        i += 1
        continue


# ════════════════════════════════════════════════════════════
# §5  REFERENCE EXTRACTION
# ════════════════════════════════════════════════════════════

# Patterns that match function/method calls
_CALL_RE = re.compile(
    r'(?:self\.|Self::|(?:\w+)::|\b)'  # optional qualifier
    r'(\w{2,})\s*\('                    # name followed by (
)
# Patterns that match type references
_TYPE_RE = re.compile(
    r'(?::\s*|<|->|impl\s+|trait\s+|struct\s+|enum\s+)'
    r'(\w{2,})'
)

# Common identifiers to skip
_REF_NOISE = {
    "if", "for", "while", "match", "loop", "return", "break", "continue",
    "Some", "None", "Ok", "Err", "vec", "print", "println", "eprintln",
    "format", "write", "writeln", "String", "Vec", "Box", "Rc", "Arc",
    "Option", "Result", "HashMap", "HashSet", "BTreeMap",
    "len", "push", "pop", "get", "set", "new", "from", "into",
    "unwrap", "expect", "map", "filter", "collect", "iter",
    "to_string", "as_str", "as_ref", "clone", "default",
    "range", "enumerate", "zip", "take", "skip",
    "isinstance", "type", "super", "init", "self",
    "parseInt", "parseFloat", "console", "log", "warn", "error",
    "append", "extend", "insert", "remove", "sort", "reverse",
    "join", "split", "strip", "trim", "replace", "contains",
    "startswith", "endswith", "find", "index",
}


def extract_refs(lines: List[str], start: int, end: int) -> Set[str]:
    """Extract function calls and type references from a code region."""
    text = "\n".join(lines[start:end])
    refs: Set[str] = set()

    for m in _CALL_RE.finditer(text):
        name = m.group(1)
        if name not in _REF_NOISE and not name.startswith("__"):
            refs.add(name)

    for m in _TYPE_RE.finditer(text):
        name = m.group(1)
        if name not in _REF_NOISE and name[0].isupper():
            refs.add(name)

    return refs


# ════════════════════════════════════════════════════════════
# §6  CHUNK EMISSION
# ════════════════════════════════════════════════════════════

def emit_chunks(
    root: ScopeNode,
    lines: List[str],
    filename: str,
    lang: str,
    max_chunk_lines: int = 150,
    add_metadata: bool = True,
    add_refs: bool = True,
) -> List[Chunk]:
    """
    Walk the scope tree and emit chunks.

    Each syntactic block becomes one chunk.  Blocks larger than
    max_chunk_lines are split at internal semantic boundaries.
    Gaps between blocks (imports, module-level code) are captured too.
    """
    chunks: List[Chunk] = []

    # Collect the top-level blocks and fill gaps
    top_blocks = _linearize_with_gaps(root, lines)

    for node in top_blocks:
        start = node.effective_start
        end = node.end
        count = end - start

        if count <= 0:
            continue

        text = "\n".join(lines[start:end])
        if not text.strip():
            continue

        # Extract references if enabled
        refs: Set[str] = set()
        if add_refs and node.kind not in ("gap", "preamble", "tail"):
            refs = extract_refs(lines, start, end)
            node.refs = refs

        # Determine parent scope name for context
        parent_scope = ""
        if node.parent and node.parent.kind != "file":
            parent_scope = f"{node.parent.kind} {node.parent.name}"

        # Build type declaration context for impl/method chunks
        type_context = ""
        if node.kind in ("impl", "method") and node.parent:
            type_context = _find_type_context(root, node, lines)

        if count <= max_chunk_lines:
            # Single chunk — exact block boundaries
            source = _make_source(filename, node)
            chunk_text = text
            if type_context:
                chunk_text = f"// Context: {type_context}\n{chunk_text}"
            if add_metadata:
                chunk_text = _enrich(filename, lang, node, chunk_text, refs)

            chunks.append(Chunk(
                source=source, text=chunk_text,
                kind="block", name=node.name,
                block_type=node.kind, file=filename,
                refs=sorted(refs), parent_scope=parent_scope,
            ))
        else:
            # Oversized block — split at semantic sub-boundaries
            sub_chunks = _split_oversized(
                node, lines, filename, lang,
                max_chunk_lines, add_metadata, refs,
                parent_scope, type_context,
            )
            chunks.extend(sub_chunks)

    # Filter out noise: tiny gap/tail/header chunks with no real content
    return [c for c in chunks if not _is_noise_chunk(c)]


def _is_noise_chunk(chunk: Chunk) -> bool:
    """Return True if a chunk is too small/empty to be worth embedding."""
    if chunk.kind in ("file_summary", "block_part"):
        return False
    if chunk.block_type not in ("gap", "tail", "impl_header", "preamble"):
        return False
    # Count non-trivial lines (not blank, not just braces/punctuation)
    meaningful = 0
    for line in chunk.text.split("\n"):
        stripped = line.strip()
        # Skip metadata header line
        if stripped.startswith("File:") and "Language:" in stripped:
            continue
        if stripped and stripped not in ("}", "{", "};", "}", ");", "end"):
            meaningful += 1
    return meaningful < 3


def _linearize_with_gaps(root: ScopeNode, lines: List[str]) -> List[ScopeNode]:
    """
    Take the top-level children of root and interleave gap nodes
    for any uncovered line ranges.  Recurse into large containers
    (impl, class, module) to expose their children as top-level chunks.
    """
    # Flatten: for container nodes (impl, class, module, namespace),
    # expose their children directly but keep the container itself
    # as a chunk if it has meaningful preamble.
    flat: List[ScopeNode] = []

    for child in sorted(root.children, key=lambda n: n.effective_start):
        if child.children and child.kind in ("impl", "class", "module",
                                              "namespace", "test_module"):
            # Emit the container's own preamble (signature, opening lines)
            # then each child method/function as its own chunk
            _flatten_container(child, flat)
        else:
            flat.append(child)

    if not flat:
        # No blocks detected — emit the whole file as one or more gaps
        if len(lines) > 0:
            flat.append(ScopeNode(
                kind="gap", name="<file>",
                start=0, end=len(lines), indent=0,
            ))
        return flat

    # Fill gaps between blocks
    result: List[ScopeNode] = []
    prev_end = 0

    for node in sorted(flat, key=lambda n: n.effective_start):
        gap_start = prev_end
        block_start = node.effective_start

        if gap_start < block_start:
            # There's a gap
            gap_text = "\n".join(lines[gap_start:block_start]).strip()
            if gap_text:
                result.append(ScopeNode(
                    kind="preamble" if prev_end == 0 else "gap",
                    name="", start=gap_start, end=block_start, indent=0,
                ))

        result.append(node)
        prev_end = max(prev_end, node.end)

    # Trailing content
    if prev_end < len(lines):
        tail_text = "\n".join(lines[prev_end:]).strip()
        if tail_text:
            result.append(ScopeNode(
                kind="tail", name="",
                start=prev_end, end=len(lines), indent=0,
            ))

    return result


def _flatten_container(container: ScopeNode, out: List[ScopeNode]):
    """
    Flatten a container (impl, class) by emitting its preamble
    and each child as a separate block.  Children retain their
    parent pointer for context.
    """
    # The container's preamble: from its start to the first child's start
    if container.children:
        first_child_start = min(c.effective_start for c in container.children)
        if container.effective_start < first_child_start:
            preamble_text_end = first_child_start
            out.append(ScopeNode(
                kind=f"{container.kind}_header", name=container.name,
                start=container.effective_start, end=preamble_text_end,
                indent=container.indent,
                signature=container.signature,
                parent=container.parent,
            ))

        for child in sorted(container.children, key=lambda n: n.effective_start):
            child.parent = container  # ensure parent ref
            if child.children and child.kind in ("impl", "class", "module"):
                _flatten_container(child, out)
            else:
                out.append(child)

        # Trailing content after last child
        last_child_end = max(c.end for c in container.children)
        if last_child_end < container.end:
            trailing = "\n".join(
                [] # we skip trailing close-braces — they're noise
            )
    else:
        # Container with no detected children — emit as-is
        out.append(container)


# ── Semantic splitting of oversized blocks ──────────────────

def _split_oversized(
    node: ScopeNode,
    lines: List[str],
    filename: str,
    lang: str,
    max_lines: int,
    add_metadata: bool,
    refs: Set[str],
    parent_scope: str,
    type_context: str,
) -> List[Chunk]:
    """
    Split a block that exceeds max_lines at internal semantic boundaries.
    Each part carries the function signature as context and
    continuation hints linking to adjacent parts.
    """
    start = node.effective_start
    end = node.end
    block_lines = lines[start:end]

    # Find split points: ranked by strength
    split_candidates = _find_split_points(block_lines, lang, node.indent)

    # Greedily split at the best boundaries
    parts = _greedy_split(block_lines, split_candidates, max_lines)

    total_parts = len(parts)
    result: List[Chunk] = []

    for idx, (part_start, part_end) in enumerate(parts):
        abs_start = start + part_start
        abs_end = start + part_end
        part_text = "\n".join(lines[abs_start:abs_end])

        if not part_text.strip():
            continue

        # Build continuation hints
        hints: List[str] = []
        if type_context:
            hints.append(f"// Context: {type_context}")
        if node.signature:
            hints.append(f"// Signature: {node.signature}")
        if idx > 0:
            prev_end_line = lines[start + parts[idx-1][1] - 1].strip()
            hints.append(f"// \u2190 part {idx}/{total_parts}: ...{prev_end_line[:60]}")
        if idx < total_parts - 1:
            next_start_line = lines[start + parts[idx+1][0]].strip()
            hints.append(f"// \u2192 continues in part {idx+2}/{total_parts}: {next_start_line[:60]}...")

        full_text = "\n".join(hints) + "\n" + part_text if hints else part_text

        part_node = ScopeNode(
            kind=node.kind, name=node.name,
            start=abs_start, end=abs_end,
            indent=node.indent,
            signature=node.signature,
        )

        if add_metadata:
            full_text = _enrich(
                filename, lang, part_node, full_text, refs,
                part_info=(idx + 1, total_parts),
            )

        source = _make_source(filename, node, part=(idx + 1, total_parts))

        result.append(Chunk(
            source=source, text=full_text,
            kind="block_part", name=node.name,
            block_type=node.kind, file=filename,
            refs=sorted(refs), parent_scope=parent_scope,
            part={"index": idx + 1, "total": total_parts},
        ))

    return result


@dataclass
class SplitPoint:
    """A candidate location to split a block, with a strength score."""
    line: int       # line offset within the block
    strength: int   # higher = better split point

    # Strength levels:
    # 5 = double blank line
    # 4 = section comment (// ── or # ── or //===)
    # 3 = single blank line after a complete sub-scope
    # 2 = single blank line
    # 1 = comment line


_SECTION_COMMENT_RE = re.compile(
    r'^\s*(?://|#|/\*)\s*[═─━─\-=]{3,}'
)


def _find_split_points(block_lines: List[str], lang: str,
                       base_indent: int) -> List[SplitPoint]:
    """Identify semantic split points within a block."""
    points: List[SplitPoint] = []
    n = len(block_lines)

    for i in range(1, n - 1):  # skip first and last lines
        line = block_lines[i]
        stripped = line.strip()

        # Double blank line
        if not stripped and i + 1 < n and not block_lines[i + 1].strip():
            points.append(SplitPoint(i, 5))
            continue

        # Section comment
        if _SECTION_COMMENT_RE.match(line):
            points.append(SplitPoint(i, 4))
            continue

        # Single blank line
        if not stripped:
            # Check if previous line closed a sub-scope (brace back to base)
            if i > 0 and '}' in block_lines[i - 1]:
                points.append(SplitPoint(i, 3))
            else:
                points.append(SplitPoint(i, 2))
            continue

        # Standalone comment line (not inside code)
        if stripped.startswith("//") or stripped.startswith("#"):
            if i > 0 and not block_lines[i - 1].strip():
                points.append(SplitPoint(i, 1))

    return points


def _greedy_split(block_lines: List[str], candidates: List[SplitPoint],
                  max_lines: int) -> List[Tuple[int, int]]:
    """
    Greedily split the block into parts, each ≤ max_lines,
    choosing the highest-strength split point within each window.
    """
    n = len(block_lines)
    if n <= max_lines:
        return [(0, n)]

    # Sort candidates by line position
    candidates.sort(key=lambda sp: sp.line)

    parts: List[Tuple[int, int]] = []
    pos = 0

    while pos < n:
        if n - pos <= max_lines:
            parts.append((pos, n))
            break

        # Find the best split point within [pos, pos + max_lines)
        window_end = pos + max_lines
        best: Optional[SplitPoint] = None

        for sp in candidates:
            if sp.line <= pos:
                continue
            if sp.line >= window_end:
                break
            # Prefer the highest-strength point closest to the middle
            # but any strong point beats a weak one
            if best is None or sp.strength > best.strength:
                best = sp
            elif (sp.strength == best.strength
                  and abs(sp.line - (pos + max_lines // 2))
                      < abs(best.line - (pos + max_lines // 2))):
                best = sp

        if best:
            parts.append((pos, best.line))
            pos = best.line
        else:
            # No good split point — force split at max
            parts.append((pos, window_end))
            pos = window_end

    return parts


# ── Type context resolution ─────────────────────────────────

def _find_type_context(root: ScopeNode, node: ScopeNode,
                       lines: List[str]) -> str:
    """
    For an impl block or method, find the associated struct/class/trait
    signature and return a compact version for context injection.
    """
    target_name = ""
    if node.kind == "impl" and node.name:
        target_name = node.name
    elif node.parent and node.parent.kind in ("impl", "class"):
        target_name = node.parent.name

    if not target_name:
        return ""

    # Search the tree for the struct/class/trait with this name
    found = _find_node_by_name(root, target_name,
                               {"struct", "class", "trait", "interface",
                                "enum", "protocol"})
    if found:
        # Return a compact signature: the struct/class line + field names
        sig = found.signature or lines[found.start].strip()
        # If it's a struct with fields, grab just the field names
        if found.end - found.start > 1 and found.end - found.start < 30:
            body = "\n".join(lines[found.start:found.end])
            return _compact_type_sig(body)
        return sig

    return ""


def _find_node_by_name(node: ScopeNode, name: str,
                       kinds: Set[str]) -> Optional[ScopeNode]:
    """DFS search for a node with the given name and kind."""
    if node.name == name and node.kind in kinds:
        return node
    for child in node.children:
        found = _find_node_by_name(child, name, kinds)
        if found:
            return found
    return None


def _compact_type_sig(body: str) -> str:
    """Reduce a struct/class body to a compact signature."""
    lines_list = body.strip().split("\n")
    if len(lines_list) <= 3:
        return " ".join(l.strip() for l in lines_list)
    # Take first line (struct Name {), last line (}), and field names
    first = lines_list[0].strip()
    fields: List[str] = []
    for l in lines_list[1:]:
        stripped = l.strip()
        if stripped and stripped != "}" and stripped != "{":
            # Extract field name
            name_match = re.match(r'(?:pub\s+)?(\w+)\s*:', stripped)
            if name_match:
                fields.append(name_match.group(1))
            else:
                # Might be a method or other — just take first word
                word = stripped.split()[0].rstrip(':,')
                if word not in ("//", "///", "/*", "*"):
                    fields.append(word)
    if fields:
        return f"{first} /* fields: {', '.join(fields[:15])} */  }}"
    return first


# ── Metadata enrichment ─────────────────────────────────────

def _enrich(filename: str, lang: str, node: ScopeNode,
            text: str, refs: Set[str],
            part_info: Optional[Tuple[int, int]] = None) -> str:
    """Prepend metadata header to chunk text."""
    lang_display = LANG_DISPLAY.get(lang, lang.title())
    contents = _detect_contents(text)

    type_str = node.kind
    if part_info:
        type_str = f"{node.kind} (part {part_info[0]}/{part_info[1]})"

    name_part = f" | Block: {node.name}" if node.name else ""
    refs_part = ""
    if refs:
        top_refs = sorted(refs)[:10]
        refs_part = f" | Refs: {', '.join(top_refs)}"

    header = (
        f"File: {filename} | Language: {lang_display} | "
        f"Type: {type_str}{name_part} | "
        f"Lines: {node.effective_start + 1}-{node.end} | "
        f"Contains: {contents}{refs_part}"
    )
    return f"{header}\n{text}"


def _detect_contents(text: str) -> str:
    """Detect what constructs appear in a chunk."""
    tags: List[str] = []
    checks = [
        (["fn ", "def ", "function ", "func "], "functions"),
        (["struct ", "class ", "interface "], "types"),
        (["enum "], "enums"),
        (["impl "], "impl"),
        (["trait "], "traits"),
        (["use ", "import ", "require(", "#include"], "imports"),
        (["const ", "static ", "let "], "declarations"),
    ]
    for patterns, tag in checks:
        if any(p in text for p in patterns):
            tags.append(tag)

    test_markers = ["#[test]", "#[cfg(test)]", "def test_", "@Test",
                    "@test", "describe(", "it(", "test("]
    if any(m in text for m in test_markers):
        tags.append("tests")

    error_markers = ["Error", "Result<", "unwrap(", "expect(",
                     "panic!", "try ", "catch ", "except ", "raise "]
    if any(m in text for m in error_markers):
        tags.append("error_handling")

    return ", ".join(tags) if tags else "code"


def _make_source(filename: str, node: ScopeNode,
                 part: Optional[Tuple[int, int]] = None) -> str:
    """Build a structured source label."""
    base = f"{filename}:{node.kind}"
    if node.name:
        base += f":{node.name}"
    base += f":{node.effective_start + 1}-{node.end}"
    if part:
        base += f"[{part[0]}/{part[1]}]"
    return base


# ════════════════════════════════════════════════════════════
# §7  FILE SUMMARY GENERATION
# ════════════════════════════════════════════════════════════

def generate_file_summary(
    root: ScopeNode,
    lines: List[str],
    filename: str,
    lang: str,
) -> Chunk:
    """
    Generate a synthetic file-level summary chunk that acts as a
    table-of-contents for broad queries.
    """
    lang_display = LANG_DISPLAY.get(lang, lang.title())
    total_lines = len(lines)

    # Collect block inventory
    inventory: Dict[str, List[str]] = {}
    all_refs: Set[str] = set()
    _collect_inventory(root, inventory, all_refs, lines)

    parts: List[str] = [
        f"File: {filename} | Language: {lang_display} | "
        f"Lines: {total_lines} | Type: file_summary",
        "",
    ]

    # Summarize by block type
    for kind in ("function", "method", "struct", "class", "enum",
                 "trait", "interface", "impl", "module", "namespace",
                 "type_alias", "constant", "macro", "test_module"):
        names = inventory.get(kind, [])
        if names:
            kind_display = kind.replace("_", " ").title()
            if len(names) <= 8:
                parts.append(f"{kind_display}s: {', '.join(names)}")
            else:
                parts.append(
                    f"{kind_display}s ({len(names)}): "
                    f"{', '.join(names[:6])}, ... +{len(names)-6} more"
                )

    # Import summary
    imports = _extract_imports(lines, lang)
    if imports:
        if len(imports) <= 10:
            parts.append(f"Imports: {', '.join(imports)}")
        else:
            parts.append(
                f"Imports ({len(imports)}): "
                f"{', '.join(imports[:8])}, ... +{len(imports)-8} more"
            )

    # Key references (most-referenced identifiers across blocks)
    if all_refs:
        top_refs = sorted(all_refs)[:15]
        parts.append(f"Key references: {', '.join(top_refs)}")

    text = "\n".join(parts)

    return Chunk(
        source=f"{filename}:summary",
        text=text,
        kind="file_summary",
        name=filename,
        block_type="file_summary",
        file=filename,
        refs=sorted(all_refs)[:20],
    )


def _collect_inventory(node: ScopeNode, inv: Dict[str, List[str]],
                       refs: Set[str], lines: List[str]):
    """Recursively collect block names and references."""
    if node.kind != "file" and node.name:
        inv.setdefault(node.kind, []).append(node.name)
    if node.refs:
        refs.update(node.refs)
    for child in node.children:
        _collect_inventory(child, inv, refs, lines)


def _extract_imports(lines: List[str], lang: str) -> List[str]:
    """Extract imported module/crate names."""
    imports: List[str] = []
    for line in lines[:100]:  # imports are usually at the top
        stripped = line.strip()
        if lang == "rust":
            m = re.match(r'^use\s+(?:crate::)?(\w+)', stripped)
            if m:
                imports.append(m.group(1))
        elif lang == "python":
            m = re.match(r'^(?:from\s+(\w+)|import\s+(\w+))', stripped)
            if m:
                imports.append(m.group(1) or m.group(2))
        elif lang in ("javascript", "typescript"):
            m = re.match(r'^import\s+.*?from\s+[\'"]([^\'"]+)', stripped)
            if m:
                imports.append(m.group(1).split("/")[-1])
        elif lang == "go":
            m = re.match(r'^\s*"([^"]+)"', stripped)
            if m:
                imports.append(m.group(1).split("/")[-1])
        elif lang in ("c", "c++"):
            m = re.match(r'^#include\s*[<"]([^>"]+)', stripped)
            if m:
                imports.append(m.group(1))
        elif lang == "java":
            m = re.match(r'^import\s+(?:static\s+)?([\w.]+)', stripped)
            if m:
                imports.append(m.group(1).split(".")[-1])

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            result.append(imp)
    return result


# ════════════════════════════════════════════════════════════
# §8  MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def chunk_file(
    filename: str,
    content: str,
    language: str = "",
    max_chunk_lines: int = 150,
    add_metadata: bool = True,
    add_summaries: bool = True,
    add_refs: bool = True,
) -> List[Dict]:
    """
    Main entry point: lex → parse → extract refs → emit chunks.
    Returns a list of dicts ready for JSON serialization.
    """
    lang = detect_language(filename, language)
    lines = content.split("\n")

    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return []

    # Step 1: Lex
    line_infos = scan_source(content, lang)

    # Step 2: Parse scope tree
    root = parse_scope_tree(line_infos, lines, lang)

    # Step 3: Extract references for all blocks
    if add_refs:
        _extract_all_refs(root, lines)

    # Step 4: Emit chunks
    chunks = emit_chunks(
        root, lines, filename, lang,
        max_chunk_lines=max_chunk_lines,
        add_metadata=add_metadata,
        add_refs=add_refs,
    )

    # Step 5: File summary
    if add_summaries and len(lines) > 20:
        summary = generate_file_summary(root, lines, filename, lang)
        chunks.insert(0, summary)

    # Convert to dicts
    return [_chunk_to_dict(c) for c in chunks]


def _extract_all_refs(node: ScopeNode, lines: List[str]):
    """Recursively extract refs for all nodes in the tree."""
    if node.kind != "file":
        node.refs = extract_refs(lines, node.effective_start, node.end)
    for child in node.children:
        _extract_all_refs(child, lines)


def _chunk_to_dict(c: Chunk) -> Dict:
    """Convert a Chunk to a JSON-serializable dict."""
    d: Dict = {
        "source": c.source,
        "text": c.text,
        "kind": c.kind,
    }
    if c.name:
        d["name"] = c.name
    if c.block_type:
        d["block_type"] = c.block_type
    if c.file:
        d["file"] = c.file
    if c.refs:
        d["refs"] = c.refs
    if c.part:
        d["part"] = c.part
    if c.parent_scope:
        d["parent_scope"] = c.parent_scope
    return d


# ════════════════════════════════════════════════════════════
# §9  CLI INTERFACE
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CodeWriter RAG Chunker v2 — Lexer/Parser-based chunking"
    )
    parser.add_argument("--max-chunk-lines", type=int, default=150,
                        help="Hard limit before splitting a block (default: 150)")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Skip metadata header enrichment")
    parser.add_argument("--no-summaries", action="store_true",
                        help="Skip file-level summary chunks")
    parser.add_argument("--no-refs", action="store_true",
                        help="Skip cross-reference extraction")
    parser.add_argument("--json-pretty", action="store_true",
                        help="Pretty-print output JSON")

    # Backward-compatible aliases
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="(Ignored — kept for v1 compatibility)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="(Ignored — kept for v1 compatibility)")
    parser.add_argument("--min-merge", type=int, default=None,
                        help="(Ignored — kept for v1 compatibility)")

    args = parser.parse_args()

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print("[]")
            return
        files = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON input: {e}"}),
              file=sys.stderr)
        sys.exit(1)

    if not isinstance(files, list):
        files = [files]

    all_chunks: List[Dict] = []

    for f in files:
        name = f.get("name", "unknown")
        content = f.get("content", "")
        language = f.get("language", "")

        chunks = chunk_file(
            filename=name,
            content=content,
            language=language,
            max_chunk_lines=args.max_chunk_lines,
            add_metadata=not args.no_metadata,
            add_summaries=not args.no_summaries,
            add_refs=not args.no_refs,
        )
        all_chunks.extend(chunks)

    indent = 2 if args.json_pretty else None
    print(json.dumps(all_chunks, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
