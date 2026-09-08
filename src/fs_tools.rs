//! In-process filesystem tools for the agentic loop.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::tools::ToolResult;

/// Byte ceiling on any single result. Matches `tools.rs` MAX_OUTPUT so the
/// model sees one output contract regardless of which tool it called — the
/// loop's token budget then cuts further if the window is tight.
const MAX_OUTPUT: usize = 8192;
/// Lines `read_file` returns when the model does not say. The byte cap above
/// usually binds first on real source; the footer names where it stopped so
/// the next call can page from there.
const DEFAULT_READ_LINES: usize = 2000;
/// Result lines `grep_files` returns before truncating.
const DEFAULT_GREP_LIMIT: usize = 250;
/// Paths `glob_files` returns before truncating.
const MAX_GLOB_RESULTS: usize = 200;
/// Entries `list_dir` returns before truncating.
const MAX_LS_ENTRIES: usize = 400;
/// Bytes sniffed for a NUL when deciding a file is binary.
const SNIFF_BYTES: usize = 8192;

/// Directories never walked by `glob_files` / `grep_files`. A build tree or a
/// vendored dependency set outnumbers the source by orders of magnitude, and
/// the model has no way to tell it is drowning — it just gets 200 hits in
/// `target/` and concludes the symbol is everywhere.
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", "out", "vendor",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".idea",
    ".vscode", ".cargo", ".next", ".nuxt", "coverage",
];

/// Depth cap for the recursive walk. Deep enough for any real source tree,
/// shallow enough that a symlink loop the filesystem did not catch terminates.
const MAX_WALK_DEPTH: usize = 24;

/// Stateless. It held a set of files-already-read while `write_file` gated
/// overwrites on having read the file first; that gate is gone (see
/// `write_file`), and a safety mechanism nothing consults is worse than none,
/// because it still reads like protection.
pub struct FsTools;

impl FsTools {
    pub fn new() -> Self {
        Self
    }

    /// Dispatch one call. `root` is the workspace the UI selected (or the
    /// server's own directory when it selected none); nothing outside it is
    /// reachable. `args` is already normalised by `tools::normalize_args`.
    pub fn execute(&self, name: &str, args: &serde_json::Value, root: &Path) -> ToolResult {
        match name {
            "read_file" => self.read_file(args, root),
            "write_file" => self.write_file(args, root),
            "edit_file" => self.edit_file(args, root),
            "list_dir" => list_dir(args, root),
            "glob_files" => glob_files(args, root),
            "grep_files" => grep_files(args, root),
            other => ToolResult::err(format!("error: unknown tool '{other}'")),
        }
    }

    fn read_file(&self, args: &serde_json::Value, root: &Path) -> ToolResult {
        let raw = match arg_str(args, &["file_path", "path"]) {
            Some(s) => s,
            None => {
                return ToolResult::err(
                    "error: read_file needs `file_path` — the file to read".into(),
                )
            }
        };
        let path = match resolve(root, raw) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(e),
        };
        if !path.exists() {
            return ToolResult::err(format!("error: no such file: {}", rel_path(root, &path)));
        }
        if path.is_dir() {
            return ToolResult::err(format!(
                "error: {} is a directory — use list_dir to see what is in it",
                rel_path(root, &path)
            ));
        }
        if is_binary(&path) {
            return ToolResult::err(format!("error: {} is a binary file", rel_path(root, &path)));
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("error: {e}")),
        };
        if content.is_empty() {
            return ToolResult::ok(format!("[{} is empty]", rel_path(root, &path)));
        }

        let lines: Vec<&str> = content.lines().collect();
        // 1-based, matching the numbers the output prints — a model that just
        // read `412\tfoo` and wants the next page passes 413, not 412.
        let start = arg_usize(args, &["offset", "start", "start_line"]).unwrap_or(1).max(1);
        if start > lines.len() {
            return ToolResult::err(format!(
                "error: offset {start} is past the end of {} ({} lines)",
                rel_path(root, &path),
                lines.len()
            ));
        }
        let limit = arg_usize(args, &["limit", "count", "lines"]).unwrap_or(DEFAULT_READ_LINES).max(1);

        let mut out = String::new();
        let mut shown = 0usize;
        let mut capped = false;
        for (i, line) in lines.iter().enumerate().skip(start - 1).take(limit) {
            let entry = format!("{}\t{}\n", i + 1, line);
            // Stop on a whole line: a half line with a line number in front of
            // it reads as real content and gets quoted back as if it were.
            if out.len() + entry.len() > MAX_OUTPUT && shown > 0 {
                capped = true;
                break;
            }
            out.push_str(&entry);
            shown += 1;
        }
        let last = start + shown - 1;
        if capped || last < lines.len() || start > 1 {
            out.push_str(&format!(
                "\n[showing lines {start}-{last} of {}{}]",
                lines.len(),
                if last < lines.len() {
                    format!(" — continue with offset={}", last + 1)
                } else {
                    String::new()
                }
            ));
        }
        ToolResult::ok(out)
    }

    fn write_file(&self, args: &serde_json::Value, root: &Path) -> ToolResult {
        let raw = match arg_str(args, &["file_path", "path"]) {
            Some(s) => s,
            None => {
                return ToolResult::err(
                    "error: write_file needs `file_path` — the file to write".into(),
                )
            }
        };
        let content = match arg_str(args, &["content", "text", "code"]) {
            Some(s) => s,
            None => {
                return ToolResult::err(
                    "error: write_file needs `content` — the full new contents of the file".into(),
                )
            }
        };
        let path = match resolve(root, raw) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(e),
        };
        if path.is_dir() {
            return ToolResult::err(format!("error: {} is a directory", rel_path(root, &path)));
        }
        if path.exists() {
            // write_file creates; edit_file modifies. There is no third case.
            //
            // This started as the reference's read-before-overwrite guard, and
            // a real run showed why that guard is not one: asked a read-only
            // question about this repo, the 7B read a 1662-line source file
            // five times and then called write_file on it with a single line
            // of content. Having-been-read was satisfied — reading first is
            // what the model does anyway — so the guard would have passed the
            // clobber through. Only the tool-round budget stopped it, and
            // there is no git history here to recover from.
            //
            // A whole-file rewrite is still reachable: edit_file with the old
            // contents as old_string. That is the same operation with the
            // destroyed text quoted, which is exactly the step worth forcing.
            let lines = std::fs::read_to_string(&path).map(|c| c.lines().count());
            return ToolResult::err(format!(
                "error: {} already exists{} — write_file only creates new files. \
                 Use edit_file to change it.",
                rel_path(root, &path),
                lines.map_or(String::new(), |n| format!(" ({n} lines)"))
            ));
        }
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                return ToolResult::err(format!("error: cannot create {}: {e}", rel_path(root, dir)));
            }
        }
        match write_verified(&path, content) {
            Ok(()) => {
                ToolResult::ok(format!(
                    "wrote {} ({} lines, {} bytes)",
                    rel_path(root, &path),
                    content.lines().count(),
                    content.len()
                ))
            }
            Err(e) => ToolResult::err(format!("error: {e}")),
        }
    }

    fn edit_file(&self, args: &serde_json::Value, root: &Path) -> ToolResult {
        let raw = match arg_str(args, &["file_path", "path"]) {
            Some(s) => s,
            None => {
                return ToolResult::err(
                    "error: edit_file needs `file_path` — the file to change".into(),
                )
            }
        };
        let (Some(old), Some(new)) = (
            arg_str(args, &["old_string", "old", "search"]),
            arg_str(args, &["new_string", "new", "replace"]).or(Some("")),
        ) else {
            return ToolResult::err(
                "error: edit_file needs `old_string` (exact text to find) and `new_string`".into(),
            );
        };
        if old.is_empty() {
            return ToolResult::err(
                "error: edit_file `old_string` is empty — pass the exact text to replace, \
                 or use write_file to create the file whole"
                    .into(),
            );
        }
        if old == new {
            return ToolResult::err(
                "error: edit_file `old_string` and `new_string` are identical".into(),
            );
        }
        let path = match resolve(root, raw) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(e),
        };
        if !path.exists() {
            return ToolResult::err(format!("error: no such file: {}", rel_path(root, &path)));
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("error: {e}")),
        };
        let hits = content.matches(old).count();
        if hits == 0 {
            // The overwhelmingly common cause is whitespace, and a model that
            // is only told "not found" retries the same string with a
            // different guess at the surrounding lines.
            return ToolResult::err(format!(
                "error: `old_string` does not appear in {}. It must match the file byte for \
                 byte, including indentation — read_file the region and copy it exactly.",
                rel_path(root, &path)
            ));
        }
        let replace_all = arg_bool(args, &["replace_all", "all", "global"]).unwrap_or(false);
        if hits > 1 && !replace_all {
            return ToolResult::err(format!(
                "error: `old_string` appears {hits} times in {}. Include more surrounding \
                 lines so it is unique, or pass replace_all=true.",
                rel_path(root, &path)
            ));
        }
        let updated = if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        };
        match write_verified(&path, &updated) {
            Ok(()) => {
                let before = content.lines().count();
                let after = updated.lines().count();
                ToolResult::ok(format!(
                    "edited {} — {hits} replacement{} ({before} → {after} lines)",
                    rel_path(root, &path),
                    if hits == 1 { "" } else { "s" }
                ))
            }
            Err(e) => ToolResult::err(format!("error: {e}")),
        }
    }
}

impl Default for FsTools {
    fn default() -> Self {
        Self::new()
    }
}

fn list_dir(args: &serde_json::Value, root: &Path) -> ToolResult {
    let raw = arg_str(args, &["path", "dir", "directory", "file_path"]).unwrap_or(".");
    let dir = match resolve(root, raw) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(e),
    };
    if !dir.exists() {
        return ToolResult::err(format!("error: no such directory: {}", rel_path(root, &dir)));
    }
    if !dir.is_dir() {
        return ToolResult::err(format!(
            "error: {} is a file — use read_file",
            rel_path(root, &dir)
        ));
    }
    let show_all = arg_bool(args, &["all", "hidden"]).unwrap_or(false);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => return ToolResult::err(format!("error: {e}")),
    };
    // Directories first, then files, each alphabetical: a listing sorted the
    // way the filesystem happened to return it makes the model re-read it.
    let mut dirs: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !show_all && name.starts_with('.') {
            continue;
        }
        let meta = entry.metadata().ok();
        let is_dir = meta.as_ref().is_some_and(|m| m.is_dir());
        if is_dir {
            dirs.push(format!("d {:>8}  {name}/", ""));
        } else {
            let size = meta.as_ref().map_or(0, |m| m.len());
            let kind = if entry.file_type().is_ok_and(|t| t.is_symlink()) { 'l' } else { '-' };
            files.push(format!("{kind} {:>8}  {name}", human_size(size)));
        }
    }
    dirs.sort();
    files.sort();
    let total = dirs.len() + files.len();
    if total == 0 {
        return ToolResult::ok(format!("{}: (empty)", rel_path(root, &dir)));
    }
    let mut out = format!("{}:\n", rel_path(root, &dir));
    for line in dirs.iter().chain(files.iter()).take(MAX_LS_ENTRIES) {
        out.push_str(line);
        out.push('\n');
    }
    if total > MAX_LS_ENTRIES {
        out.push_str(&format!("[{} more entries not shown]\n", total - MAX_LS_ENTRIES));
    }
    ToolResult::ok(cap_output(out))
}

/// A search that returns nothing is retried with the constraint most likely
/// to have been the mistake removed, up to this many times.
///
/// Not prompt guidance — enforced here. A model that gets an empty result
/// reads it as a fact about the codebase, not as a fact about its own
/// filter, and stops looking: observed live, a 7B narrowed to `glob="*.py"`
/// in a Rust repo, took the empty result as proof the symbol did not exist,
/// and spent every remaining round asserting so. Widening costs one tree
/// walk; being wrong costs the whole turn.
const MAX_SEARCH_RETRIES: usize = 3;

/// One rung of the widening ladder: what to search with, and how to say what
/// was relaxed to get there.
struct Attempt<T> {
    /// Empty for the first attempt — nothing was relaxed yet.
    relaxed: String,
    what: T,
}

/// Build the ladder: the call as made, then up to `MAX_SEARCH_RETRIES`
/// strictly wider versions. Each rung keeps the previous rung's relaxations,
/// so the ladder only ever broadens and cannot cycle.
fn ladder<T>(first: T, mut steps: Vec<(String, T)>) -> Vec<Attempt<T>> {
    let mut out = vec![Attempt { relaxed: String::new(), what: first }];
    steps.truncate(MAX_SEARCH_RETRIES);
    for (relaxed, what) in steps {
        out.push(Attempt { relaxed, what });
    }
    out
}

/// Prefix a widened result with what was actually run, so the model does not
/// read hits from a broader search as answering the narrow question it asked.
///
/// The wording matters more than it looks. This first read
/// "[nothing matched as asked — retried …]" followed by the hits, and the 7B
/// took the leading clause as a failure and re-issued the identical call
/// eight times, burning the whole round budget before answering (correctly)
/// from results it had had since round one. Lead with the answer, put the
/// caveat second, and say outright not to repeat the call.
fn note_widening(relaxed: &str, body: String) -> String {
    if relaxed.is_empty() {
        body
    } else {
        format!(
            "[SUCCESS — results below. They come from retrying {relaxed}, because the \
             call exactly as you wrote it matched nothing. Do not repeat that call; \
             use these results.]\n{body}"
        )
    }
}

fn glob_files(args: &serde_json::Value, root: &Path) -> ToolResult {
    let Some(pattern) = arg_str(args, &["pattern", "glob", "query"]) else {
        return ToolResult::err(
            "error: glob_files needs `pattern` — e.g. \"**/*.rs\" or \"src/**/mod.rs\"".into(),
        );
    };
    let base = match resolve(root, arg_str(args, &["path", "dir", "directory"]).unwrap_or(".")) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(e),
    };
    if !base.is_dir() {
        return ToolResult::err(format!("error: not a directory: {}", rel_path(root, &base)));
    }

    // Widen by pattern, since a glob has no other constraint to give up:
    // first ignore case, then match the pattern anywhere in the name. The
    // second rung is what turns a guessed `config` into `config.toml`.
    // Case travels as a flag, not as an inline `(?i)`: `glob_regex` escapes
    // every regex metacharacter in what it is handed, so a smuggled `(?i)`
    // arrives as four literal characters to match against a filename.
    let mut steps: Vec<(String, (String, bool))> = Vec::new();
    if pattern.to_lowercase() != pattern.to_uppercase() {
        steps.push(("ignoring case".into(), (pattern.to_string(), true)));
    }
    let loose = format!("*{}*", pattern.trim_matches('*'));
    if loose != pattern {
        steps.push((
            format!("with `{loose}`, matching the name anywhere"),
            (loose, true),
        ));
    }

    // The walk is the expensive part and the file list does not change
    // between rungs, so do it once and re-match in memory.
    let mut files: Vec<(std::time::SystemTime, String)> = Vec::new();
    walk(&base, 0, &mut |file, rel| {
        let mtime = file
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        files.push((mtime, rel.to_string()));
    });

    let mut tried: Vec<String> = Vec::new();
    for attempt in ladder((pattern.to_string(), false), steps) {
        let (pat, ci) = &attempt.what;
        // An invalid pattern only matters on the first rung, where the model
        // wrote it; a widened rung that fails to compile is just skipped.
        let re = match glob_regex_ci(pat, *ci) {
            Ok(r) => r,
            Err(e) if attempt.relaxed.is_empty() => {
                return ToolResult::err(format!("error: bad pattern `{pattern}`: {e}"))
            }
            Err(_) => continue,
        };
        let mut hits: Vec<&(std::time::SystemTime, String)> = files
            .iter()
            .filter(|(_, rel)| {
                // Against the relative path AND the bare filename, so both
                // "**/*.rs" and "*.rs" find a nested file — the model uses
                // them interchangeably.
                let name = rel.rsplit('/').next().unwrap_or(rel);
                re.is_match(rel) || re.is_match(name)
            })
            .collect();
        if hits.is_empty() {
            tried.push(if attempt.relaxed.is_empty() {
                format!("`{pattern}`")
            } else {
                attempt.relaxed.clone()
            });
            continue;
        }
        // Newest first: in a repo the model is working in, recency is the
        // best available proxy for relevance.
        hits.sort_by(|a, b| b.0.cmp(&a.0));
        let total = hits.len();
        let mut out = String::new();
        for (_, rel) in hits.iter().take(MAX_GLOB_RESULTS) {
            out.push_str(rel);
            out.push('\n');
        }
        if total > MAX_GLOB_RESULTS {
            out.push_str(&format!(
                "[{} of {total} matches shown, newest first — narrow the pattern for the rest]\n",
                MAX_GLOB_RESULTS
            ));
        }
        return ToolResult::ok(cap_output(note_widening(&attempt.relaxed, out)));
    }
    ToolResult::ok(format!(
        "no file under {} matches, after {} attempt(s): {}. \
         There are {} file(s) there — list_dir or grep_files may find it by content.",
        rel_path(root, &base),
        tried.len(),
        tried.join("; "),
        files.len()
    ))
}

/// What one rung of the grep ladder searches with.
struct GrepPass {
    re: regex::Regex,
    /// `None` once the glob has been given up.
    glob: Option<regex::Regex>,
    /// False once the search has been widened past the requested subdirectory.
    scoped: bool,
}

fn grep_files(args: &serde_json::Value, root: &Path) -> ToolResult {
    let Some(pattern) = arg_str(args, &["pattern", "query", "regex", "search"]) else {
        return ToolResult::err(
            "error: grep_files needs `pattern` — the regular expression to search for".into(),
        );
    };
    let target = match resolve(root, arg_str(args, &["path", "dir", "directory", "file_path"]).unwrap_or(".")) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(e),
    };
    if !target.exists() {
        return ToolResult::err(format!("error: no such path: {}", rel_path(root, &target)));
    }
    let case_insensitive = arg_bool(args, &["-i", "i", "case_insensitive", "ignore_case"])
        .unwrap_or(false);
    let build = |p: &str, ci: bool| {
        regex::RegexBuilder::new(p).case_insensitive(ci).size_limit(1 << 20).build()
    };
    let re = match build(pattern, case_insensitive) {
        Ok(r) => r,
        Err(e) => {
            // Regex syntax the model got wrong is worth naming precisely: the
            // alternative is a "no matches" it reads as "the code is not here".
            return ToolResult::err(format!(
                "error: `{pattern}` is not a valid regular expression: {e}"
            ));
        }
    };
    let glob_arg = arg_str(args, &["glob", "include", "type"]);
    let file_filter = match glob_arg {
        Some(g) => match glob_regex(g) {
            Ok(r) => Some(r),
            Err(e) => return ToolResult::err(format!("error: bad glob `{g}`: {e}")),
        },
        None => None,
    };
    let mode = arg_str(args, &["output_mode", "mode"]).unwrap_or("content");
    let ctx = arg_usize(args, &["-C", "C", "context"]).unwrap_or(0).min(10);
    let limit = arg_usize(args, &["head_limit", "limit", "max_results"])
        .unwrap_or(DEFAULT_GREP_LIMIT)
        .max(1);

    // Ordered by how often each constraint turns out to be the mistake, most
    // likely first: the glob (a guessed extension), then the directory (a
    // guessed location), then case. Each rung keeps the previous relaxations.
    let scoped_start = target != root && target.is_dir();
    let mut steps: Vec<(String, GrepPass)> = Vec::new();
    if file_filter.is_some() {
        steps.push((
            format!("without the `glob` filter (`{}`)", glob_arg.unwrap_or("")),
            GrepPass { re: re.clone(), glob: None, scoped: scoped_start },
        ));
    }
    if scoped_start {
        steps.push((
            "across the whole workspace".into(),
            GrepPass { re: re.clone(), glob: None, scoped: false },
        ));
    }
    if !case_insensitive {
        if let Ok(ci) = build(pattern, true) {
            steps.push((
                "ignoring case".into(),
                GrepPass { re: ci, glob: None, scoped: false },
            ));
        }
    }

    // One walk feeds every rung. Widening past the requested subdirectory
    // means the list has to cover the workspace, so walk from the root and
    // let each rung decide whether to look outside `target`. Binary files are
    // sniffed here, once, rather than on every rung.
    let walk_base = if steps.iter().any(|(_, p)| !p.scoped) { root } else { target.as_path() };
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    if target.is_file() {
        files.push((target.clone(), rel_path(root, &target)));
    } else {
        walk(walk_base, 0, &mut |file, _rel| {
            if !is_binary(file) {
                files.push((file.to_path_buf(), rel_path(root, file)));
            }
        });
    }
    let in_target = |p: &Path| p.starts_with(&target);

    let mut tried: Vec<String> = Vec::new();
    for attempt in ladder(
        GrepPass { re, glob: file_filter, scoped: scoped_start },
        steps,
    ) {
        let pass = &attempt.what;
        let mut lines_out: Vec<String> = Vec::new();
        let mut files_out: Vec<String> = Vec::new();
        let mut counts: Vec<(String, usize)> = Vec::new();
        let mut scanned = 0usize;
        let mut total_matches = 0usize;

        for (path, rel) in &files {
            if pass.scoped && !in_target(path) {
                continue;
            }
            if let Some(f) = &pass.glob {
                let name = rel.rsplit('/').next().unwrap_or(rel);
                if !f.is_match(rel) && !f.is_match(name) {
                    continue;
                }
            }
            let Ok(content) = std::fs::read_to_string(path) else { continue };
            scanned += 1;
            let lines: Vec<&str> = content.lines().collect();
            let mut hits = 0usize;
            // Context windows overlap on dense matches; emitting each one
            // separately repeats the same lines. Mark, then print once.
            let mut wanted: Vec<bool> = vec![false; lines.len()];
            for (i, line) in lines.iter().enumerate() {
                if pass.re.is_match(line) {
                    hits += 1;
                    let lo = i.saturating_sub(ctx);
                    let hi = (i + ctx).min(lines.len().saturating_sub(1));
                    for w in wanted.iter_mut().take(hi + 1).skip(lo) {
                        *w = true;
                    }
                }
            }
            if hits == 0 {
                continue;
            }
            total_matches += hits;
            files_out.push(rel.clone());
            counts.push((rel.clone(), hits));
            if mode != "content" {
                continue;
            }
            let mut prev = 0usize;
            for (i, line) in lines.iter().enumerate() {
                if !wanted[i] {
                    continue;
                }
                // A gap between context windows, so the model does not read
                // two distant regions as adjacent code.
                if prev != 0 && i > prev {
                    lines_out.push("--".into());
                }
                // grep's own convention: ':' is a match, '-' is context.
                let sep = if pass.re.is_match(line) { ':' } else { '-' };
                lines_out.push(format!("{rel}{sep}{}{sep}{line}", i + 1));
                prev = i + 1;
            }
        }

        if total_matches == 0 {
            tried.push(format!(
                "{} — {scanned} file(s)",
                if attempt.relaxed.is_empty() { "as asked".into() } else { attempt.relaxed.clone() }
            ));
            continue;
        }

        let mut out = String::new();
        match mode {
            "files_with_matches" | "files" => {
                files_out.sort();
                for f in files_out.iter().take(limit) {
                    out.push_str(f);
                    out.push('\n');
                }
                if files_out.len() > limit {
                    out.push_str(&format!("[{} more files]\n", files_out.len() - limit));
                }
            }
            "count" => {
                counts.sort();
                for (f, n) in counts.iter().take(limit) {
                    out.push_str(&format!("{f}:{n}\n"));
                }
                out.push_str(&format!(
                    "[{total_matches} matches in {} file(s)]\n",
                    counts.len()
                ));
            }
            _ => {
                for line in lines_out.iter().take(limit) {
                    out.push_str(line);
                    out.push('\n');
                }
                if lines_out.len() > limit {
                    out.push_str(&format!(
                        "[{} of {} lines shown ({total_matches} matches in {} files) — \
                         narrow with `glob` or output_mode=\"files_with_matches\"]\n",
                        limit,
                        lines_out.len(),
                        files_out.len()
                    ));
                }
            }
        }
        return ToolResult::ok(cap_output(note_widening(&attempt.relaxed, out)));
    }

    // Every rung came back empty. Say what was actually searched at each one:
    // "0 file(s)" on the first rung is the model's filter being wrong, and
    // a real count on the last rung is the symbol genuinely not being there.
    ToolResult::ok(format!(
        "no match for `{pattern}` after {} attempt(s): {}. \
         Nothing under {} contains it.",
        tried.len(),
        tried.join("; "),
        rel_path(root, &target)
    ))
}

// ── shared helpers ──

/// Resolve a model-supplied path against the workspace root and refuse
/// anything outside it.
///
/// `canonicalize` alone cannot do this: `write_file` targets a path that does
/// not exist yet. So the lexical join is normalised first (which is what
/// actually stops `../../..`), then the deepest existing ancestor is
/// canonicalised and the containment check is made against that — which is
/// what stops a symlink pointing out of the tree.
fn resolve(root: &Path, raw: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("error: empty path".into());
    }
    let given = Path::new(raw);
    let joined = if given.is_absolute() { given.to_path_buf() } else { root.join(given) };
    let lexical = normalize(&joined);

    // Canonicalise the longest existing prefix, then re-append the rest.
    let mut existing = lexical.as_path();
    let mut tail: Vec<Component> = Vec::new();
    while !existing.exists() {
        let Some(parent) = existing.parent() else { break };
        if let Some(name) = existing.file_name() {
            tail.push(Component::Normal(name));
        }
        existing = parent;
    }
    let mut real = match std::fs::canonicalize(existing) {
        Ok(p) => p,
        Err(_) => lexical.clone(),
    };
    for c in tail.iter().rev() {
        real.push(c.as_os_str());
    }
    let real_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !real.starts_with(&real_root) {
        return Err(format!(
            "error: {} is outside the workspace ({}) — these tools cannot reach it",
            show(&real),
            show(&real_root)
        ));
    }
    Ok(real)
}

/// Lexical `.`/`..` folding. Purely textual on purpose — it runs before the
/// filesystem is touched, so a `..` in a path that does not exist yet is still
/// resolved rather than being handed to `create_dir_all`.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Windows `canonicalize` returns `\\?\C:\…`, which is correct and unusable:
/// the model copies what it is shown into the next call, and the verbatim
/// prefix is not something it has ever seen in a source tree.
fn show(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

/// Workspace-relative, forward-slashed — the form the model should pass back.
///
/// Every message the model reads goes through here rather than through
/// `show`. An absolute Windows path is ~120 characters of context per line,
/// and worse, it is not the form the model should be writing into its next
/// call: showing it `C:\…\ws\src\main.rs` teaches it to send that back.
fn rel_path(root: &Path, p: &Path) -> String {
    let real_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let rel = p.strip_prefix(&real_root).unwrap_or(p);
    let s = rel.display().to_string().replace('\\', "/");
    // The root itself strips to nothing, which would render as a bare ":".
    if s.is_empty() { ".".into() } else { s }
}

/// Recursive walk yielding (absolute path, workspace-relative path) for every
/// file, skipping build trees and hidden directories. pub(crate): the RAG
/// path indexer reuses it so indexing and the fs tools agree on what a
/// project tree is (same SKIP_DIRS, same depth cap).
pub(crate) fn walk(base: &Path, depth: usize, f: &mut impl FnMut(&Path, &str)) {
    walk_from(base, base, depth, f)
}

fn walk_from(base: &Path, dir: &Path, depth: usize, f: &mut impl FnMut(&Path, &str)) {
    if depth > MAX_WALK_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) || name.starts_with('.') {
                continue;
            }
            walk_from(base, &path, depth + 1, f);
        } else if ft.is_file() {
            let rel = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .display()
                .to_string()
                .replace('\\', "/");
            f(&path, &rel);
        }
    }
}

/// Compile a glob to a regex. Uses the real regex engine rather than a
/// hand-rolled matcher so a pattern containing `+`, `(`, or `[` — all legal
/// in a filename and all regex metacharacters — matches the file the model
/// meant instead of quietly matching nothing.
fn glob_regex(pattern: &str) -> Result<regex::Regex, regex::Error> {
    glob_regex_ci(pattern, false)
}

fn glob_regex_ci(pattern: &str, case_insensitive: bool) -> Result<regex::Regex, regex::Error> {
    let mut re = String::from("^");
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    // `**/` matches zero or more directories, so `**/*.rs`
                    // finds `main.rs` at the root as well as nested files.
                    if chars.peek() == Some(&'/') {
                        chars.next();
                        re.push_str("(?:.*/)?");
                    } else {
                        re.push_str(".*");
                    }
                } else {
                    re.push_str("[^/]*");
                }
            }
            '?' => re.push_str("[^/]"),
            '{' => re.push('('),
            '}' => re.push(')'),
            ',' => re.push('|'),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
    }
    re.push('$');
    regex::RegexBuilder::new(&re).case_insensitive(case_insensitive).build()
}

/// NUL in the first 8 KB. The same test every other tool in this tree uses,
/// and the reason `grep_files` never returns a screen of mojibake.
/// pub(crate) for the RAG path indexer.
pub(crate) fn is_binary(path: &Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut buf = [0u8; SNIFF_BYTES];
    let Ok(n) = f.read(&mut buf) else { return false };
    buf[..n].contains(&0)
}

/// Write through a temp file in the same directory, swap it in atomically,
/// then read it back and assert it is what was intended. A partial write that
/// reports success is the one failure the model cannot detect for itself —
/// it will build the rest of the turn on a file that is not what it asked for.
fn write_verified(path: &Path, content: &str) -> Result<(), String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let stem = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let tmp = dir.join(format!(".{stem}.{}.tmp", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, content) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("writing {}: {e}", show(&tmp)));
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("replacing {}: {e}", show(path)));
    }
    match std::fs::read_to_string(path) {
        Ok(back) if back == content => Ok(()),
        Ok(_) => Err(format!(
            "{} does not match what was written — the file was changed by something else, \
             re-read it before trying again",
            show(path)
        )),
        Err(e) => Err(format!("verifying {}: {e}", show(path))),
    }
}

fn cap_output(mut s: String) -> String {
    if s.len() > MAX_OUTPUT {
        let cut = (0..=MAX_OUTPUT).rev().find(|i| s.is_char_boundary(*i)).unwrap_or(0);
        s.truncate(cut);
        s.push_str("\n[truncated]");
    }
    s
}

fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    if bytes < K {
        format!("{bytes}B")
    } else if bytes < K * K {
        format!("{:.1}K", bytes as f64 / K as f64)
    } else if bytes < K * K * K {
        format!("{:.1}M", bytes as f64 / (K * K) as f64)
    } else {
        format!("{:.1}G", bytes as f64 / (K * K * K) as f64)
    }
}

/// First present alias, as a string. Qwen drifts between `file_path` and
/// `path`, `pattern` and `query`, across turns of one session — accepting the
/// synonyms costs nothing and saves a corrective round every time it does.
fn arg_str<'a>(args: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| args.get(*k).and_then(|v| v.as_str()))
}

fn arg_usize(args: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|k| {
        args.get(*k).and_then(|v| {
            // Numbers arrive as strings whenever the call came through the
            // XML template, which has no types.
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
                .map(|n| n as usize)
        })
    })
}

fn arg_bool(args: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|k| {
        args.get(*k).and_then(|v| {
            v.as_bool()
                .or_else(|| match v.as_str()?.trim().to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => Some(true),
                    "false" | "no" | "0" => Some(false),
                    _ => None,
                })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Fresh per-test workspace under the OS temp dir.
    struct Ws(PathBuf);
    impl Ws {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("rl_fs_test_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(std::fs::canonicalize(&dir).unwrap())
        }
        fn root(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Ws {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolve_refuses_traversal_and_absolute_escapes() {
        let ws = Ws::new("resolve");
        assert!(resolve(ws.root(), "../escape.txt").is_err());
        assert!(resolve(ws.root(), "a/../../escape.txt").is_err());
        let outside = std::env::temp_dir().join("rl_fs_outside.txt");
        assert!(resolve(ws.root(), &outside.display().to_string()).is_err());
        // A nested path that does not exist yet must still resolve — that is
        // the whole reason resolve() cannot just canonicalize.
        let ok = resolve(ws.root(), "new/dir/file.txt").unwrap();
        assert!(ok.starts_with(ws.root()));
    }

    #[test]
    fn write_file_creates_but_refuses_overwrite() {
        let ws = Ws::new("write");
        let fs = FsTools::new();
        let r = fs.execute(
            "write_file",
            &json!({"file_path": "sub/new.txt", "content": "one\ntwo\n"}),
            ws.root(),
        );
        assert!(r.ok, "{}", r.output);
        assert_eq!(
            std::fs::read_to_string(ws.root().join("sub/new.txt")).unwrap(),
            "one\ntwo\n"
        );
        let again = fs.execute(
            "write_file",
            &json!({"file_path": "sub/new.txt", "content": "clobber"}),
            ws.root(),
        );
        assert!(!again.ok);
        assert!(again.output.contains("edit_file"), "{}", again.output);
    }

    #[test]
    fn edit_file_enforces_uniqueness_and_names_whitespace_on_miss() {
        let ws = Ws::new("edit");
        let fs = FsTools::new();
        std::fs::write(ws.root().join("f.txt"), "x\ny\nx\n").unwrap();
        let dup = fs.execute(
            "edit_file",
            &json!({"file_path": "f.txt", "old_string": "x", "new_string": "z"}),
            ws.root(),
        );
        assert!(!dup.ok);
        assert!(dup.output.contains("replace_all"), "{}", dup.output);
        let all = fs.execute(
            "edit_file",
            &json!({"file_path": "f.txt", "old_string": "x", "new_string": "z", "replace_all": true}),
            ws.root(),
        );
        assert!(all.ok, "{}", all.output);
        assert_eq!(std::fs::read_to_string(ws.root().join("f.txt")).unwrap(), "z\ny\nz\n");
        let miss = fs.execute(
            "edit_file",
            &json!({"file_path": "f.txt", "old_string": "absent", "new_string": "q"}),
            ws.root(),
        );
        assert!(!miss.ok);
        assert!(miss.output.contains("byte for byte"), "{}", miss.output);
    }

    #[test]
    fn glob_widening_finds_bare_name_and_flags_the_retry() {
        let ws = Ws::new("glob");
        std::fs::write(ws.root().join("config.toml"), "x").unwrap();
        let r = glob_files(&json!({"pattern": "config"}), ws.root());
        assert!(r.ok, "{}", r.output);
        assert!(r.output.starts_with("[SUCCESS — results below"), "{}", r.output);
        assert!(r.output.contains("config.toml"), "{}", r.output);
    }

    #[test]
    fn grep_widening_relaxes_a_wrong_glob_and_reports_true_absence() {
        let ws = Ws::new("grep");
        std::fs::write(ws.root().join("lib.rs"), "pub fn needle() {}\n").unwrap();
        let widened = grep_files(&json!({"pattern": "needle", "glob": "*.py"}), ws.root());
        assert!(widened.ok, "{}", widened.output);
        assert!(widened.output.starts_with("[SUCCESS — results below"), "{}", widened.output);
        assert!(widened.output.contains("lib.rs"), "{}", widened.output);
        let absent = grep_files(&json!({"pattern": "no_such_symbol_anywhere"}), ws.root());
        assert!(absent.output.contains("attempt(s)"), "{}", absent.output);
        assert!(absent.output.contains("Nothing under"), "{}", absent.output);
    }

    #[test]
    fn read_file_pages_with_offset_and_footer() {
        let ws = Ws::new("read");
        let fs = FsTools::new();
        let body: String = (1..=50).map(|i| format!("line{i}\n")).collect();
        std::fs::write(ws.root().join("f.txt"), &body).unwrap();
        let r = fs.execute(
            "read_file",
            &json!({"file_path": "f.txt", "offset": 10, "limit": 5}),
            ws.root(),
        );
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains("10\tline10"), "{}", r.output);
        assert!(r.output.contains("continue with offset=15"), "{}", r.output);
    }
}
