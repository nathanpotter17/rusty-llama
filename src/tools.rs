//! Agentic tool runtime: models with tool-calling support can execute
//! Python/bash, SAVE their own tools into data/py-tools/ and data/bash-tools/,
//! and reuse them across turns. Gated by `[tools] enabled` in config.toml —
//! this executes model-generated code locally, on purpose.
//!
//! Ported from rusty-streamer's streamer-server. Tool-call *parsing* is not
//! ported: llama-server's --jinja templates parse the model's native format
//! and hand us OpenAI-shaped `tool_calls`, so this module only declares and
//! executes tools.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::fs_tools::FsTools;

pub struct ToolRuntime {
    pub py_dir: PathBuf,
    pub bash_dir: PathBuf,
    /// Confinement for the tools that execute code. See Sandbox.
    sandbox: Sandbox,
    /// Resolved bash executable. On Windows a bare "bash" usually hits
    /// System32's WSL relay, which fails without a distro — prefer Git Bash.
    bash_exe: PathBuf,
    /// The filesystem half of the suite. Runs in-process, so it is confined by
    /// path resolution rather than by the sandbox above — see `fs_tools`.
    fs: FsTools,
}

/// Tool names the runtime knows. Shared with the serving layer, which uses
/// them to decide whether a truncated or malformed turn was *trying* to call
/// something (`looks_like_tool_attempt`).
///
/// `write_file` and `edit_file` write inside the selected workspace, and
/// nowhere else: `fs_tools::resolve` refuses any path that leaves it, so the
/// blast radius is the repo the user picked. `save_tool` also writes, but only
/// into the runtime's own tool directories, which `sanitize()` confines to a
/// bare filename.
pub const TOOL_NAMES: [&str; 11] = [
    "run_bash",
    "run_python",
    "save_tool",
    "run_tool",
    "rag_search",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
];

/// Tools handled in-process by `fs_tools` rather than by spawning anything.
const FS_TOOL_NAMES: [&str; 6] = [
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
];

/// Whether a tool belongs to the filesystem set — the set the small hardware
/// tiers declare (and the only set they may execute).
pub fn is_fs_tool(name: &str) -> bool {
    FS_TOOL_NAMES.contains(&name)
}

const TIMEOUT_SECS: u64 = 60;
const MAX_OUTPUT: usize = 8192;

/// Outcome of one tool call: `output` goes to the model, `ok`/`status` let
/// the caller assert success vs failure (UI status chips, [FAILED] framing).
pub struct ToolResult {
    pub ok: bool,
    /// Compact machine-readable status: "ok", "exit N", "timeout", "error".
    pub status: String,
    pub output: String,
}

impl ToolResult {
    pub fn ok(output: String) -> Self {
        Self { ok: true, status: "ok".into(), output }
    }
    pub fn err(output: String) -> Self {
        Self { ok: false, status: "error".into(), output }
    }
}

/// On Windows "python3" is the dead Microsoft Store alias; on Linux it is the
/// name that reliably exists.
fn python_exe() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

fn resolve_bash(config_override: &str) -> PathBuf {
    if !config_override.is_empty() {
        return PathBuf::from(config_override);
    }
    if let Ok(p) = std::env::var("RUSTY_LLAMA_BASH") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    #[cfg(windows)]
    for cand in [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ] {
        if std::path::Path::new(cand).exists() {
            return PathBuf::from(cand);
        }
    }
    PathBuf::from("bash")
}

/// How shell-executing tools are confined.
///
/// The workspace boundary is enforced by the kernel or it does not exist: a
/// command filter cannot bound `run_bash`, because `python -c` walks straight
/// past anything that greps for `rm`. Where no sandbox exists (Windows), the
/// shell tools run unconfined by explicit choice, and the boot log says so —
/// a boundary that silently is not one is worse than a known-absent one,
/// because you stop watching for it.
enum Sandbox {
    /// bubblewrap: the workspace and tool dirs are the only project paths the
    /// process can see at all.
    Bwrap(PathBuf),
    /// No confinement available on this platform. Shell tools reach the whole
    /// filesystem, exactly as any program the user runs would.
    None,
}

/// Where bubblewrap is, if it is anywhere.
fn find_bwrap() -> Option<PathBuf> {
    if !cfg!(unix) {
        return None;
    }
    if let Ok(p) = std::env::var("RUSTY_LLAMA_BWRAP") {
        if !p.is_empty() && Path::new(&p).exists() {
            return Some(PathBuf::from(p));
        }
    }
    ["/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

/// bubblewrap where it exists, nothing where it does not.
fn resolve_sandbox() -> Sandbox {
    find_bwrap().map(Sandbox::Bwrap).unwrap_or(Sandbox::None)
}

impl ToolRuntime {
    /// A command that can read anything but write only to the tool directories.
    ///
    /// Binds are applied in order, so the read-only root is laid down first and
    /// the writable paths override it. `--die-with-parent` keeps a wedged tool
    /// from outliving the server.
    fn confined(&self, cwd: &Path, program: &str, args: &[String]) -> Result<Command, String> {
        let bwrap = match &self.sandbox {
            Sandbox::Bwrap(p) => p,
            Sandbox::None => {
                let mut c = Command::new(program);
                for a in args {
                    c.arg(a);
                }
                c.current_dir(cwd);
                return Ok(c);
            }
        };
        let mut c = Command::new(bwrap);
        // System paths read-only so an interpreter exists at all; -try because
        // /lib and /lib64 are symlinks into /usr on merged-usr distributions.
        for p in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"] {
            c.arg("--ro-bind-try").arg(p).arg(p);
        }
        c.arg("--proc").arg("/proc")
            .arg("--dev").arg("/dev")
            .arg("--tmpfs").arg("/tmp")
            // The only project paths that exist inside: the chosen workspace
            // and the tool directories. Everything else — home, other repos,
            // the rest of the disk — is simply not in the namespace, so there
            // is nothing to escape to rather than a rule to get around.
            .arg("--bind").arg(cwd).arg(cwd)
            .arg("--bind").arg(&self.py_dir).arg(&self.py_dir)
            .arg("--bind").arg(&self.bash_dir).arg(&self.bash_dir)
            .arg("--chdir").arg(cwd)
            .arg("--die-with-parent")
            .arg("--")
            .arg(program);
        for a in args {
            c.arg(a);
        }
        Ok(c)
    }
}

impl ToolRuntime {
    /// `data_dir` holds the runtime's own tool directories; `bash_override`
    /// is the `[tools] bash` config value (empty = autodetect).
    pub fn new(data_dir: &Path, bash_override: &str) -> std::io::Result<Self> {
        let py_dir = data_dir.join("py-tools");
        let bash_dir = data_dir.join("bash-tools");
        std::fs::create_dir_all(&py_dir)?;
        std::fs::create_dir_all(&bash_dir)?;
        // bwrap binds need absolute paths; these are built from "." at boot.
        let py_dir = std::fs::canonicalize(&py_dir).unwrap_or(py_dir);
        let bash_dir = std::fs::canonicalize(&bash_dir).unwrap_or(bash_dir);
        let bash_exe = resolve_bash(bash_override);
        let sandbox = resolve_sandbox();
        match &sandbox {
            Sandbox::Bwrap(p) => eprintln!(
                "[tools] sandbox: {} — shell tools see only the workspace and {} / {}",
                p.display(), py_dir.display(), bash_dir.display()
            ),
            Sandbox::None => eprintln!(
                "[tools] no sandbox on this platform — shell tools reach the whole \
                 filesystem, not just the workspace"
            ),
        }
        eprintln!("[tools] bash: {}", bash_exe.display());
        Ok(Self { py_dir, bash_dir, bash_exe, sandbox, fs: FsTools::new() })
    }

    /// The OpenAI `tools` array sent with every agentic request; llama-server's
    /// jinja template renders it into the model's native declaration format.
    ///
    /// Ordered deliberately: the filesystem tools come first because they are
    /// what the model should reach for, and a model scanning a long tool list
    /// picks from the top of it. `run_bash` sits below them and its description
    /// says what it is *for* — an escape hatch for the things nothing above
    /// covers (git, cargo, tests), not the way to read and write files.
    ///
    /// `fs_only` (the small hardware tiers) declares just the filesystem set:
    /// measured, the full block renders to ~1150 tokens of system prefix and
    /// fs-only to ~760 — a real fraction of an 8-16k window. The caller must
    /// refuse execution of undeclared tools too; declaration is not a boundary
    /// (though llama-server's grammar also cannot emit an undeclared call).
    /// `rag_available` additionally declares rag_search — on BOTH tiers: its
    /// declaration costs ~60 tokens and retrieval is precisely the cheap path
    /// a small window wants (one round instead of a grep-read chain). It is
    /// executed by the serving layer (it needs the vector store), not here.
    pub fn tool_specs(&self, fs_only: bool, rag_available: bool) -> serde_json::Value {
        let mut specs = self.full_tool_specs();
        let Some(arr) = specs.as_array_mut() else { return specs };
        if fs_only {
            arr.retain(|t| t["function"]["name"].as_str().is_some_and(is_fs_tool));
        }
        if rag_available {
            let decl = serde_json::json!({"type": "function", "function": {
                "name": "rag_search",
                "description": "Semantic search over the indexed workspace. Returns the most relevant chunks with their source file. Use when you know WHAT you want but not WHERE it is; grep_files is better once you know the name to look for.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "how many chunks to return (default 5, max 20). Start small and ask again with a larger limit only if the first results miss — every chunk you pull costs context."}
                }, "required": ["query"]}}});
            // After the grep/glob cluster: retrieval is the "know WHAT, not
            // WHERE" complement to grep's "know the name" — adjacency teaches
            // the model they are alternatives, not duplicates.
            let at = arr.iter()
                .position(|t| t["function"]["name"] == "list_dir")
                .unwrap_or(arr.len());
            arr.insert(at, decl);
        }
        specs
    }

    fn full_tool_specs(&self) -> serde_json::Value {
        serde_json::json!([
            {"type": "function", "function": {"name": "read_file", "description": "Read a text file. Returns lines prefixed with their line number. Use offset/limit to page through a long file rather than reading it whole.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer", "description": "first line to show, 1-based"}, "limit": {"type": "integer", "description": "how many lines"}}, "required": ["file_path"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file. old_string must match byte for byte including indentation, and must be unique unless replace_all is true. The write is atomic and read back to verify. This is the right way to change code — never sed or a shell redirect.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["file_path", "old_string", "new_string"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Create a NEW file, making parent directories as needed. It will not overwrite a file that already exists — use edit_file to change one.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
            {"type": "function", "function": {"name": "grep_files", "description": "Regular-expression search across the workspace. output_mode content (default) returns path:line:text, files_with_matches returns paths only, count returns per-file totals.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string", "description": "optional; only search files matching this, e.g. *.rs. Omit it unless you are sure of the extension — a glob that matches nothing searches nothing."}, "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]}, "-i": {"type": "boolean"}, "-C": {"type": "integer", "description": "context lines"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "glob_files", "description": "Find files by path pattern (** matches any depth), newest first. Use when you know the shape of the filename but not where it is.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "list_dir", "description": "List one directory with file types and sizes", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "all": {"type": "boolean", "description": "include dotfiles"}}, "required": []}}},
            {"type": "function", "function": {"name": "run_python", "description": "Execute Python code; stdout+stderr returned", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
            {"type": "function", "function": {"name": "run_bash", "description": "Execute a bash command for work the tools above do not cover — running tests, git, build commands. Do not use it to read, search or edit files.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "save_tool", "description": "Save a reusable tool script into py-tools/ (lang=python) or bash-tools/ (lang=bash) for later runs", "parameters": {"type": "object", "properties": {"lang": {"type": "string", "enum": ["python", "bash"]}, "name": {"type": "string"}, "code": {"type": "string"}}, "required": ["lang", "name", "code"]}}},
            {"type": "function", "function": {"name": "run_tool", "description": "Run a previously saved tool by name with arguments", "parameters": {"type": "object", "properties": {"lang": {"type": "string", "enum": ["python", "bash"]}, "name": {"type": "string"}, "args": {"type": "string"}}, "required": ["lang", "name"]}}}
        ])
    }

    /// Static system-prompt addendum for agentic turns. The declarations
    /// themselves travel in the `tools` array; this carries only the behavior
    /// the schemas cannot. Byte-stable per conversation, so it never disturbs
    /// the KV prefix. The fs-only variant must not mention saved tools —
    /// naming a capability that is not declared invites calls to it.
    pub fn system_addendum(&self, fs_only: bool) -> String {
        let base = "\n\nTools are available for this conversation. Paths are relative \
                    to the workspace and nothing outside it can be reached. Read a \
                    file before you edit it, and quote old_string from what read_file \
                    showed you.";
        if fs_only {
            base.to_string()
        } else {
            format!(
                "{base} Saved tools live in {} and {}.",
                self.py_dir.display(),
                self.bash_dir.display()
            )
        }
    }

    /// Execute one parsed tool call; always returns SOMETHING printable.
    /// `name`/`args` come from llama-server's parsed `tool_calls` (arguments
    /// already through `normalize_args`). `workspace` roots the fs tools and
    /// is the working directory for run_python/run_bash/run_tool, so relative
    /// paths hit the repo the user chose without the model having to `cd`.
    pub fn execute(&self, name: &str, args: &serde_json::Value, workspace: Option<&Path>) -> ToolResult {
        if FS_TOOL_NAMES.contains(&name) {
            // The filesystem tools are rooted at the workspace. Without one
            // they would be rooted at the server's own directory, where the
            // model would be editing this server's source while answering a
            // question about someone else's repo — refuse instead.
            let Some(root) = workspace else {
                return ToolResult::err(format!(
                    "error: {name} needs a workspace — none is selected. \
                     Set one in config.toml [tools] or the request."
                ));
            };
            return self.fs.execute(name, args, root);
        }
        match name {
            "run_python" => {
                let code = args["code"].as_str().unwrap_or("");
                if code.trim().is_empty() {
                    return ToolResult::err(
                        "error: run_python called with empty `code` — pass the Python source in the `code` argument".into(),
                    );
                }
                let path = self.py_dir.join("_scratch.py");
                if let Err(e) = std::fs::write(&path, code) {
                    return ToolResult::err(format!("error: {e}"));
                }
                let dir = workspace.unwrap_or(&self.py_dir).to_path_buf();
                match self.confined(&dir, python_exe(), &[path.display().to_string()]) {
                    Ok(mut c) => run_captured(&mut c),
                    Err(why) => ToolResult::err(format!("error: {why}")),
                }
            }
            "run_bash" => {
                let cmd = args["command"].as_str().unwrap_or("");
                if cmd.trim().is_empty() {
                    // bash -c "" exits 0 — an empty call must fail loudly, not
                    // report ok with no output.
                    return ToolResult::err(
                        "error: run_bash called with empty `command` — pass the shell command in the `command` argument".into(),
                    );
                }
                let dir = workspace.unwrap_or(&self.bash_dir).to_path_buf();
                let bash = self.bash_exe.display().to_string();
                match self.confined(&dir, &bash, &["-c".into(), cmd.to_string()]) {
                    Ok(mut c) => run_captured(&mut c),
                    Err(why) => ToolResult::err(format!("error: {why}")),
                }
            }
            "save_tool" => {
                let lang = args["lang"].as_str().unwrap_or("python");
                let name = sanitize(args["name"].as_str().unwrap_or(""));
                let code = args["code"].as_str().unwrap_or("");
                if name.is_empty() {
                    return ToolResult::err("error: tool name required".into());
                }
                let (dir, ext) = if lang == "bash" {
                    (&self.bash_dir, "sh")
                } else {
                    (&self.py_dir, "py")
                };
                let path = dir.join(format!("{name}.{ext}"));
                match std::fs::write(&path, code) {
                    Ok(()) => ToolResult::ok(format!("saved: {}", path.display())),
                    Err(e) => ToolResult::err(format!("error: {e}")),
                }
            }
            "run_tool" => {
                let lang = args["lang"].as_str().unwrap_or("python");
                let name = sanitize(args["name"].as_str().unwrap_or(""));
                let extra = args["args"].as_str().unwrap_or("");
                if lang == "bash" {
                    let path = self.bash_dir.join(format!("{name}.sh"));
                    // Forward slashes: bash -c strips backslashes from the
                    // unquoted-looking Windows path inside the command string.
                    let script = path.display().to_string().replace('\\', "/");
                    let dir = workspace.unwrap_or(&self.bash_dir).to_path_buf();
                    let bash = self.bash_exe.display().to_string();
                    match self.confined(&dir, &bash, &["-c".into(), format!("bash '{script}' {extra}")]) {
                        Ok(mut c) => run_captured(&mut c),
                        Err(why) => ToolResult::err(format!("error: {why}")),
                    }
                } else {
                    let path = self.py_dir.join(format!("{name}.py"));
                    let mut argv = vec![path.display().to_string()];
                    argv.extend(extra.split_whitespace().map(str::to_string));
                    let dir = workspace.unwrap_or(&self.py_dir).to_path_buf();
                    match self.confined(&dir, python_exe(), &argv) {
                        Ok(mut c) => run_captured(&mut c),
                        Err(why) => ToolResult::err(format!("error: {why}")),
                    }
                }
            }
            other => ToolResult::err(format!("error: unknown tool '{other}'")),
        }
    }
}

/// Arguments of a parsed tool call, whichever shape the model used.
///
/// llama-server hands us `function.arguments` as a JSON string; models still
/// vary the shape inside it — sometimes the args object directly, sometimes
/// wrapped under an `arguments`/`params` key, sometimes flattened beside a
/// repeated `name`. Callers pass the whole parsed call object here.
pub fn normalize_args(parsed: &serde_json::Value) -> serde_json::Value {
    let raw = match ["arguments", "parameters", "params", "args"]
        .iter()
        .find_map(|k| parsed.get(*k))
        .cloned()
    {
        Some(v) => v,
        // Flattened form: {"name": "run_bash", "command": "ls"} — the
        // parameters sit beside the name rather than under an "arguments"
        // key. Qwen-family models emit this regularly; treating it as "no
        // arguments" ran the tool with an empty command and reported a
        // bare `error` the model could not learn anything from.
        None => {
            let mut m = parsed.as_object().cloned().unwrap_or_default();
            m.remove("name");
            serde_json::Value::Object(m)
        }
    };
    match raw {
        serde_json::Value::String(s) => serde_json::from_str(&s).unwrap_or(serde_json::Value::Null),
        v => v,
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(64)
        .collect()
}

/// Run with captured stdout/stderr, a hard timeout, and output truncation.
fn run_captured(cmd: &mut Command) -> ToolResult {
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("error: spawn: {e}")),
    };
    // Drain pipes on threads so a full pipe can never deadlock the child.
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out_h = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let err_h = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(TIMEOUT_SECS);
    let mut exit: Option<std::process::ExitStatus> = None;
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(s)) => {
                exit = Some(s);
                break false;
            }
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break true;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break false,
        }
    };
    let mut text = String::new();
    let out = out_h.join().unwrap_or_default();
    let err = err_h.join().unwrap_or_default();
    text.push_str(&String::from_utf8_lossy(&out));
    if !err.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&err));
    }
    if timed_out {
        text.push_str(&format!("\n[killed after {TIMEOUT_SECS}s timeout]"));
    }
    if text.trim().is_empty() {
        text = "(no output)".into();
    }
    if text.len() > MAX_OUTPUT {
        let cut: String = text.chars().take(MAX_OUTPUT).collect();
        text = format!("{cut}\n[truncated]");
    }
    let (ok, status) = if timed_out {
        (false, "timeout".to_string())
    } else {
        match exit {
            Some(s) if s.success() => (true, "ok".to_string()),
            Some(s) => (false, format!("exit {}", s.code().unwrap_or(-1))),
            None => (false, "error".to_string()),
        }
    };
    ToolResult { ok, status, output: text }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Ported from streamer_server.rs: a wrapped `arguments` object must win
    // over keys flattened beside the name.
    #[test]
    fn wrapped_arguments_still_win_over_flattened_keys() {
        let call = json!({"name": "run_bash", "command": "WRONG",
                          "arguments": {"command": "ls"}});
        let args = normalize_args(&call);
        assert_eq!(args["command"].as_str(), Some("ls"));
    }

    #[test]
    fn flattened_args_are_lifted_and_name_removed() {
        let call = json!({"name": "run_bash", "command": "ls -la"});
        let args = normalize_args(&call);
        assert_eq!(args["command"].as_str(), Some("ls -la"));
        assert!(args.get("name").is_none());
    }

    #[test]
    fn stringified_arguments_blob_is_parsed() {
        let call = json!({"name": "read_file", "arguments": "{\"file_path\": \"src/main.rs\"}"});
        let args = normalize_args(&call);
        assert_eq!(args["file_path"].as_str(), Some("src/main.rs"));
    }

    #[test]
    fn sanitize_confines_tool_names_to_bare_filenames() {
        assert_eq!(sanitize("../../etc/passwd"), "etcpasswd");
        assert_eq!(sanitize("my_tool-2"), "my_tool-2");
    }

    #[test]
    fn slim_specs_declare_exactly_the_fs_set() {
        let rt = ToolRuntime::new(&std::env::temp_dir().join("rl_tools_test"), "").unwrap();
        let names = |v: &serde_json::Value| -> Vec<String> {
            v.as_array().unwrap().iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };
        // full_tool_specs holds every tool except rag_search (serving-layer).
        let full = names(&rt.tool_specs(false, false));
        assert_eq!(full.len(), TOOL_NAMES.len() - 1);
        let slim = names(&rt.tool_specs(true, false));
        assert_eq!(slim.len(), FS_TOOL_NAMES.len());
        assert!(slim.iter().all(|n| is_fs_tool(n)));
        // The fs-first ordering survives the filter.
        assert_eq!(slim[0], "read_file");
        // And the slim addendum never names undeclared capabilities.
        assert!(!rt.system_addendum(true).contains("Saved tools"));
    }

    #[test]
    fn rag_search_is_declared_on_both_tiers_next_to_grep() {
        let rt = ToolRuntime::new(&std::env::temp_dir().join("rl_tools_test"), "").unwrap();
        for fs_only in [false, true] {
            let specs = rt.tool_specs(fs_only, true);
            let arr = specs.as_array().unwrap();
            let pos = arr.iter()
                .position(|t| t["function"]["name"] == "rag_search")
                .expect("rag_search declared");
            // Sits with the search cluster, right after glob_files.
            assert_eq!(arr[pos - 1]["function"]["name"], "glob_files");
            let without = rt.tool_specs(fs_only, false);
            assert!(without.as_array().unwrap().iter()
                .all(|t| t["function"]["name"] != "rag_search"));
        }
    }
}
