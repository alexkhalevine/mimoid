use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;

const SIDECAR_HOST: &str = "127.0.0.1";
// Deliberately different from Symmetriad's (8756) -- these are two
// independently installed apps on the same codebase, and sharing a port
// meant whichever one launched second would kill and replace the other's
// sidecar process, silently pointing its frontend at the wrong backend's
// (separate, likely near-empty) database. See PLAN entry / PR description
// for the incident this was found from.
const SIDECAR_PORT: u16 = 8757;
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(3);
const MAX_CONSECUTIVE_RESTARTS: u32 = 5;

const BOOTSTRAP_MARKER: &str = ".bootstrap-complete";
const MIN_DISK_GB: f64 = 10.0;
const MIN_MEMORY_GB: f64 = 8.0;

pub const BOOTSTRAP_STATUS_EVENT: &str = "sidecar-bootstrap-status";

/// Progress reported to the frontend during first-run setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BootstrapStatus {
    Checking,
    InsufficientResources { message: String },
    Installing { line: String },
    Ready,
    Error { message: String },
}

/// Shared, pollable copy of the latest bootstrap status, so the frontend can
/// fetch the current state even if it starts listening after an event fired.
#[derive(Default)]
pub struct BootstrapState(pub Mutex<Option<BootstrapStatus>>);

fn emit_status(app: &AppHandle, state: &tauri::State<'_, BootstrapState>, status: BootstrapStatus) {
    let _ = app.emit(BOOTSTRAP_STATUS_EVENT, status.clone());
    if let Ok(mut guard) = state.0.try_lock() {
        *guard = Some(status);
    }
}

/// Disk/RAM preflight check. Returns `Err(message)` if the machine doesn't
/// meet the minimum requirements to run the local ML stack.
pub fn preflight(app: &AppHandle) -> Result<(), String> {
    use sysinfo::{Disks, System};

    let mut sys = System::new();
    sys.refresh_memory();
    let total_memory_gb = sys.total_memory() as f64 / 1_073_741_824.0;
    if total_memory_gb < MIN_MEMORY_GB {
        return Err(format!(
            "This machine has {total_memory_gb:.1} GB of RAM; Mimoid needs at least {MIN_MEMORY_GB:.0} GB to run the local LLM, speech, and voice models."
        ));
    }

    let target_dir = data_root(app);
    let disks = Disks::new_with_refreshed_list();
    let available_gb = disks
        .iter()
        .filter(|d| target_dir.starts_with(d.mount_point()))
        .map(|d| d.available_space() as f64 / 1_073_741_824.0)
        .fold(None::<f64>, |best, gb| match best {
            Some(b) if b >= gb => Some(b),
            _ => Some(gb),
        })
        .or_else(|| {
            disks
                .iter()
                .map(|d| d.available_space() as f64 / 1_073_741_824.0)
                .reduce(f64::max)
        });

    if let Some(available_gb) = available_gb {
        if available_gb < MIN_DISK_GB {
            return Err(format!(
                "Only {available_gb:.1} GB of free disk space is available; Mimoid needs at least {MIN_DISK_GB:.0} GB to download local models."
            ));
        }
    }

    Ok(())
}

fn data_root(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Where a venv's own interpreter/pip live -- Windows venvs use `Scripts\`
/// with `.exe` suffixes, everywhere else it's `bin/`.
fn venv_python_path(venv_dir: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn venv_pip_path(venv_dir: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    }
}

/// The system Python used to create the venv in the first place (before the
/// venv's own interpreter exists). Windows' python.org installer registers
/// `python`, not `python3`. Used only as a last-resort fallback (see
/// `python_bin`) and by the test helper below -- the real venv-creation
/// path uses `find_system_python`, which actually checks the version.
fn system_python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

/// Interpreter names to search for a suitable system Python, newest first.
/// Just trying `python3` isn't enough: macOS's Xcode Command Line Tools
/// ships a `python3` stub that can sit on an ancient version (3.9, with a
/// pip from 2020) for years after release, well below what this app's
/// pinned dependencies need -- `coqui-tts` alone declares `>=3.10`, and pip
/// then silently discards every version above what that floor allows
/// (`ERROR: Could not find a version that satisfies the requirement
/// coqui-tts==...`) with no indication that the real problem is the
/// interpreter, not the package. Homebrew and python.org installs register
/// version-suffixed binaries, so checking those first finds a working
/// interpreter on a machine that has one anywhere on PATH.
#[cfg(not(windows))]
const PYTHON_CANDIDATES: &[&str] = &["python3.13", "python3.12", "python3.11", "python3.10", "python3"];
#[cfg(windows)]
const PYTHON_CANDIDATES: &[&str] = &["python", "python3"];

/// Floor for the interpreter used to create the sidecar's venv. Matches
/// README's documented "Python 3.11+" prerequisite (a version newer than
/// `coqui-tts`'s own `>=3.10` floor, for headroom against the next pinned
/// dependency that raises its requirement).
const MIN_PYTHON: (u32, u32) = (3, 11);

/// Finds the newest `PYTHON_CANDIDATES` entry on PATH that meets
/// `MIN_PYTHON`, asking each one directly for its own version rather than
/// trusting the binary name -- a `python3.11` on PATH is real evidence,
/// but a bare `python3` could be anything.
async fn find_system_python() -> Result<String, String> {
    find_system_python_on(None).await
}

async fn find_system_python_on(path_override: Option<&std::ffi::OsStr>) -> Result<String, String> {
    let mut newest_found: Option<((u32, u32), String)> = None;
    for candidate in PYTHON_CANDIDATES {
        let Some(version) = python_version_on(candidate, path_override).await else {
            continue;
        };
        if version >= MIN_PYTHON {
            return Ok((*candidate).to_string());
        }
        let is_newer = match &newest_found {
            Some((best, _)) => version > *best,
            None => true,
        };
        if is_newer {
            newest_found = Some((version, (*candidate).to_string()));
        }
    }

    Err(match newest_found {
        Some((version, name)) => format!(
            "Mimoid needs Python {}.{}+ but only found {name} (Python {}.{}). {}",
            MIN_PYTHON.0, MIN_PYTHON.1, version.0, version.1, python_install_hint()
        ),
        None => format!(
            "Mimoid needs Python {}.{}+ but no Python interpreter was found on PATH. {}",
            MIN_PYTHON.0, MIN_PYTHON.1, python_install_hint()
        ),
    })
}

fn python_install_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install a newer one (e.g. `brew install python@3.11`), then restart Mimoid."
    } else if cfg!(windows) {
        "Install a newer one from python.org, then restart Mimoid."
    } else {
        "Install a newer one via your distro's package manager, then restart Mimoid."
    }
}

/// Runs `bin -c "..."` to ask it its own `major.minor` version. Returns
/// `None` for a binary that isn't on PATH, isn't actually Python, or
/// otherwise fails to run -- callers treat that the same as "not usable"
/// rather than a hard error, since probing several candidate names is the
/// normal case, not an exceptional one.
async fn python_version(bin: impl AsRef<std::ffi::OsStr>) -> Option<(u32, u32)> {
    python_version_on(bin, None).await
}

async fn python_version_on(
    bin: impl AsRef<std::ffi::OsStr>,
    path_override: Option<&std::ffi::OsStr>,
) -> Option<(u32, u32)> {
    let mut command = Command::new(bin);
    command.args(["-c", "import sys; print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")"]);
    if let Some(path) = path_override {
        command.env("PATH", path);
    }
    let output = command.output().await.ok()?;
    if !output.status.success() {
        return None;
    }
    parse_python_version(&String::from_utf8_lossy(&output.stdout))
}

fn parse_python_version(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Finds PIDs of processes listening on `port`.
///
/// Tries `lsof` first (always present on macOS), then `ss` from iproute2
/// (always present on a systemd Linux, where `lsof` is an optional package
/// a minimal Arch install won't have). Returns empty if neither is around,
/// which just means the stale-port cleanup below no-ops.
#[cfg(unix)]
async fn find_pid_on_port(port: u16) -> Vec<String> {
    if let Ok(output) = Command::new("lsof")
        .args(["-ti", &format!(":{port}")])
        .output()
        .await
    {
        if output.status.success() {
            let pids: Vec<String> = std::str::from_utf8(&output.stdout)
                .unwrap_or("")
                .split_whitespace()
                .map(String::from)
                .collect();
            if !pids.is_empty() {
                return pids;
            }
        }
    }

    find_pid_on_port_ss(port).await
}

/// `ss -lptnH "sport = :PORT"` prints one row per listening socket, with the
/// owning process in a trailing `users:(("uvicorn",pid=1234,fd=7))` field.
/// Only the pid= numbers are of interest.
#[cfg(unix)]
async fn find_pid_on_port_ss(port: u16) -> Vec<String> {
    let Ok(output) = Command::new("ss")
        .args(["-lptnH", &format!("sport = :{port}")])
        .output()
        .await
    else {
        return Vec::new();
    };

    parse_ss_pids(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(unix)]
fn parse_ss_pids(text: &str) -> Vec<String> {
    let mut pids: Vec<String> = text
        .split("pid=")
        .skip(1)
        .filter_map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            (!digits.is_empty()).then_some(digits)
        })
        .collect();
    // One process can hold both the IPv4 and IPv6 listener, and `ss` prints
    // a row for each -- killing the same pid twice is harmless but noisy.
    pids.sort();
    pids.dedup();
    pids
}

#[cfg(unix)]
async fn kill_pid(pid: &str) {
    let _ = Command::new("kill").args(["-9", pid]).status().await;
}

#[cfg(windows)]
async fn find_pid_on_port(port: u16) -> Vec<String> {
    let Ok(output) = Command::new("netstat").args(["-ano"]).output().await else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let needle = format!(":{port} ");
    text.lines()
        .filter(|line| line.contains(&needle) && line.contains("LISTENING"))
        .filter_map(|line| line.split_whitespace().last())
        .map(String::from)
        .collect()
}

#[cfg(windows)]
async fn kill_pid(pid: &str) {
    let _ = Command::new("taskkill").args(["/F", "/PID", pid]).status().await;
}

/// Supervises the Python sidecar process: spawns it, restarts it if it
/// crashes (up to a limit, to avoid a hot restart loop), and kills it on
/// app shutdown.
pub struct SidecarManager {
    child: Mutex<Option<Child>>,
}

impl SidecarManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            child: Mutex::new(None),
        })
    }

    pub fn health_url() -> String {
        format!("http://{SIDECAR_HOST}:{SIDECAR_PORT}/health")
    }

    /// Where the sidecar's Python source lives. In dev, that's `sidecar/`
    /// next to `src-tauri/` in the repo. In a packaged app it's bundled as
    /// a Tauri resource (read-only) and resolved at runtime.
    fn source_dir(app: &AppHandle) -> PathBuf {
        if cfg!(debug_assertions) {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("sidecar")
        } else {
            app.path()
                .resource_dir()
                .expect("resource dir must resolve in a packaged app")
                .join("sidecar")
        }
    }

    /// Where the sidecar's venv lives. In dev it's alongside the source; in
    /// a packaged app the source dir is read-only, so the venv is created
    /// in the app's writable data dir instead.
    fn venv_dir(app: &AppHandle) -> PathBuf {
        if cfg!(debug_assertions) {
            Self::source_dir(app).join(".venv")
        } else {
            data_root(app).join("sidecar-venv")
        }
    }

    fn bootstrap_marker(app: &AppHandle) -> PathBuf {
        Self::venv_dir(app).join(BOOTSTRAP_MARKER)
    }

    fn python_bin(app: &AppHandle) -> PathBuf {
        let venv_python = venv_python_path(&Self::venv_dir(app));
        if venv_python.exists() {
            venv_python
        } else {
            PathBuf::from(system_python())
        }
    }

    /// Creates the venv (if missing) and installs `requirements.txt` into
    /// it, streaming progress as `BootstrapStatus::Installing` events. A
    /// no-op if a previous run already completed successfully (marked by
    /// `BOOTSTRAP_MARKER` inside the venv).
    pub async fn bootstrap(app: &AppHandle, state: &tauri::State<'_, BootstrapState>) -> Result<(), String> {
        emit_status(app, state, BootstrapStatus::Checking);

        if let Err(message) = preflight(app) {
            emit_status(app, state, BootstrapStatus::InsufficientResources { message: message.clone() });
            return Err(message);
        }

        let marker = Self::bootstrap_marker(app);
        if marker.exists() {
            emit_status(app, state, BootstrapStatus::Ready);
            return Ok(());
        }

        let venv_dir = Self::venv_dir(app);
        let source_dir = Self::source_dir(app);

        let venv_python = venv_python_path(&venv_dir);
        let venv_python_ok = if venv_python.exists() {
            python_version(&venv_python).await.is_some_and(|v| v >= MIN_PYTHON)
        } else {
            false
        };

        if !venv_python_ok {
            if venv_python.exists() {
                // A venv exists from a previous run, but its interpreter is
                // below MIN_PYTHON -- almost always because `system_python`
                // used to just take whatever `python3` resolved to, with no
                // version check. That venv's pip can never actually satisfy
                // `requirements.txt` (see PYTHON_CANDIDATES's comment), so
                // rebuilding it is the fix, not something to retry.
                emit_status(
                    app,
                    state,
                    BootstrapStatus::Installing { line: "Rebuilding Python environment with a newer interpreter…".into() },
                );
                let _ = tokio::fs::remove_dir_all(&venv_dir).await;
            }

            emit_status(app, state, BootstrapStatus::Installing { line: "Looking for a Python interpreter…".into() });
            let python = find_system_python().await.map_err(|message| bootstrap_fail(app, state, message))?;

            emit_status(app, state, BootstrapStatus::Installing { line: "Creating Python environment…".into() });
            run_streamed(app, state, Command::new(&python).args(["-m", "venv", venv_dir.to_string_lossy().as_ref()]))
                .await
                .map_err(|e| bootstrap_fail(app, state, format!("Failed to create Python environment: {e}")))?;
        }

        emit_status(app, state, BootstrapStatus::Installing { line: "Installing sidecar dependencies…".into() });
        let pip = venv_pip_path(&venv_dir);
        let requirements = source_dir.join("requirements.txt");
        run_streamed(
            app,
            state,
            Command::new(pip).args([
                "install",
                "-r",
                requirements.to_string_lossy().as_ref(),
            ]),
        )
        .await
        .map_err(|e| bootstrap_fail(app, state, format!("Failed to install sidecar dependencies: {e}")))?;

        if let Some(parent) = marker.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::write(&marker, b"ok").await;

        emit_status(app, state, BootstrapStatus::Ready);
        Ok(())
    }

    /// If a leftover process (e.g. from a crashed previous run, or a dev
    /// session that wasn't cleanly stopped) is already bound to our fixed
    /// port, every spawn attempt below fails with "address already in use"
    /// the same way, and `supervise()` burns through all its restart
    /// attempts in seconds with no clear signal to the user (the frontend
    /// just sees "sidecar not reachable" forever). Since 8757 is a port
    /// specific to this app, anything already listening on it at startup is
    /// overwhelmingly likely to be a stale Mimoid sidecar rather than an
    /// unrelated program, so it's safe to clear before our first spawn.
    async fn kill_stale_port_occupant() {
        let pids = find_pid_on_port(SIDECAR_PORT).await;
        if pids.is_empty() {
            return;
        }

        for pid in &pids {
            eprintln!("[sidecar] port {SIDECAR_PORT} already in use by pid {pid}, killing stale process");
            kill_pid(pid).await;
        }
        // Give the OS a moment to actually release the socket before we try
        // to bind it again.
        sleep(Duration::from_millis(300)).await;
    }

    async fn spawn(&self, app: &AppHandle) -> std::io::Result<()> {
        let mut command = Command::new(Self::python_bin(app));
        command
            .args([
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                SIDECAR_HOST,
                "--port",
                &SIDECAR_PORT.to_string(),
            ])
            .current_dir(Self::source_dir(app))
            .kill_on_drop(true);

        // In dev, config.py's own default (a `data/` folder next to the
        // sidecar source) is exactly where the rest of this file's `if
        // cfg!(debug_assertions)` branches already put things, and it's
        // handy to have visible in the repo tree -- left alone.
        //
        // In a packaged app that default instead resolves *inside the app
        // bundle itself* (source_dir() above, in its own words: "read-only"
        // in this branch), because config.py can't tell dev and packaged
        // apps apart the way this function can. Every local memory a user
        // has saved lives wherever this points, so pointing it at the
        // bundle isn't cosmetic: a reinstall or update replaces the whole
        // bundle from the fresh build, silently taking the database with
        // it. Redirecting it here, to the same writable per-app directory
        // the venv above already uses, is what config.py's own migration
        // (see its MIMOID_DATA_DIR handling) is for -- it moves anything a
        // pre-fix version already wrote into the old spot across, once.
        if !cfg!(debug_assertions) {
            command.env("MIMOID_DATA_DIR", data_root(app).join("data"));
        }

        let child = command.spawn()?;

        *self.child.lock().await = Some(child);
        Ok(())
    }

    /// Spawns the sidecar and starts the background supervisor task.
    pub async fn start(self: &Arc<Self>, app: &AppHandle) {
        Self::kill_stale_port_occupant().await;

        if let Err(err) = self.spawn(app).await {
            eprintln!("[sidecar] failed to spawn: {err}");
        }

        let manager = Arc::clone(self);
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            manager.supervise(app).await;
        });
    }

    async fn supervise(self: Arc<Self>, app: AppHandle) {
        let mut consecutive_restarts = 0u32;
        loop {
            sleep(HEALTH_CHECK_INTERVAL).await;

            let exited = {
                let mut guard = self.child.lock().await;
                match guard.as_mut() {
                    Some(child) => matches!(child.try_wait(), Ok(Some(_))),
                    None => true,
                }
            };

            if !exited {
                consecutive_restarts = 0;
                continue;
            }

            if consecutive_restarts >= MAX_CONSECUTIVE_RESTARTS {
                eprintln!("[sidecar] exceeded max restart attempts, giving up");
                break;
            }

            consecutive_restarts += 1;
            eprintln!(
                "[sidecar] process exited, restarting ({consecutive_restarts}/{MAX_CONSECUTIVE_RESTARTS})"
            );
            if let Err(err) = self.spawn(&app).await {
                eprintln!("[sidecar] restart failed: {err}");
            }
        }
    }

    /// Kills the sidecar process. Called on app quit.
    pub async fn shutdown(&self) {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}

fn bootstrap_fail(app: &AppHandle, state: &tauri::State<'_, BootstrapState>, message: String) -> String {
    emit_status(app, state, BootstrapStatus::Error { message: message.clone() });
    message
}

/// Runs a command to completion, streaming each line of its combined
/// stdout/stderr as an `Installing` progress event.
async fn run_streamed(
    app: &AppHandle,
    state: &tauri::State<'_, BootstrapState>,
    command: &mut Command,
) -> Result<(), String> {
    let mut child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Read both streams concurrently so a full stderr (or stdout) pipe
    // buffer can't stall the other and deadlock the child process.
    let read_stdout = async {
        if let Some(stdout) = stdout {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                emit_status(app, state, BootstrapStatus::Installing { line });
            }
        }
    };
    let read_stderr = async {
        if let Some(stderr) = stderr {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                emit_status(app, state, BootstrapStatus::Installing { line });
            }
        }
    };
    tokio::join!(read_stdout, read_stderr);

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command exited with status {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn parse_ss_pids_reads_pids_out_of_ss_output() {
        // Real `ss -lptnH "sport = :8756"` output: a process holding both the
        // IPv4 and IPv6 listener, which ss reports as two rows.
        let text = "LISTEN 0 2048 0.0.0.0:8756 0.0.0.0:* users:((\"python3\",pid=4242,fd=7))\n\
                    LISTEN 0 2048    [::]:8756    [::]:* users:((\"python3\",pid=4242,fd=8))\n";
        assert_eq!(parse_ss_pids(text), vec!["4242".to_string()]);
    }

    #[cfg(unix)]
    #[test]
    fn parse_ss_pids_handles_several_holders_and_no_holders() {
        let two = "users:((\"a\",pid=12,fd=7))\nusers:((\"b\",pid=345,fd=7))\n";
        assert_eq!(parse_ss_pids(two), vec!["12".to_string(), "345".to_string()]);

        // Header-only / empty output, and a row with no process info at all
        // (what ss prints without the privileges to attribute the socket).
        assert!(parse_ss_pids("").is_empty());
        assert!(parse_ss_pids("LISTEN 0 2048 0.0.0.0:8756 0.0.0.0:*\n").is_empty());
    }

    #[test]
    fn parse_python_version_reads_major_minor() {
        assert_eq!(parse_python_version("3.11\n"), Some((3, 11)));
        assert_eq!(parse_python_version("3.9"), Some((3, 9)));
        assert_eq!(parse_python_version("3.13"), Some((3, 13)));
        assert_eq!(parse_python_version(""), None);
        assert_eq!(parse_python_version("not-a-version"), None);
    }

    #[tokio::test]
    async fn python_version_reports_whatever_python3_is_on_this_machine() {
        // A smoke test against the real PATH -- every dev machine and CI
        // runner this crate builds on has *some* python3 (the sidecar venv
        // creation depends on it existing at all), so this should never be
        // None in practice. Doesn't assert a specific version since that's
        // exactly the thing this module can't assume.
        let version = python_version("python3").await;
        assert!(version.is_some(), "expected a python3 on PATH");
        assert_eq!(version.unwrap().0, 3);
    }

    #[tokio::test]
    async fn python_version_returns_none_for_a_binary_that_does_not_exist() {
        assert_eq!(python_version("definitely-not-a-real-interpreter-xyz").await, None);
    }

    #[cfg(unix)]
    /// Writes an executable shell shim named `name` into `dir` that prints
    /// `version` and exits 0 -- stands in for a real Python interpreter so
    /// `find_system_python_on`'s PATH search can be tested deterministically,
    /// independent of whatever interpreters happen to be installed on the
    /// machine actually running this test.
    fn write_python_shim(dir: &std::path::Path, name: &str, version: &str) {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\necho {version}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn find_system_python_picks_the_first_candidate_that_meets_min_python() {
        let dir = tempdir();
        // An old bare `python3` (the Xcode Command Line Tools scenario) sits
        // alongside a properly versioned `python3.11` -- PYTHON_CANDIDATES
        // must prefer the versioned, sufficient one.
        write_python_shim(dir.path(), "python3", "3.9");
        write_python_shim(dir.path(), "python3.11", "3.11");

        let found = find_system_python_on(Some(dir.path().as_os_str())).await;
        assert_eq!(found, Ok("python3.11".to_string()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn find_system_python_reports_the_newest_too_old_interpreter_it_found() {
        // The exact shape of the bug this guards: only an old `python3` is
        // on PATH (e.g. macOS's Xcode Command Line Tools stub), same as
        // what produced "Could not find a version that satisfies the
        // requirement coqui-tts==0.27.5" with no indication the real
        // problem was the interpreter. The error must name what was found
        // and how to fix it, not just fail silently into a bad venv.
        let dir = tempdir();
        write_python_shim(dir.path(), "python3", "3.9");

        let err = find_system_python_on(Some(dir.path().as_os_str())).await.unwrap_err();
        assert!(err.contains("3.9"), "error should mention the found version: {err}");
        assert!(err.contains("python3"), "error should name the found interpreter: {err}");
        assert!(err.contains("3.11"), "error should mention the required floor: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn find_system_python_reports_when_nothing_is_found() {
        let dir = tempdir();
        let err = find_system_python_on(Some(dir.path().as_os_str())).await.unwrap_err();
        assert!(err.contains("no Python interpreter was found"), "got: {err}");
    }

    /// Bare-bones temp-dir helper (no external crate) -- removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[cfg(unix)]
    fn tempdir() -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "mimoid-sidecar-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    #[tokio::test]
    async fn kill_stale_port_occupant_frees_the_port() {
        // Simulate a leftover process bound to our port, as if left behind
        // by a crashed previous run -- a *separate* process, not something
        // in this test's own PID, since the fix has to find and kill an
        // unrelated process holding the port.
        let mut holder = std::process::Command::new(system_python())
            .args([
                "-c",
                &format!(
                    "import socket, time; s = socket.socket(); \
                     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); \
                     s.bind(('127.0.0.1', {SIDECAR_PORT})); s.listen(1); time.sleep(30)"
                ),
            ])
            .spawn()
            .expect("failed to spawn port holder");

        // Give it a moment to actually bind before we check.
        for _ in 0..20 {
            if std::net::TcpListener::bind(("127.0.0.1", SIDECAR_PORT)).is_err() {
                break;
            }
            sleep(Duration::from_millis(50)).await;
        }
        assert!(
            std::net::TcpListener::bind(("127.0.0.1", SIDECAR_PORT)).is_err(),
            "expected the port to be occupied by the holder process"
        );

        SidecarManager::kill_stale_port_occupant().await;

        let listener = std::net::TcpListener::bind(("127.0.0.1", SIDECAR_PORT))
            .expect("port should be free again after killing the stale occupant");
        drop(listener);

        let _ = holder.try_wait(); // reap if it's already dead, to avoid a zombie
        let _ = holder.kill(); // belt-and-braces in case our own kill somehow missed it
    }
}
