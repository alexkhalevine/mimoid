#[cfg(not(target_os = "linux"))]
use std::process::Stdio;
use std::sync::Arc;

use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::net::check_health;

pub const HEALTH_URL: &str = "http://127.0.0.1:11434/api/version";

/// Manages a local `ollama serve` process. Only takes ownership of (and
/// later kills) the process if this app is the one that started it -- if
/// Ollama is already running (e.g. as a pre-existing system service), start()
/// no-ops and shutdown() leaves it alone.
///
/// On Linux this manager never owns a process at all: systemd does. start()
/// there is a detect-and-report probe rather than a spawn (see its comment).
pub struct OllamaManager {
    child: Mutex<Option<Child>>,
}

impl OllamaManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            child: Mutex::new(None),
        })
    }

    pub async fn is_installed() -> bool {
        Command::new("ollama")
            .arg("--version")
            .output()
            .await
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    pub async fn start(&self) -> Result<(), String> {
        if check_health(HEALTH_URL).await {
            return Ok(());
        }

        // On Linux, Ollama installs as a systemd service that owns the
        // process, its `ollama` user, and the model directory under
        // /usr/share/ollama. Spawning our own `ollama serve` as a child
        // would either lose the race to bind 11434 against a unit that's
        // merely slow to start, or -- worse -- win it and run a second
        // server as the desktop user, which then can't see any model the
        // service already pulled. Neither is worth the convenience, so on
        // Linux we detect and hand the user the one-liner instead.
        #[cfg(target_os = "linux")]
        {
            return Err(if Self::is_installed().await {
                "Ollama is installed but not running. Start it with \
                 `systemctl --user start ollama`, or `sudo systemctl start ollama` \
                 if you installed it system-wide."
                    .to_string()
            } else {
                "Ollama isn't installed. On Arch: `sudo pacman -S ollama` \
                 (or `ollama-cuda` for NVIDIA GPU acceleration), then \
                 `sudo systemctl enable --now ollama`."
                    .to_string()
            });
        }

        #[cfg(not(target_os = "linux"))]
        {
            let child = Command::new("ollama")
                .arg("serve")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .map_err(|err| err.to_string())?;

            *self.child.lock().await = Some(child);
            Ok(())
        }
    }

    /// Kills the ollama process, but only if this manager spawned it.
    pub async fn stop(&self) {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}
