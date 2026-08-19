use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};

/// A snapshot of Mimoid's own resource footprint: the app process, its
/// Python sidecar child (and that child's children, e.g. ffmpeg), plus
/// Ollama — summed together, not the whole machine.
#[derive(Clone, Debug, Serialize)]
pub struct SystemStats {
    /// Combined CPU usage of the footprint processes, as a percentage of the
    /// whole machine (0–100, normalized by logical core count).
    pub cpu_percent: f32,
    /// Combined resident memory (RSS) of the footprint processes, in bytes.
    pub mem_used_bytes: u64,
    /// Total physical memory on the machine, in bytes (for context).
    pub total_mem_bytes: u64,
    /// How many processes the footprint covered.
    pub process_count: usize,
}

/// Holds a persistent `System` so per-process CPU usage can be computed as a
/// delta between polls (sysinfo needs two refreshes spaced apart for that).
pub struct SystemMonitor {
    system: Mutex<System>,
    primed: AtomicBool,
}

impl SystemMonitor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            system: Mutex::new(System::new()),
            primed: AtomicBool::new(false),
        })
    }

    /// Refreshes process/memory info and returns Mimoid's footprint.
    /// Blocking (the first call sleeps briefly to establish a CPU baseline),
    /// so callers should run it off the async runtime.
    pub fn stats(&self) -> SystemStats {
        let mut sys = self.system.lock().expect("system monitor mutex poisoned");

        // The very first CPU reading needs a baseline refresh followed by a
        // short wait; after that the ~2s poll cadence provides the delta.
        if !self.primed.swap(true, Ordering::SeqCst) {
            sys.refresh_processes(ProcessesToUpdate::All, true);
            std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        }

        sys.refresh_processes(ProcessesToUpdate::All, true);
        sys.refresh_memory();

        let processes = sys.processes();
        let mut included: HashSet<Pid> = HashSet::new();

        // The app itself and everything it spawned (sidecar, ffmpeg, …).
        if let Ok(current) = sysinfo::get_current_pid() {
            included.insert(current);
            loop {
                let mut added = false;
                for (pid, process) in processes {
                    if included.contains(pid) {
                        continue;
                    }
                    if let Some(parent) = process.parent() {
                        if included.contains(&parent) {
                            included.insert(*pid);
                            added = true;
                        }
                    }
                }
                if !added {
                    break;
                }
            }
        }

        // Ollama typically runs as its own service outside our process tree,
        // so pull it in by name (covers `ollama serve` and `ollama runner`).
        for (pid, process) in processes {
            if process.name().to_string_lossy().to_lowercase().contains("ollama") {
                included.insert(*pid);
            }
        }

        let mut cpu = 0.0f32;
        let mut mem = 0u64;
        for pid in &included {
            if let Some(process) = processes.get(pid) {
                cpu += process.cpu_usage();
                mem += process.memory();
            }
        }

        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f32;

        SystemStats {
            cpu_percent: cpu / cores,
            mem_used_bytes: mem,
            total_mem_bytes: sys.total_memory(),
            process_count: included.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footprint_includes_current_process() {
        let monitor = SystemMonitor::new();
        let stats = monitor.stats();
        // The test binary is itself part of the footprint, so its resident
        // memory must show up even with no sidecar/Ollama running.
        assert!(stats.mem_used_bytes > 0, "expected current process RSS > 0");
        assert!(stats.process_count >= 1);
        assert!(stats.total_mem_bytes > 0);
        assert!(stats.cpu_percent >= 0.0);
    }
}
