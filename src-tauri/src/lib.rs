mod monitor;
mod net;
mod ollama;
mod sidecar;

use std::sync::Arc;

use monitor::{SystemMonitor, SystemStats};
use ollama::OllamaManager;
use sidecar::{BootstrapState, BootstrapStatus, SidecarManager};
use tauri::Manager;

#[tauri::command]
async fn sidecar_status() -> bool {
    net::check_health(&SidecarManager::health_url()).await
}

#[tauri::command]
async fn get_bootstrap_status(state: tauri::State<'_, BootstrapState>) -> Result<Option<BootstrapStatus>, String> {
    Ok(state.0.lock().await.clone())
}

#[tauri::command]
async fn ollama_status() -> bool {
    net::check_health(ollama::HEALTH_URL).await
}

#[tauri::command]
async fn ollama_installed() -> bool {
    OllamaManager::is_installed().await
}

#[tauri::command]
async fn start_ollama(manager: tauri::State<'_, Arc<OllamaManager>>) -> Result<(), String> {
    manager.start().await
}

#[tauri::command]
async fn stop_ollama(manager: tauri::State<'_, Arc<OllamaManager>>) -> Result<(), String> {
    manager.stop().await;
    Ok(())
}

#[tauri::command]
async fn system_stats(monitor: tauri::State<'_, Arc<SystemMonitor>>) -> Result<SystemStats, String> {
    let monitor = monitor.inner().clone();
    tauri::async_runtime::spawn_blocking(move || monitor.stats())
        .await
        .map_err(|err| err.to_string())
}

/// Writes bytes to an absolute path the user picked via a native save dialog
/// (`@tauri-apps/plugin-dialog`'s `save()`). Plain std::fs, not the fs
/// plugin's JS API -- the path already came from a user-driven picker, so it
/// doesn't need to be pre-declared in the fs scope.
#[tauri::command]
async fn write_file_bytes(path: String, contents: Vec<u8>) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || std::fs::write(&path, contents))
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sidecar_manager = SidecarManager::new();
    let ollama_manager = OllamaManager::new();
    let system_monitor = SystemMonitor::new();

    let startup_sidecar = sidecar_manager.clone();
    let shutdown_sidecar = sidecar_manager.clone();
    let shutdown_ollama = ollama_manager.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(sidecar_manager)
        .manage(ollama_manager)
        .manage(system_monitor)
        .manage(BootstrapState::default())
        .invoke_handler(tauri::generate_handler![
            sidecar_status,
            ollama_status,
            ollama_installed,
            start_ollama,
            stop_ollama,
            get_bootstrap_status,
            system_stats,
            write_file_bytes,
        ])
        .setup(move |app| {
            let manager = startup_sidecar.clone();
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<BootstrapState>();
                if let Err(err) = SidecarManager::bootstrap(&handle, &state).await {
                    eprintln!("[sidecar] bootstrap failed: {err}");
                    return;
                }
                manager.start(&handle).await;
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            let sidecar = shutdown_sidecar.clone();
            let ollama = shutdown_ollama.clone();
            tauri::async_runtime::block_on(async move {
                sidecar.shutdown().await;
                ollama.stop().await;
            });
        }
    });
}
