import { invoke } from "@tauri-apps/api/core";

export function ollamaInstalled(): Promise<boolean> {
  return invoke<boolean>("ollama_installed");
}

export function ollamaRunning(): Promise<boolean> {
  return invoke<boolean>("ollama_status");
}

export function startOllama(): Promise<void> {
  return invoke<void>("start_ollama");
}

export function stopOllama(): Promise<void> {
  return invoke<void>("stop_ollama");
}
