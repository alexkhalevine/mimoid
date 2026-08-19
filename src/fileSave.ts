import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";

/**
 * Prompts the user with a native "Save As" dialog and writes the blob's
 * bytes to the chosen path via a Rust command (plain std::fs::write,
 * unrestricted by the fs plugin's scope -- the path already came from a
 * user-driven native picker). Returns false if the user cancelled.
 */
export async function saveBlobAs(
  blob: Blob,
  defaultFileName: string,
  extensions: string[],
): Promise<boolean> {
  const path = await save({
    defaultPath: defaultFileName,
    filters: [{ name: extensions.join("/").toUpperCase(), extensions }],
  });
  if (!path) return false;

  const contents = Array.from(new Uint8Array(await blob.arrayBuffer()));
  await invoke("write_file_bytes", { path, contents });
  return true;
}
