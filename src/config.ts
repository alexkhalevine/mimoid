// Deliberately different from Symmetriad's (8756) -- see src-tauri's
// sidecar.rs for why sharing a port between these two independently
// installed sibling apps was a real, confirmed bug.
export const SIDECAR_BASE_URL = "http://127.0.0.1:8757";
