import architectureSource from "../docs/architecture.mmd?raw";

// mermaid is a sizeable library and only needed on this one tab, so it's
// imported dynamically here rather than at module load time -- keeps it in
// its own chunk instead of bloating the app's initial parse.
async function loadMermaid() {
  const { default: mermaid } = await import("mermaid");
  // Matches the app's dark design tokens (styles.css :root) so the diagram
  // reads as part of the UI rather than a pasted-in light-mode graphic.
  // Per-node colors come from the classDefs in architecture.mmd itself;
  // these variables only cover what those classDefs don't set (subgraph
  // cluster boxes, edge lines/labels, and the diagram's own font).
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    themeVariables: {
      background: "transparent",
      fontFamily: "Space Grotesk, system-ui, sans-serif",
      fontSize: "14px",
      primaryColor: "#141826",
      primaryTextColor: "#dce3e9",
      primaryBorderColor: "#232a3a",
      lineColor: "#4a5570",
      textColor: "#dce3e9",
      clusterBkg: "#0c0e13",
      clusterBorder: "#232a3a",
      edgeLabelBackground: "#0c0e13",
    },
  });
  return mermaid;
}

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 3;
const ZOOM_STEP = 0.25;
const ZOOM_DEFAULT = 1;

// The diagram's own natural pixel size at zoom 1 -- read once from its
// viewBox right after render (see initArchitecture), which for a
// mermaid-emitted SVG always matches the pixel dimensions it laid itself
// out at. 0 until a successful render has happened.
let baseWidth = 0;
let baseHeight = 0;
let zoom = ZOOM_DEFAULT;

function applyZoom(svgEl: SVGSVGElement, zoomLevelEl: HTMLElement, zoomOutBtn: HTMLButtonElement, zoomInBtn: HTMLButtonElement): void {
  if (!baseWidth || !baseHeight) return;
  svgEl.style.width = `${baseWidth * zoom}px`;
  svgEl.style.height = `${baseHeight * zoom}px`;
  zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
  zoomOutBtn.disabled = zoom <= ZOOM_MIN;
  zoomInBtn.disabled = zoom >= ZOOM_MAX;
}

export async function initArchitecture(): Promise<void> {
  const container = document.querySelector<HTMLElement>("#architecture-diagram");
  const errorEl = document.querySelector<HTMLElement>("#architecture-error");
  const toolbar = document.querySelector<HTMLElement>("#architecture-toolbar");
  const zoomOutBtn = document.querySelector<HTMLButtonElement>("#architecture-zoom-out");
  const zoomInBtn = document.querySelector<HTMLButtonElement>("#architecture-zoom-in");
  const zoomResetBtn = document.querySelector<HTMLButtonElement>("#architecture-zoom-reset");
  const zoomLevelEl = document.querySelector<HTMLElement>("#architecture-zoom-level");
  if (!container || !errorEl || !toolbar || !zoomOutBtn || !zoomInBtn || !zoomResetBtn || !zoomLevelEl) return;

  try {
    const mermaid = await loadMermaid();
    const { svg } = await mermaid.render("architecture-diagram-svg", architectureSource.trim());
    container.innerHTML = svg;
    // mermaid.render() emits width="100%" plus an inline `max-width` sized to
    // the diagram's natural layout, and a viewBox matching that same size --
    // that viewBox is the one reliable source for "how big is this at 1x",
    // since the width/height attributes themselves are just "100%"/absent.
    // Both get stripped below: sizing moves entirely to the explicit
    // width/height applyZoom() sets per zoom level, so the diagram can
    // render bigger than its container and the container's own
    // `overflow: auto` (styles.css) picks up real horizontal + vertical
    // scrollbars instead of the old shrink-to-fit behavior.
    const svgEl = container.querySelector("svg");
    if (svgEl) {
      baseWidth = svgEl.viewBox.baseVal.width;
      baseHeight = svgEl.viewBox.baseVal.height;
      svgEl.removeAttribute("width");
      svgEl.removeAttribute("height");
      svgEl.removeAttribute("style");
      zoom = ZOOM_DEFAULT;
      applyZoom(svgEl, zoomLevelEl, zoomOutBtn, zoomInBtn);
      toolbar.hidden = false;
    }
  } catch (err) {
    console.error("[architecture] failed to render diagram", err);
    errorEl.textContent = "Couldn't render the architecture diagram.";
    errorEl.hidden = false;
    return;
  }

  const rezoom = (next: number) => {
    const svgEl = container.querySelector<SVGSVGElement>("svg");
    if (!svgEl) return;
    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
    applyZoom(svgEl, zoomLevelEl, zoomOutBtn, zoomInBtn);
  };
  zoomOutBtn.addEventListener("click", () => rezoom(zoom - ZOOM_STEP));
  zoomInBtn.addEventListener("click", () => rezoom(zoom + ZOOM_STEP));
  zoomResetBtn.addEventListener("click", () => rezoom(ZOOM_DEFAULT));
}
