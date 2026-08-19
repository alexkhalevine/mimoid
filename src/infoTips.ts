// Click-to-expand help text for the ⓘ buttons scattered across Train mode.
// These used to be a plain span with a `title` attribute (a native hover
// tooltip) -- unreliable in Tauri's WebView, so every hint is now a real
// <button aria-controls="..."> that toggles a sibling <p class="info-tip-text">
// on click instead, which also gets keyboard (Enter/Space) support for free.
export function initInfoTips(): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>(".info-tip");
  for (const button of buttons) {
    const targetId = button.getAttribute("aria-controls");
    const target = targetId ? document.getElementById(targetId) : null;
    if (!target) continue;

    button.addEventListener("click", () => {
      const expanded = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!expanded));
      target.hidden = expanded;
    });
  }
}
