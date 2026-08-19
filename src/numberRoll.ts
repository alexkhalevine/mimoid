const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Animates `el`'s text from `from` to `to` over `ms` (default 400ms, the
 * spec's number-roll token), appending `suffix` (e.g. "%"). Falls back to an
 * instant swap under prefers-reduced-motion. */
export function rollNumber(el: HTMLElement, from: number, to: number, ms = 400, suffix = ""): void {
  if (from === to || reducedMotion()) {
    el.textContent = `${to}${suffix}`;
    return;
  }
  const startedAt = performance.now();
  const step = (now: number) => {
    const progress = Math.min(1, (now - startedAt) / ms);
    const value = Math.round(from + (to - from) * progress);
    el.textContent = `${value}${suffix}`;
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
