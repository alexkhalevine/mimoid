import { openGoogleCalendarModal } from "./googleCalendarModal";
import {
  getToolSettings,
  updateToolSettings,
  type ToolSettings,
  type WeatherLocation,
} from "./sidecar";

/** The Tools tab: which tools the twin may reach for, and their config.
 *
 * Tools are the one place the twin answers from live data instead of from
 * what it was trained on or remembers, so this tab is deliberately explicit
 * about which ones are on and whether each one leaves the machine. */
export function initTools(): void {
  const view = document.querySelector<HTMLElement>("#tools-view");
  if (!view) return;

  const clockEnabled = view.querySelector<HTMLInputElement>("#tools-clock-enabled");
  const timezone = view.querySelector<HTMLInputElement>("#tools-timezone");
  const timezoneSave = view.querySelector<HTMLButtonElement>("#tools-timezone-save");
  const preview = view.querySelector<HTMLElement>("#tools-clock-preview");
  const warning = view.querySelector<HTMLElement>("#tools-model-warning");
  const warningDetail = view.querySelector<HTMLElement>("#tools-model-warning-detail");
  const error = view.querySelector<HTMLElement>("#tools-error");
  const weatherEnabled = view.querySelector<HTMLInputElement>("#tools-weather-enabled");
  const weatherLocationsEl = view.querySelector<HTMLDivElement>("#tools-weather-locations");
  const weatherAdd = view.querySelector<HTMLButtonElement>("#tools-weather-add");
  const weatherSave = view.querySelector<HTMLButtonElement>("#tools-weather-save");
  const weatherError = view.querySelector<HTMLElement>("#tools-weather-error");
  const calendarEnabled = view.querySelector<HTMLInputElement>("#tools-calendar-enabled");
  const calendarConfirm = view.querySelector<HTMLInputElement>("#tools-calendar-confirm");
  const calendarStatus = view.querySelector<HTMLElement>("#tools-calendar-status");
  const calendarManage = view.querySelector<HTMLButtonElement>("#tools-calendar-manage");
  const calendarError = view.querySelector<HTMLElement>("#tools-calendar-error");
  if (
    !clockEnabled ||
    !timezone ||
    !timezoneSave ||
    !weatherEnabled ||
    !weatherLocationsEl ||
    !weatherAdd ||
    !weatherSave ||
    !calendarEnabled ||
    !calendarConfirm ||
    !calendarStatus ||
    !calendarManage
  )
    return;

  const showError = (message: string | null) => {
    if (!error) return;
    error.hidden = message === null;
    error.textContent = message ?? "";
  };

  const showWeatherError = (message: string | null) => {
    if (!weatherError) return;
    weatherError.hidden = message === null;
    weatherError.textContent = message ?? "";
  };

  const showCalendarError = (message: string | null) => {
    if (!calendarError) return;
    calendarError.hidden = message === null;
    calendarError.textContent = message ?? "";
  };

  const renderPreview = (settings: ToolSettings) => {
    if (!preview) return;
    // Shows the effect of the timezone field in the user's own terms --
    // an IANA name is easy to typo and hard to verify by eye.
    try {
      preview.textContent = `Right now the twin would say it is ${new Intl.DateTimeFormat(undefined, {
        dateStyle: "full",
        timeStyle: "short",
        ...(settings.timezone ? { timeZone: settings.timezone } : {}),
      }).format(new Date())}.`;
    } catch {
      preview.textContent = `“${settings.timezone}” isn't a timezone name the system recognises — falling back to this Mac's own time.`;
    }
  };

  // Edited locally (add/remove/type) and only sent to the sidecar on Save,
  // same explicit-save shape as the timezone field -- typing a coordinate
  // shouldn't fire a request per keystroke.
  let editableLocations: WeatherLocation[] = [];

  const renderWeatherLocations = () => {
    weatherLocationsEl.innerHTML = "";
    editableLocations.forEach((location, index) => {
      const row = document.createElement("div");
      row.className = "memory-form-row tools-weather-row";

      const nameInput = document.createElement("input");
      nameInput.type = "text";
      nameInput.placeholder = "Name";
      nameInput.value = location.name;
      nameInput.className = "tools-weather-name";
      nameInput.setAttribute("aria-label", "Place name");
      nameInput.addEventListener("input", () => {
        editableLocations[index] = { ...editableLocations[index], name: nameInput.value };
      });

      const latInput = document.createElement("input");
      latInput.type = "number";
      latInput.step = "any";
      latInput.placeholder = "Latitude";
      latInput.value = String(location.latitude);
      latInput.className = "tools-weather-coord";
      latInput.setAttribute("aria-label", `Latitude of ${location.name || "this place"}`);
      latInput.addEventListener("input", () => {
        editableLocations[index] = { ...editableLocations[index], latitude: Number(latInput.value) };
      });

      const lonInput = document.createElement("input");
      lonInput.type = "number";
      lonInput.step = "any";
      lonInput.placeholder = "Longitude";
      lonInput.value = String(location.longitude);
      lonInput.className = "tools-weather-coord";
      lonInput.setAttribute("aria-label", `Longitude of ${location.name || "this place"}`);
      lonInput.addEventListener("input", () => {
        editableLocations[index] = { ...editableLocations[index], longitude: Number(lonInput.value) };
      });

      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.className = "tools-weather-remove";
      removeButton.textContent = "×";
      removeButton.setAttribute("aria-label", `Remove ${location.name || "this place"}`);
      removeButton.addEventListener("click", () => {
        editableLocations = editableLocations.filter((_, i) => i !== index);
        renderWeatherLocations();
      });

      row.append(nameInput, latInput, lonInput, removeButton);
      weatherLocationsEl.appendChild(row);
    });
  };

  const render = (settings: ToolSettings) => {
    clockEnabled.checked = settings.clock_enabled;
    timezone.value = settings.timezone;
    renderPreview(settings);
    if (warning && warningDetail) {
      warning.hidden = settings.model_supports_tools;
      warningDetail.textContent =
        ` Mimoid is set to “${settings.model}”, which wasn't trained to call tools — ` +
        `the twin will answer normally but never look anything up. Switch to a tool-capable ` +
        `model (qwen2.5, llama3.1 or newer) to use this tab.`;
    }
    weatherEnabled.checked = settings.weather_enabled;
    editableLocations = settings.weather_locations.map((location) => ({ ...location }));
    renderWeatherLocations();

    calendarEnabled.checked = settings.calendar_enabled;
    calendarConfirm.checked = settings.calendar_confirm;
    calendarStatus.textContent = settings.calendar_connected
      ? "Connected"
      : settings.calendar_configured
        ? "Not connected"
        : "Not set up";
  };

  const save = async (
    patch: Partial<ToolSettings>,
    button?: HTMLButtonElement,
    setError: (message: string | null) => void = showError,
  ) => {
    if (button) button.disabled = true;
    setError(null);
    try {
      render(await updateToolSettings(patch));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      if (button) button.disabled = false;
    }
  };

  clockEnabled.addEventListener("change", () => {
    void save({ clock_enabled: clockEnabled.checked });
  });
  timezoneSave.addEventListener("click", () => {
    void save({ timezone: timezone.value }, timezoneSave);
  });

  weatherEnabled.addEventListener("change", () => {
    void save({ weather_enabled: weatherEnabled.checked }, undefined, showWeatherError);
  });
  weatherAdd.addEventListener("click", () => {
    editableLocations = [...editableLocations, { name: "", latitude: 0, longitude: 0 }];
    renderWeatherLocations();
  });
  weatherSave.addEventListener("click", () => {
    const cleaned = editableLocations
      .map((location) => ({
        name: location.name.trim(),
        latitude: location.latitude,
        longitude: location.longitude,
      }))
      .filter((location) => location.name && Number.isFinite(location.latitude) && Number.isFinite(location.longitude));
    if (cleaned.length === 0) {
      showWeatherError("Add at least one place with a name and valid coordinates.");
      return;
    }
    void save({ weather_locations: cleaned }, weatherSave, showWeatherError);
  });

  calendarEnabled.addEventListener("change", () => {
    void save({ calendar_enabled: calendarEnabled.checked }, undefined, showCalendarError);
  });
  calendarConfirm.addEventListener("change", () => {
    void save({ calendar_confirm: calendarConfirm.checked }, undefined, showCalendarError);
  });
  calendarManage.addEventListener("click", () => {
    // The modal owns the connect/disconnect flow; this card just reflects
    // the outcome once something actually changes, via the same
    // getToolSettings()-then-render() round trip every other save() here
    // uses -- calendar_configured/calendar_connected live on ToolSettings
    // precisely so this card doesn't need its own separate fetch.
    openGoogleCalendarModal(() => {
      void (async () => {
        try {
          render(await getToolSettings());
        } catch (err) {
          showCalendarError((err as Error).message);
        }
      })();
    });
  });

  void (async () => {
    try {
      render(await getToolSettings());
    } catch (err) {
      showError((err as Error).message);
    }
  })();
}
