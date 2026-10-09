"use strict";
const $ = (id) => document.getElementById(id);
let csrf = "",
  users = [],
  isAdmin = false,
  selected = null,
  resetUser = null;
const base = new URL("./", location.href);
const fmt = (n) => Number(n || 0).toLocaleString();
const date = (n) => (n ? new Date(n).toLocaleString() : "No uploads yet");
function element(tag, text, className) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}
function notify(text) {
  $("notice").textContent = text;
  $("notice").hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => ($("notice").hidden = true), 7000);
}
async function api(path, method = "GET", body) {
  const r = await fetch(new URL(path, base), {
    method,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) {
    if (r.status === 401) showLogin();
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Please check the form values.",
    );
  }
  return data;
}
function action(text, callback, style = "secondary") {
  const b = element("button", text, style);
  b.type = "button";
  b.onclick = async () => {
    b.disabled = true;
    try {
      await callback();
    } catch (e) {
      notify(e.message);
    } finally {
      b.disabled = false;
    }
  };
  return b;
}
function showLogin() {
  csrf = "";
  $("dashboard").hidden = true;
  $("signin").hidden = false;
  $("apk-release").hidden = true;
  $("apk-download").removeAttribute("href");
}
function page(name) {
  for (const id of ["overview", "people", "device-page", "android-app"])
    $(id).hidden = id !== name;
  document
    .querySelectorAll("nav button")
    .forEach((b) => b.classList.toggle("selected", b.dataset.page === name));
  $("page-title").textContent =
    name === "overview"
      ? "Overview"
      : name === "people"
        ? "People & devices"
        : name === "android-app"
          ? "Android app"
          : selected.username + " · Devices";
}
async function refreshRelease() {
  try {
    const release = await api("account/android-release");
    $("apk-version").textContent = "Version " + release.version_name;
    $("apk-details").textContent =
      (release.size_bytes / 1024 / 1024).toFixed(1) + " MB · " +
      (release.minimum_sdk === 30 ? "Android 11 or newer" : "Android API " + release.minimum_sdk + " or newer");
    $("apk-download").href = new URL(release.download_path, base).href;
    $("apk-changelog").replaceChildren(
      ...release.changelog.map((note) => element("li", note)),
    );
    $("apk-status").hidden = true;
    $("apk-release").hidden = false;
  } catch (e) {
    $("apk-release").hidden = true;
    $("apk-download").removeAttribute("href");
    $("apk-status").textContent = e.message;
    $("apk-status").hidden = false;
  }
}
function stats(container, values) {
  container.replaceChildren();
  for (const [label, value] of values) {
    const box = element("div", undefined, "stat");
    box.append(element("span", label), element("strong", fmt(value)));
    container.append(box);
  }
}
function table(container, headers, rows) {
  const wrap = element("div", undefined, "table-wrap"),
    t = element("table"),
    head = element("tr");
  headers.forEach((h) => head.append(element("th", h)));
  const thead = element("thead");
  thead.append(head);
  t.append(thead);
  const body = element("tbody");
  for (const row of rows) {
    const tr = element("tr");
    for (const v of row) {
      const td = element("td");
      if (v instanceof Node) td.append(v);
      else td.textContent = v;
      tr.append(td);
    }
    body.append(tr);
  }
  t.append(body);
  wrap.append(t);
  container.replaceChildren(wrap);
}
async function refresh() {
  users = await api("admin/users");
  stats($("totals"), [
    ["People", users.length],
    ["Devices", users.reduce((s, u) => s + Number(u.devices), 0)],
    ["Runs", users.reduce((s, u) => s + Number(u.runs), 0)],
    ["Embeddings", users.reduce((s, u) => s + Number(u.samples), 0)],
  ]);
  table(
    $("activity"),
    ["Participant", "Devices", "Embeddings", "Last upload"],
    users.map((u) => [
      u.username,
      fmt(u.devices),
      fmt(u.samples),
      date(u.last_upload_ms),
    ]),
  );
  table(
    $("user-list"),
    [
      "Participant",
      "Created by",
      "Created",
      "Status",
      "Devices",
      "Embeddings",
      "Actions",
    ],
    users.map((u) => {
      const actions = element("div");
      actions.append(action("Devices", () => openDevices(u)));
      if (isAdmin)
        actions.append(
          action("Password", () => {
            resetUser = u;
            $("reset-for").textContent = u.username;
            $("password-dialog").showModal();
          }),
          action("Revoke sessions", async () => {
            if (confirm("Sign out " + u.username + " on every device?")) {
              await api("admin/users/" + u.id + "/revoke", "POST");
              notify("Sessions revoked");
              await refresh();
            }
          }),
        );
      if (isAdmin && !u.administrator)
        actions.append(
          action(
            u.active ? "Disable" : "Enable",
            async () => {
              if (
                u.active &&
                !confirm("Disable uploads and sign-in for " + u.username + "?")
              )
                return;
              await api("admin/users/" + u.id, "PATCH", { active: !u.active });
              await refresh();
            },
            u.active ? "danger" : "secondary",
          ),
        );
      return [
        u.username + (u.administrator ? " · Admin" : ""),
        u.created_by || "Not recorded",
        u.created_ms ? date(u.created_ms) : "—",
        element(
          "span",
          u.active ? "Active" : "Disabled",
          "badge" + (u.active ? "" : " off"),
        ),
        u.devices,
        fmt(u.samples),
        actions,
      ];
    }),
  );
  const created = await api("account/creations");
  table(
    $("created-list"),
    ["Account you created", "Created"],
    created.map((u) => [u.username, date(u.created_ms)]),
  );
  $("updated").textContent = "Updated " + new Date().toLocaleTimeString();
  if (selected && !$("device-page").hidden) await openDevices(selected);
  if (!$("android-app").hidden) await refreshRelease();
}
async function openDevices(user) {
  selected = user;
  const devices = await api("admin/users/" + user.id + "/devices");
  $("devices").replaceChildren();
  if (!devices.length)
    $("devices").append(
      element(
        "article",
        "No device has uploaded yet. Devices appear automatically after the first run or calibration upload.",
      ),
    );
  for (const d of devices) {
    const card = element("article", undefined, "device");
    card.append(
      element("h2", d.label || d.model || "Android device"),
      element("code", d.id),
      element(
        "p",
        (d.model || "Model available after first run") +
          (d.android_sdk ? " · Android API " + d.android_sdk : "") +
          " · Last upload: " +
          date(d.last_seen_ms),
      ),
    );
    const summary = element("div", undefined, "totals");
    stats(summary, [
      ["Embeddings", d.samples],
      ["Runs (complete / total)", d.complete_runs],
      ["Applications", d.packages],
    ]);
    summary.children[1].querySelector("strong").textContent =
      fmt(d.complete_runs) + " / " + fmt(d.runs);
    card.append(summary);
    const metrics = element("div", undefined, "metrics"),
      m = d.latest_metrics;
    for (const [label, value] of [
      [
        "Calibrated rate",
        d.calibration ? d.calibration.selected_fps + " /s" : "Not uploaded",
      ],
      ["Target rate", m.target_fps ? m.target_fps + " /s" : "—"],
      [
        "Preprocessing",
        m.preprocess_ms != null ? m.preprocess_ms.toFixed(2) + " ms" : "—",
      ],
      [
        "Inference",
        m.inference_ms != null ? m.inference_ms.toFixed(2) + " ms" : "—",
      ],
      ["Capture", m.capture_source || "—"],
      ["Mode", m.sampling_mode || "—"],
      ["Encoder", m.processing_backend || "—"],
      ["Preprocessor", m.preprocessing_backend || "—"],
    ])
      metrics.append(element("p", label + ": " + value));
    card.append(
      metrics,
      element(
        "p",
        "Timing values are from the latest uploaded sample.",
        "muted",
      ),
    );
    const form = element("div", undefined, "device-edit"),
      label = element("label", "Device name"),
      input = element("input");
    input.value = d.label;
    input.maxLength = 128;
    input.placeholder = d.model || "e.g. Work phone";
    label.append(input);
    form.append(
      label,
      action("Save name", async () => {
        await api("admin/users/" + user.id + "/devices/" + d.id, "PATCH", {
          label: input.value,
          active: d.active,
        });
        await openDevices(user);
      }),
      action(
        d.active ? "Disable uploads" : "Enable uploads",
        async () => {
          if (
            d.active &&
            !confirm(
              "Disable uploads from this device? Its recorded data will be retained.",
            )
          )
            return;
          await api("admin/users/" + user.id + "/devices/" + d.id, "PATCH", {
            label: input.value,
            active: !d.active,
          });
          await openDevices(user);
        },
        d.active ? "danger" : "secondary",
      ),
    );
    if (isAdmin)
      card.append(
        element(
          "p",
          "Disabling blocks uploads; the phone can continue recording locally. After enabling, tap Retry upload in the Android app.",
          "muted",
        ),
      );
    card.append(
      element(
        "span",
        d.active ? "Uploads enabled" : "Uploads disabled",
        "badge" + (d.active ? "" : " off"),
      ),
      ...(isAdmin ? [form] : []),
    );
    $("devices").append(card);
  }
  page("device-page");
}
async function signedIn(session) {
  csrf = session.csrf;
  isAdmin = session.administrator;
  selected = null;
  page("overview");
  $("people-description").textContent = isAdmin
    ? "Manage access and inspect each device independently."
    : "View your devices and create accounts for other participants. Only administrators can manage their access or see their recordings.";
  $("identity").textContent = session.username;
  $("signin").hidden = true;
  $("dashboard").hidden = false;
  $("server-address").value = base.href;
  await refresh();
}
$("login").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await signedIn(
      await api(
        "admin/login",
        "POST",
        Object.fromEntries(new FormData(e.target)),
      ),
    );
    e.target.reset();
  } catch (err) {
    notify(err.message);
  }
};
$("new-user").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api(
      "account/users",
      "POST",
      Object.fromEntries(new FormData(e.target)),
    );
    e.target.reset();
    $("user-dialog").close();
    await refresh();
    notify("Participant created");
  } catch (err) {
    notify(err.message);
  }
};
$("reset-password").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api(
      "admin/users/" + resetUser.id,
      "PATCH",
      Object.fromEntries(new FormData(e.target)),
    );
    e.target.reset();
    $("password-dialog").close();
    notify("Password changed and sessions revoked");
    await refresh();
  } catch (err) {
    notify(err.message);
  }
};
$("add-user").onclick = () => $("user-dialog").showModal();
$("back").onclick = () => {
  selected = null;
  page("people");
};
$("refresh").onclick = () => refresh().catch((e) => notify(e.message));
$("logout").onclick = async () => {
  try {
    await api("admin/logout", "POST");
    showLogin();
  } catch (e) {
    notify(e.message);
  }
};
document
  .querySelectorAll("[data-close]")
  .forEach((b) => (b.onclick = () => $(b.dataset.close).close()));
document.querySelectorAll("[data-page]").forEach(
  (b) =>
    (b.onclick = () => {
      selected = null;
      page(b.dataset.page);
      if (b.dataset.page === "android-app") refreshRelease();
    }),
);
api("admin/session")
  .then(signedIn)
  .catch((e) => {
    showLogin();
    if (e.message !== "Sign in to continue") notify(e.message);
  });
setInterval(() => {
  if (
    csrf &&
    !document.hidden &&
    !document.querySelector("dialog[open]") &&
    !$("device-page").contains(document.activeElement)
  )
    refresh().catch((e) => notify(e.message));
}, 30000);
