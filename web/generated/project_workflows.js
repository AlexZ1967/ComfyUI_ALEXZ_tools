// frontend/src/project_workflows.ts
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// frontend/src/workflow_file_save.ts
var WorkflowFileSaver = class {
  constructor(options) {
    this.options = options;
  }
  options;
  targets = /* @__PURE__ */ new Map();
  busy = /* @__PURE__ */ new Set();
  /** Save to an explicit native-picker destination or reuse this graph's chosen handle. */
  async save(saveAs = false) {
    const id = this.options.graphId();
    if (this.busy.has(id)) throw new Error("\u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435 \u044D\u0442\u043E\u0433\u043E workflow \u0443\u0436\u0435 \u0432\u044B\u043F\u043E\u043B\u043D\u044F\u0435\u0442\u0441\u044F.");
    const previous = this.targets.get(id);
    const chooseFile = saveAs || !previous;
    const picker = this.options.picker();
    if (chooseFile && !picker) throw new Error("\u0411\u0440\u0430\u0443\u0437\u0435\u0440 \u043D\u0435 \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u0438\u0432\u0430\u0435\u0442 \u0441\u0438\u0441\u0442\u0435\u043C\u043D\u044B\u0439 \u0434\u0438\u0430\u043B\u043E\u0433 \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u044F. \u0418\u0441\u043F\u043E\u043B\u044C\u0437\u0443\u0439\u0442\u0435 \u0441\u0442\u0430\u043D\u0434\u0430\u0440\u0442\u043D\u044B\u0439 Export \u0438 \u0432\u043A\u043B\u044E\u0447\u0438\u0442\u0435 \u0432 \u043D\u0430\u0441\u0442\u0440\u043E\u0439\u043A\u0430\u0445 \u0431\u0440\u0430\u0443\u0437\u0435\u0440\u0430 \u0437\u0430\u043F\u0440\u043E\u0441 \u043F\u0430\u043F\u043A\u0438 \u0434\u043B\u044F \u0441\u043A\u0430\u0447\u0438\u0432\u0430\u043D\u0438\u044F.");
    this.busy.add(id);
    try {
      let handle;
      if (chooseFile) {
        try {
          handle = await picker({
            id: "alexz-workflow-json",
            suggestedName: previous?.handle.name ?? this.options.suggestedName(),
            types: [{ description: "ComfyUI workflow (JSON)", accept: { "application/json": [".json"] } }],
            ...previous ? { startIn: previous.handle } : {}
          });
        } catch (error) {
          if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") return { cancelled: true };
          throw error;
        }
      } else handle = previous.handle;
      if (this.options.graphId() !== id) throw new Error("\u0410\u043A\u0442\u0438\u0432\u043D\u044B\u0439 workflow \u0438\u0437\u043C\u0435\u043D\u0438\u043B\u0441\u044F. \u041F\u043E\u0432\u0442\u043E\u0440\u0438\u0442\u0435 \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435 \u0434\u043B\u044F \u043D\u0443\u0436\u043D\u043E\u0433\u043E \u0433\u0440\u0430\u0444\u0430.");
      const workflow = await this.options.workflow();
      if (this.options.graphId() !== id) throw new Error("\u0410\u043A\u0442\u0438\u0432\u043D\u044B\u0439 workflow \u0438\u0437\u043C\u0435\u043D\u0438\u043B\u0441\u044F. \u041F\u043E\u0432\u0442\u043E\u0440\u0438\u0442\u0435 \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435 \u0434\u043B\u044F \u043D\u0443\u0436\u043D\u043E\u0433\u043E \u0433\u0440\u0430\u0444\u0430.");
      const text = JSON.stringify(workflow, null, 2) + "\n";
      if (!chooseFile && previous) {
        const current = await handle.getFile();
        if (current.lastModified !== previous.lastModified || current.size !== previous.size) throw new Error("\u0424\u0430\u0439\u043B \u0438\u0437\u043C\u0435\u043D\u0451\u043D \u0434\u0440\u0443\u0433\u0438\u043C \u043F\u0440\u0438\u043B\u043E\u0436\u0435\u043D\u0438\u0435\u043C. \u0418\u0441\u043F\u043E\u043B\u044C\u0437\u0443\u0439\u0442\u0435 \xAB\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u043A\u0430\u043A\u2026\xBB \u0438 \u0432\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0444\u0430\u0439\u043B \u0437\u0430\u043D\u043E\u0432\u043E, \u0435\u0441\u043B\u0438 \u0445\u043E\u0442\u0438\u0442\u0435 \u0435\u0433\u043E \u043F\u0435\u0440\u0435\u0437\u0430\u043F\u0438\u0441\u0430\u0442\u044C.");
      }
      const writer = await handle.createWritable();
      try {
        await writer.write(text);
        await writer.close();
      } catch (error) {
        await writer.abort().catch(() => {
        });
        throw error;
      }
      const file = await handle.getFile();
      this.targets.set(id, { handle, lastModified: file.lastModified, size: file.size });
      return { cancelled: false, filename: handle.name };
    } finally {
      this.busy.delete(id);
    }
  }
};

// frontend/src/project_workflows.ts
var runtime = app;
var route = "/alexz/project-workflows";
var savedByGraph = /* @__PURE__ */ new Map();
var activeDialog = null;
var fileSaver = new WorkflowFileSaver({
  graphId: () => String(runtime.rootGraph?.id ?? "current"),
  workflow: async () => (await runtime.graphToPrompt(runtime.rootGraph)).workflow,
  suggestedName: () => {
    const hint = runtime.rootGraph?.extra?.alexz_project_workflow;
    return typeof hint === "string" && !/[\\/]/.test(hint) ? hint : "workflow.json";
  },
  picker: () => {
    const picker = window.showSaveFilePicker;
    return picker?.bind(window);
  }
});
var nativeStatus = null;
async function saveNative(saveAs) {
  try {
    const result = await fileSaver.save(saveAs);
    if (nativeStatus?.isConnected) nativeStatus.textContent = result.cancelled ? "\u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435 \u043E\u0442\u043C\u0435\u043D\u0435\u043D\u043E, \u0444\u0430\u0439\u043B\u044B \u043D\u0435 \u0438\u0437\u043C\u0435\u043D\u0435\u043D\u044B." : `\u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u043E: ${result.filename}`;
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    if (nativeStatus?.isConnected) nativeStatus.textContent = text;
    else window.alert(text);
  }
}
async function request(options) {
  const response = await api.fetchApi(route, options);
  if (response.status === 404) throw new Error("\u041C\u0430\u0440\u0448\u0440\u0443\u0442 \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u044F \u0435\u0449\u0451 \u043D\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043D. \u041F\u0435\u0440\u0435\u0437\u0430\u043F\u0443\u0441\u0442\u0438\u0442\u0435 ComfyUI \u0438 \u043E\u0431\u043D\u043E\u0432\u0438\u0442\u0435 \u0441\u0442\u0440\u0430\u043D\u0438\u0446\u0443.");
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : `\u041E\u0448\u0438\u0431\u043A\u0430 HTTP ${response.status}`);
  return data;
}
function jsonName(value) {
  const name = value.trim();
  return name.toLowerCase().endsWith(".json") ? name.slice(0, -5) + ".json" : name + ".json";
}
async function openProjectWorkflowSave() {
  if (activeDialog?.isConnected) {
    activeDialog.focus();
    return;
  }
  const dialog = document.createElement("dialog");
  activeDialog = dialog;
  dialog.className = "alexz-project-workflow-dialog";
  dialog.setAttribute("aria-label", "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430");
  const style = document.createElement("style");
  style.textContent = `
      .alexz-project-workflow-dialog { width:min(520px,90vw); padding:24px; border:1px solid var(--border-color,#666); border-radius:10px; color:var(--fg-color,#eee); background:var(--comfy-menu-bg,#252525); }
      .alexz-project-workflow-dialog::backdrop { background:#0009; }
      .alexz-project-workflow-dialog form { display:flex; flex-direction:column; gap:14px; }
      .alexz-project-workflow-dialog h2 { margin:0; font-size:1.2rem; }
      .alexz-project-workflow-dialog label { display:flex; flex-direction:column; gap:6px; }
      .alexz-project-workflow-dialog input,.alexz-project-workflow-dialog select { width:100%; padding:8px; color:inherit; background:var(--comfy-input-bg,#333); border:1px solid var(--border-color,#666); border-radius:4px; }
      .alexz-project-workflow-dialog p { margin:0; overflow-wrap:anywhere; }
      .alexz-project-workflow-dialog .actions { display:flex; gap:10px; justify-content:flex-end; }
      .alexz-project-workflow-dialog button { padding:8px 14px; border:1px solid var(--border-color,#666); border-radius:4px; background:var(--comfy-input-bg,#333); color:inherit; cursor:pointer; }
      .alexz-project-workflow-dialog button:disabled { opacity:.5; cursor:wait; }
    `;
  const form = document.createElement("form");
  const heading = document.createElement("h2");
  heading.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430";
  const description = document.createElement("p");
  description.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u0441\u044F \u0442\u0435\u043A\u0443\u0449\u0438\u0439 \u0433\u0440\u0430\u0444 \u0446\u0435\u043B\u0438\u043A\u043E\u043C. \u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044E\u0449\u0438\u0439 \u0444\u0430\u0439\u043B \u0434\u043B\u044F \u043F\u0435\u0440\u0435\u0437\u0430\u043F\u0438\u0441\u0438 \u0438\u043B\u0438 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u043D\u043E\u0432\u043E\u0435 \u0438\u043C\u044F.";
  const existingLabel = document.createElement("label");
  existingLabel.textContent = "\u0424\u0430\u0439\u043B \u043F\u0440\u043E\u0435\u043A\u0442\u0430";
  const existing = document.createElement("select");
  existing.setAttribute("aria-label", "\u0424\u0430\u0439\u043B \u043F\u0440\u043E\u0435\u043A\u0442\u0430");
  const fresh = document.createElement("option");
  fresh.value = "";
  fresh.textContent = "\u041D\u043E\u0432\u044B\u0439 \u0444\u0430\u0439\u043B\u2026";
  existing.append(fresh);
  existingLabel.append(existing);
  const nameLabel = document.createElement("label");
  nameLabel.textContent = "\u0418\u043C\u044F JSON-\u0444\u0430\u0439\u043B\u0430";
  const name = document.createElement("input");
  name.setAttribute("aria-label", "\u0418\u043C\u044F JSON-\u0444\u0430\u0439\u043B\u0430");
  name.required = true;
  name.autocomplete = "off";
  name.value = "workflow.json";
  nameLabel.append(name);
  const destination = document.createElement("p");
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.textContent = "\u0427\u0442\u0435\u043D\u0438\u0435 \u0441\u043F\u0438\u0441\u043A\u0430 \u0444\u0430\u0439\u043B\u043E\u0432\u2026";
  const actions = document.createElement("div");
  actions.className = "actions";
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "\u0417\u0430\u043A\u0440\u044B\u0442\u044C";
  close.onclick = () => dialog.close();
  const save = document.createElement("button");
  save.type = "submit";
  save.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C";
  save.disabled = true;
  actions.append(close, save);
  form.append(heading, description, existingLabel, nameLabel, destination, status, actions);
  dialog.append(style, form);
  document.body.append(dialog);
  dialog.addEventListener("close", () => {
    dialog.remove();
    if (activeDialog === dialog) activeDialog = null;
  });
  dialog.showModal();
  const entries = /* @__PURE__ */ new Map();
  let graphId = "";
  function describe() {
    const filename = jsonName(name.value);
    existing.value = entries.has(filename) ? filename : "";
    destination.textContent = `workflows/${filename} \u2014 ${entries.has(filename) ? "\u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044E\u0449\u0438\u0439 \u0444\u0430\u0439\u043B \u0431\u0443\u0434\u0435\u0442 \u043F\u0435\u0440\u0435\u0437\u0430\u043F\u0438\u0441\u0430\u043D" : "\u0431\u0443\u0434\u0435\u0442 \u0441\u043E\u0437\u0434\u0430\u043D \u043D\u043E\u0432\u044B\u0439 \u0444\u0430\u0439\u043B"}.`;
  }
  existing.onchange = () => {
    if (existing.value) name.value = existing.value;
    else name.value = "workflow.json";
    describe();
  };
  name.oninput = describe;
  try {
    const [{ workflow }, listing] = await Promise.all([runtime.graphToPrompt(runtime.rootGraph), request()]);
    if (!dialog.isConnected) return;
    graphId = String(runtime.rootGraph?.id ?? workflow.id ?? "current");
    for (const file of listing.files) {
      entries.set(file.filename, file);
      const option = document.createElement("option");
      option.value = option.textContent = file.filename;
      existing.append(option);
    }
    const hint = savedByGraph.get(graphId) ?? workflow.extra?.alexz_project_workflow;
    if (typeof hint === "string" && !/[\\/]/.test(hint)) name.value = hint;
    describe();
    status.textContent = "";
    save.disabled = false;
    name.focus();
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  }
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (save.disabled) return;
    save.disabled = true;
    close.disabled = true;
    status.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435\u2026";
    try {
      const { workflow } = await runtime.graphToPrompt(runtime.rootGraph);
      if (String(runtime.rootGraph?.id ?? workflow.id ?? "current") !== graphId) throw new Error("\u0410\u043A\u0442\u0438\u0432\u043D\u044B\u0439 workflow \u0438\u0437\u043C\u0435\u043D\u0438\u043B\u0441\u044F. \u0417\u0430\u043A\u0440\u043E\u0439\u0442\u0435 \u043E\u043A\u043D\u043E \u0438 \u043E\u0442\u043A\u0440\u043E\u0439\u0442\u0435 \u0435\u0433\u043E \u0441\u043D\u043E\u0432\u0430.");
      const filename = jsonName(name.value);
      const result = await request({
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Alexz-Workflow": "1" },
        body: JSON.stringify({ filename, workflow, expected_revision: entries.get(filename)?.revision ?? null })
      });
      savedByGraph.set(graphId, result.filename);
      if (!entries.has(result.filename)) {
        const option = document.createElement("option");
        option.value = option.textContent = result.filename;
        existing.append(option);
      }
      entries.set(result.filename, result);
      name.value = result.filename;
      describe();
      status.textContent = `\u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u043E: workflows/${result.filename}`;
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      save.disabled = false;
      close.disabled = false;
    }
  };
}
app.registerExtension({
  name: "ALEXZ.Tools.ProjectWorkflowSave",
  setup() {
    runtime.extensionManager.registerSidebarTab({
      id: "alexz-project-workflows",
      icon: "pi pi-save",
      title: "Save Workflow",
      tooltip: "Save Workflow",
      type: "custom",
      render(container) {
        const panel = document.createElement("section");
        panel.style.cssText = "padding:16px;display:flex;flex-direction:column;gap:14px";
        const description = document.createElement("p");
        description.textContent = "\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u043F\u0430\u043F\u043A\u0443 \u0438 \u0444\u0430\u0439\u043B \u0447\u0435\u0440\u0435\u0437 \u0441\u0438\u0441\u0442\u0435\u043C\u043D\u043E\u0435 \u043E\u043A\u043D\u043E \xAB\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u043A\u0430\u043A\u2026\xBB. \u0414\u043B\u044F \u043F\u043E\u0432\u0442\u043E\u0440\u043D\u043E\u0439 \u0437\u0430\u043F\u0438\u0441\u0438 \u0438\u0441\u043F\u043E\u043B\u044C\u0437\u0443\u0439\u0442\u0435 \xAB\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C\xBB.";
        const saveAs = document.createElement("button");
        saveAs.type = "button";
        saveAs.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u043A\u0430\u043A\u2026";
        saveAs.className = "comfyui-button";
        saveAs.onclick = () => {
          void saveNative(true);
        };
        const saveFile = document.createElement("button");
        saveFile.type = "button";
        saveFile.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C";
        saveFile.className = "comfyui-button";
        saveFile.onclick = () => {
          void saveNative(false);
        };
        const status = document.createElement("p");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        status.textContent = "\u041F\u043E\u0441\u043B\u0435 \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F \u0441\u0442\u0440\u0430\u043D\u0438\u0446\u044B \u0444\u0430\u0439\u043B \u043D\u0443\u0436\u043D\u043E \u0432\u044B\u0431\u0440\u0430\u0442\u044C \u0437\u0430\u043D\u043E\u0432\u043E. \u0421\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u0435 \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430 \u0434\u043E\u0441\u0442\u0443\u043F\u043D\u043E \u043E\u0442\u0434\u0435\u043B\u044C\u043D\u043E \u043D\u0438\u0436\u0435.";
        nativeStatus = status;
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430";
        button.className = "comfyui-button";
        button.onclick = () => {
          void openProjectWorkflowSave();
        };
        panel.append(description, saveAs, saveFile, status, document.createElement("hr"), button);
        container.replaceChildren(panel);
      }
    });
  },
  commands: [{
    id: "ALEXZ.Workflow.SaveAs",
    label: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C workflow \u043A\u0430\u043A\u2026",
    menubarLabel: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C workflow \u043A\u0430\u043A\u2026",
    icon: "pi pi-folder-open",
    function: () => saveNative(true)
  }, {
    id: "ALEXZ.Workflow.SaveFile",
    label: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C workflow \u0432 \u0432\u044B\u0431\u0440\u0430\u043D\u043D\u044B\u0439 \u0444\u0430\u0439\u043B",
    menubarLabel: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C workflow \u0432 \u0432\u044B\u0431\u0440\u0430\u043D\u043D\u044B\u0439 \u0444\u0430\u0439\u043B",
    icon: "pi pi-save",
    function: () => saveNative(false)
  }, {
    id: "ALEXZ.ProjectWorkflow.Save",
    label: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430",
    menubarLabel: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C \u0432 workflows \u043F\u0440\u043E\u0435\u043A\u0442\u0430",
    icon: "pi pi-save",
    function: openProjectWorkflowSave
  }],
  menuCommands: [{ path: ["File"], commands: ["ALEXZ.Workflow.SaveAs", "ALEXZ.Workflow.SaveFile", "ALEXZ.ProjectWorkflow.Save"] }]
});
//# sourceMappingURL=project_workflows.js.map
