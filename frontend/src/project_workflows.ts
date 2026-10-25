import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

import { WorkflowFileSaver, type Workflow, type WorkflowFilePicker } from "./workflow_file_save";

interface WorkflowApp {
    rootGraph?: { id?: string; extra?: Record<string, unknown> };
    graphToPrompt(graph?: unknown): Promise<{ workflow: Workflow }>;
    extensionManager: {
        registerSidebarTab(tab: { id: string; icon: string; title: string; tooltip: string; type: "custom"; render(container: HTMLElement): void }): void;
    };
}
interface FileEntry { filename: string; revision: string }
interface FileList { directory: string; files: FileEntry[] }
interface SavedFile extends FileEntry { directory: string }

const runtime = app as unknown as WorkflowApp;
const route = "/alexz/project-workflows";
const savedByGraph = new Map<string, string>();
let activeDialog: HTMLDialogElement | null = null;
const fileSaver = new WorkflowFileSaver({
    graphId: () => String(runtime.rootGraph?.id ?? "current"),
    workflow: async () => (await runtime.graphToPrompt(runtime.rootGraph)).workflow,
    suggestedName: () => {
        const hint = runtime.rootGraph?.extra?.alexz_project_workflow;
        return typeof hint === "string" && !/[\\/]/.test(hint) ? hint : "workflow.json";
    },
    picker: () => {
        const picker = (window as unknown as { showSaveFilePicker?: WorkflowFilePicker }).showSaveFilePicker;
        return picker?.bind(window);
    },
});
let nativeStatus: HTMLElement | null = null;

/** Keep the native picker in the original click handler, before any asynchronous preparation. */
async function saveNative(saveAs: boolean): Promise<void> {
    try {
        const result = await fileSaver.save(saveAs);
        if (nativeStatus?.isConnected) nativeStatus.textContent = result.cancelled ? "Сохранение отменено, файлы не изменены." : `Сохранено: ${result.filename}`;
    } catch (error) {
        const text = error instanceof Error ? error.message : String(error);
        if (nativeStatus?.isConnected) nativeStatus.textContent = text;
        else window.alert(text);
    }
}


/** Read structured route errors without hiding restart requirements. */
async function request<T>(options?: RequestInit): Promise<T> {
    const response = await api.fetchApi(route, options);
    if (response.status === 404) throw new Error("Маршрут сохранения ещё не загружен. Перезапустите ComfyUI и обновите страницу.");
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : `Ошибка HTTP ${response.status}`);
    return data as T;
}

/** Normalize the display name; the backend independently validates its filesystem scope. */
function jsonName(value: string): string {
    const name = value.trim();
    return name.toLowerCase().endsWith(".json") ? name.slice(0, -5) + ".json" : name + ".json";
}

/** Build a modal for explicit creation or overwrite in the project's workflow directory. */
async function openProjectWorkflowSave(): Promise<void> {
    if (activeDialog?.isConnected) { activeDialog.focus(); return; }
    const dialog = document.createElement("dialog");
    activeDialog = dialog;
    dialog.className = "alexz-project-workflow-dialog";
    dialog.setAttribute("aria-label", "Сохранить в workflows проекта");
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
    heading.textContent = "Сохранить в workflows проекта";
    const description = document.createElement("p");
    description.textContent = "Сохранится текущий граф целиком. Выберите существующий файл для перезаписи или задайте новое имя.";
    const existingLabel = document.createElement("label");
    existingLabel.textContent = "Файл проекта";
    const existing = document.createElement("select");
    existing.setAttribute("aria-label", "Файл проекта");
    const fresh = document.createElement("option");
    fresh.value = "";
    fresh.textContent = "Новый файл…";
    existing.append(fresh);
    existingLabel.append(existing);
    const nameLabel = document.createElement("label");
    nameLabel.textContent = "Имя JSON-файла";
    const name = document.createElement("input");
    name.setAttribute("aria-label", "Имя JSON-файла");
    name.required = true;
    name.autocomplete = "off";
    name.value = "workflow.json";
    nameLabel.append(name);
    const destination = document.createElement("p");
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = "Чтение списка файлов…";
    const actions = document.createElement("div");
    actions.className = "actions";
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Закрыть";
    close.onclick = () => dialog.close();
    const save = document.createElement("button");
    save.type = "submit";
    save.textContent = "Сохранить";
    save.disabled = true;
    actions.append(close, save);
    form.append(heading, description, existingLabel, nameLabel, destination, status, actions);
    dialog.append(style, form);
    document.body.append(dialog);
    dialog.addEventListener("close", () => { dialog.remove(); if (activeDialog === dialog) activeDialog = null; });
    dialog.showModal();

    const entries = new Map<string, FileEntry>();
    let graphId = "";
    /** Show the concrete destination and overwrite behavior before the user submits. */
    function describe(): void {
        const filename = jsonName(name.value);
        existing.value = entries.has(filename) ? filename : "";
        destination.textContent = `workflows/${filename} — ${entries.has(filename) ? "существующий файл будет перезаписан" : "будет создан новый файл"}.`;
    }
    existing.onchange = () => {
        if (existing.value) name.value = existing.value;
        else name.value = "workflow.json";
        describe();
    };
    name.oninput = describe;
    try {
        const [{ workflow }, listing] = await Promise.all([runtime.graphToPrompt(runtime.rootGraph), request<FileList>()]);
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
    form.onsubmit = async event => {
        event.preventDefault();
        if (save.disabled) return;
        save.disabled = true;
        close.disabled = true;
        status.textContent = "Сохранение…";
        try {
            const { workflow } = await runtime.graphToPrompt(runtime.rootGraph);
            if (String(runtime.rootGraph?.id ?? workflow.id ?? "current") !== graphId) throw new Error("Активный workflow изменился. Закройте окно и откройте его снова.");
            const filename = jsonName(name.value);
            const result = await request<SavedFile>({
                method: "POST",
                headers: { "Content-Type": "application/json", "X-Alexz-Workflow": "1" },
                body: JSON.stringify({ filename, workflow, expected_revision: entries.get(filename)?.revision ?? null }),
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
            status.textContent = `Сохранено: workflows/${result.filename}`;
        } catch (error) {
            status.textContent = error instanceof Error ? error.message : String(error);
        } finally { save.disabled = false; close.disabled = false; }
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
                description.textContent = "Выберите папку и файл через системное окно «Сохранить как…». Для повторной записи используйте «Сохранить».";
                const saveAs = document.createElement("button");
                saveAs.type = "button";
                saveAs.textContent = "Сохранить как…";
                saveAs.className = "comfyui-button";
                saveAs.onclick = () => { void saveNative(true); };
                const saveFile = document.createElement("button");
                saveFile.type = "button";
                saveFile.textContent = "Сохранить";
                saveFile.className = "comfyui-button";
                saveFile.onclick = () => { void saveNative(false); };
                const status = document.createElement("p");
                status.setAttribute("role", "status");
                status.setAttribute("aria-live", "polite");
                status.textContent = "После обновления страницы файл нужно выбрать заново. Сохранение в workflows проекта доступно отдельно ниже.";
                nativeStatus = status;
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = "Сохранить в workflows проекта";
                button.className = "comfyui-button";
                button.onclick = () => { void openProjectWorkflowSave(); };
                panel.append(description, saveAs, saveFile, status, document.createElement("hr"), button);
                container.replaceChildren(panel);
            },
        });
    },
    commands: [{
        id: "ALEXZ.Workflow.SaveAs",
        label: "Сохранить workflow как…",
        menubarLabel: "Сохранить workflow как…",
        icon: "pi pi-folder-open",
        function: () => saveNative(true),
    }, {
        id: "ALEXZ.Workflow.SaveFile",
        label: "Сохранить workflow в выбранный файл",
        menubarLabel: "Сохранить workflow в выбранный файл",
        icon: "pi pi-save",
        function: () => saveNative(false),
    }, {
        id: "ALEXZ.ProjectWorkflow.Save",
        label: "Сохранить в workflows проекта",
        menubarLabel: "Сохранить в workflows проекта",
        icon: "pi pi-save",
        function: openProjectWorkflowSave,
    }],
    menuCommands: [{ path: ["File"], commands: ["ALEXZ.Workflow.SaveAs", "ALEXZ.Workflow.SaveFile", "ALEXZ.ProjectWorkflow.Save"] }],
});
