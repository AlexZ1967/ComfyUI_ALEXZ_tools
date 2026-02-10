/**
 * Module: web/module_node_picker.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Module Node Picker frontend panel.
 *
 * Purpose:
 *   Renders sidebar UI, loads module/node catalogs, and runs refresh/update actions for modules and ComfyUI.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    bindModuleNodesTabRelay,
    unbindModuleNodesTabRelay,
} from "./module_node_picker_tab_relay.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const SIDEBAR_TAB_ID = "alexz-module-nodes";
const MODULE_PICKER_GUARD_KEY = "__alexz_module_node_picker_registered__";
const FALLBACK_BUTTON_ID = "alexz-module-nodes-fallback-btn";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const CONTAINER_SYNC_STATE_KEY = "__alexz_module_nodes_container_sync_state__";
const NODE_PICKER_DEBUG_KEY = "__alexz_module_picker_debug__";
const NODE_PICKER_DEBUG_STORAGE_KEY = "alexz_module_picker_debug";
const NODE_PICKER_SIDEBAR_SYNC_KEY = "__alexz_module_picker_sidebar_sync__";
const GROUP_LABELS = {
    core: "Core_Nodes",
    core_extras: "Core_Extras_Nodes",
    api: "API_Nodes",
    custom: "Custom_Nodes",
};
const MODULE_MARK_UPDATED = "✅";
const MODULE_MARK_REMOTE_UPDATE = "🟥";

/**
 * Inject the stylesheet used by the Module Node Picker panel.
 * Safely no-ops when style tag is already present.
 */
function injectStyles() {
    const styleId = "alexz-module-picker-style";
    if (document.getElementById(styleId)) {
        return;
    }
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
    .alexz-mod-picker {
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 100%;
        overflow: auto;
    }
    .alexz-mod-picker-head {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alexz-mod-picker-head-actions {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .alexz-mod-picker-debug-toggle {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 11px;
        opacity: 0.9;
        user-select: none;
        white-space: nowrap;
    }
    .alexz-mod-picker-debug-toggle input {
        margin: 0;
    }
    .alexz-mod-picker-title {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.95;
        margin-right: auto;
    }
    .alexz-mod-picker-select {
        width: 100%;
    }
    .alexz-mod-picker-help {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-height: 2.2em;
    }
    .alexz-mod-picker-help-main {
        font-size: 13px;
        line-height: 1.3;
        opacity: 0.95;
    }
    .alexz-mod-picker-help-main strong {
        font-weight: 700;
    }
    .alexz-mod-picker-help-hint {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
    }
    .alexz-mod-picker-refresh-line {
        font-size: 12px;
        opacity: 0.92;
        min-height: 1.2em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--warn {
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-diag {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 10px;
        line-height: 1.25;
        opacity: 0.82;
        border: 1px dashed var(--border-color, #555);
        border-radius: 6px;
        padding: 6px;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 130px;
        overflow: auto;
    }
    .alexz-mod-picker-comfy-alert {
        border: 1px solid #b64040;
        background: rgba(180, 64, 64, 0.16);
        color: #ff6b6b;
        border-radius: 7px;
        padding: 7px 8px;
        font-size: 12px;
        line-height: 1.3;
        font-weight: 700;
        display: none;
    }
    .alexz-mod-picker-comfy-alert.alexz-mod-picker-comfy-alert--ok {
        border-color: #2e8f61;
        background: rgba(61, 187, 126, 0.16);
        color: #3dbb7e;
    }
    .alexz-mod-picker-module-card {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .alexz-mod-picker-module-card--updated {
        border-color: #3dbb7e;
        box-shadow: inset 0 0 0 1px rgba(61, 187, 126, 0.35);
    }
    .alexz-mod-picker-module-card--clickable {
        cursor: pointer;
        transition: filter 0.12s ease;
    }
    .alexz-mod-picker-module-card--clickable:hover {
        filter: brightness(1.08);
    }
    .alexz-mod-picker-module-title {
        font-size: 12px;
        font-weight: 700;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta {
        font-size: 11px;
        opacity: 0.9;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta a {
        color: var(--link-color, #87b5ff);
        text-decoration: underline;
    }
    .alexz-mod-picker-module-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        white-space: pre-wrap;
    }
    .alexz-mod-picker-module-note {
        font-size: 10px;
        opacity: 0.8;
        line-height: 1.25em;
        white-space: pre-wrap;
    }
    .alexz-mod-picker-module-row {
        font-size: 11px;
        opacity: 0.9;
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
    }
    .alexz-mod-picker-module-row.notice {
        color: #f0b429;
    }
    .alexz-mod-picker-module-row.warn {
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-module-row.ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-module-label {
        font-weight: 700;
        opacity: 0.95;
    }
    .alexz-mod-picker-status {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border-color, #555);
        border-radius: 10px;
        padding: 1px 7px;
        font-size: 10px;
        line-height: 1.4;
        font-weight: 700;
    }
    .alexz-mod-picker-status.up-to-date {
        color: #3dbb7e;
    }
    .alexz-mod-picker-status.can-update {
        color: #f0b429;
    }
    .alexz-mod-picker-status.unknown {
        color: #b3b3b3;
    }
    .alexz-mod-picker-action-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .alexz-mod-picker-btn-small {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid var(--border-color, #555);
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        cursor: pointer;
    }
    .alexz-mod-picker-btn-small:hover {
        filter: brightness(1.08);
    }
    .alexz-mod-picker-btn-small[disabled] {
        opacity: 0.55;
        cursor: default;
        filter: none;
    }
    .alexz-mod-picker-group {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        margin-top: 2px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
    }
    .alexz-mod-picker-group-title {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 7px;
        word-break: break-all;
    }
    .alexz-mod-picker-node {
        width: 100%;
        text-align: left;
        margin-bottom: 6px;
        border: 1px solid var(--border-color, #555);
        border-radius: 6px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        padding: 7px;
        cursor: pointer;
    }
    .alexz-mod-picker-node:hover {
        filter: brightness(1.12);
    }
    .alexz-mod-picker-node.alexz-mod-picker-node--updated {
        border-color: #3dbb7e;
        box-shadow: inset 0 0 0 1px rgba(61, 187, 126, 0.35);
    }
    .alexz-mod-picker-node.alexz-mod-picker-node--new {
        border-color: #d44f4f;
        box-shadow: inset 0 0 0 1px rgba(212, 79, 79, 0.35);
    }
    .alexz-mod-picker-node-name {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .alexz-mod-picker-node-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        word-break: break-all;
    }
    .alexz-mod-picker-floating-btn {
        position: fixed;
        left: 10px;
        bottom: 10px;
        z-index: 10005;
    }`;
    document.head.appendChild(style);
}

/**
 * Place a newly created node near the visible canvas center.
 * Falls back to a fixed position if visible area is unavailable.
 */
function centerNode(node) {
    const area = app.canvas?.visible_area;
    if (area && area.length >= 4) {
        node.pos = [
            area[0] + area[2] * 0.5 - node.size[0] * 0.5,
            area[1] + area[3] * 0.5 - node.size[1] * 0.5,
        ];
    } else {
        node.pos = [200, 120];
    }
}

/**
 * Create a LiteGraph node from catalog metadata.
 * Tries internal node name first, then display name as fallback.
 */
function createNodeByInfo(nodeInfo) {
    const candidates = [nodeInfo.node_name, nodeInfo.display_name].filter(Boolean);
    for (const name of candidates) {
        const node = LiteGraph.createNode(name);
        if (node) {
            return node;
        }
    }
    return null;
}

/**
 * Fetch grouped node catalog data from backend API.
 */
async function fetchNodeCatalog() {
    const resp = await api.fetchApi("/alexz_tools/node_catalog?cache_only=1", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Fetch detailed info for a specific module, optionally forcing refresh/sync.
 */
async function fetchModuleInfo(group, moduleName, options = {}) {
    const forceRefresh = Boolean(options?.forceRefresh);
    const syncUpstream = Boolean(options?.syncUpstream);
    const cacheOnly = options?.cacheOnly === undefined
        ? (!forceRefresh && !syncUpstream)
        : Boolean(options?.cacheOnly);
    const resp = await api.fetchApi(
        `/alexz_tools/module_info?group=${encodeURIComponent(group || "")}` +
        `&module=${encodeURIComponent(moduleName || "")}` +
        `&refresh=${forceRefresh ? "1" : "0"}` +
        `&sync_upstream=${syncUpstream ? "1" : "0"}` +
        `&cache_only=${cacheOnly ? "1" : "0"}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Fetch ComfyUI repository update status and metadata.
 */
async function fetchComfyUIInfo(forceRefresh = true, acknowledge = true) {
    const resp = await api.fetchApi(
        `/alexz_tools/comfyui_info?refresh=${forceRefresh ? "1" : "0"}&acknowledge=${acknowledge ? "1" : "0"}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Start backend refresh job that recomputes module/runtime snapshots.
 */
async function refreshModuleRuntimeState() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Poll refresh job status from backend.
 */
async function fetchModuleRefreshStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Acknowledge/clear novelty markers for all modules after global refresh action.
 */
async function acknowledgeAllModuleNovelty() {
    const resp = await api.fetchApi("/alexz_tools/module_acknowledge_all", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Start backend update job for a single module, all modules, or ComfyUI.
 */
async function startModuleUpdate(scope, moduleName) {
    const resp = await api.fetchApi("/alexz_tools/module_update", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scope: scope || "single",
            module: moduleName || "",
        }),
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Poll module-update job status from backend.
 */
async function fetchModuleUpdateStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_update_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Install requirements.txt for selected custom modules in current runtime env.
 */
async function installModuleRequirements(modules) {
    const resp = await api.fetchApi("/alexz_tools/module_install_requirements", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: Array.isArray(modules) ? modules : [] }),
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Install ComfyUI requirements.txt in current runtime environment.
 */
async function installComfyUIRequirements() {
    const resp = await api.fetchApi("/alexz_tools/comfyui_install_requirements", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Format ISO timestamp for local UI display.
 */
function fmtDate(iso) {
    if (!iso) {
        return "n/a";
    }
    try {
        return new Date(iso).toLocaleString();
    } catch (err) {
        return String(iso);
    }
}

/**
 * Derive UI badge flags from module info payload.
 */
function moduleBadgesFromInfo(info) {
    const behind = Number(info?.git_behind);
    const status = String(info?.update_status || "");
    return {
        updatedBetweenRuns: Boolean(info?.updated_between_runs),
        hasRemoteUpdate: (Number.isFinite(behind) && behind > 0) || status === "can_update",
    };
}

/**
 * Derive UI badge flags from lightweight module entry in node-catalog payload.
 */
function moduleBadgesFromModuleEntry(entry) {
    return {
        updatedBetweenRuns: Boolean(entry?.updated_between_runs) || Boolean(entry?.new_module_between_runs),
        hasRemoteUpdate: Boolean(entry?.update_available),
    };
}

/**
 * Build text shown in module select option with update badges and node count.
 */
function formatModuleOption(moduleName, count, badges) {
    const marks = [];
    if (badges?.updatedBetweenRuns) {
        marks.push(MODULE_MARK_UPDATED);
    }
    if (badges?.hasRemoteUpdate) {
        marks.push(MODULE_MARK_REMOTE_UPDATE);
    }
    const prefix = marks.length ? `${marks.join(" ")} ` : "";
    return `${prefix}${moduleName} (${count})`;
}

/**
 * Render Module Node Picker UI and bind all panel event handlers.
 */
function renderPicker(container) {
    // Safety: clear any previous global tab-sync hooks before re-rendering.
    unbindContainerOwnershipSync();
    unbindModuleNodesTabRelay();

    container.innerHTML = "";

    const root = document.createElement("div");
    root.className = "alexz-mod-picker";
    container.appendChild(root);

    const head = document.createElement("div");
    head.className = "alexz-mod-picker-head";
    root.appendChild(head);

    const title = document.createElement("div");
    title.className = "alexz-mod-picker-title";
    title.textContent = "Node Picker";
    head.appendChild(title);

    const headActions = document.createElement("div");
    headActions.className = "alexz-mod-picker-head-actions";
    head.appendChild(headActions);

    const updateAllBtn = document.createElement("button");
    updateAllBtn.type = "button";
    updateAllBtn.textContent = "Update all custom_nodes";
    updateAllBtn.className = "alexz-mod-picker-btn-small";
    updateAllBtn.style.display = "none";
    headActions.appendChild(updateAllBtn);

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.textContent = "Обновить информацию о модулях";
    refreshBtn.className = "alexz-mod-picker-btn-small";
    headActions.appendChild(refreshBtn);

    const comfyInfoBtn = document.createElement("button");
    comfyInfoBtn.type = "button";
    comfyInfoBtn.textContent = "Обновить информацию о ComfyUI";
    comfyInfoBtn.className = "alexz-mod-picker-btn-small";
    headActions.appendChild(comfyInfoBtn);

    const debugToggleLabel = document.createElement("label");
    debugToggleLabel.className = "alexz-mod-picker-debug-toggle";
    const debugToggle = document.createElement("input");
    debugToggle.type = "checkbox";
    debugToggleLabel.appendChild(debugToggle);
    debugToggleLabel.append("Debug");
    headActions.appendChild(debugToggleLabel);

    const comfyAlert = document.createElement("div");
    comfyAlert.className = "alexz-mod-picker-comfy-alert";
    const comfyAlertText = document.createElement("div");
    comfyAlert.appendChild(comfyAlertText);
    const comfyUpdateBtn = document.createElement("button");
    comfyUpdateBtn.type = "button";
    comfyUpdateBtn.className = "alexz-mod-picker-btn-small";
    comfyUpdateBtn.textContent = "Update ComfyUI";
    comfyUpdateBtn.style.marginTop = "6px";
    comfyUpdateBtn.style.display = "none";
    comfyAlert.appendChild(comfyUpdateBtn);
    root.appendChild(comfyAlert);

    const groupSelect = document.createElement("select");
    groupSelect.className = "alexz-mod-picker-select";
    root.appendChild(groupSelect);

    const nodeSelect = document.createElement("select");
    nodeSelect.className = "alexz-mod-picker-select";
    root.appendChild(nodeSelect);

    const moduleFilter = document.createElement("input");
    moduleFilter.type = "text";
    moduleFilter.className = "alexz-mod-picker-select";
    moduleFilter.placeholder = "Фильтр модулей (например: Inpaint-Crop)";
    root.appendChild(moduleFilter);

    const refreshLine = document.createElement("div");
    refreshLine.className = "alexz-mod-picker-refresh-line";
    root.appendChild(refreshLine);

    const diagnostics = document.createElement("div");
    diagnostics.className = "alexz-mod-picker-diag";
    diagnostics.textContent = "diag: waiting for sidebar sync...";
    root.appendChild(diagnostics);

    const loadDebugEnabled = () => {
        try {
            const raw = window.localStorage?.getItem(NODE_PICKER_DEBUG_STORAGE_KEY);
            if (raw === null || raw === undefined) {
                return Boolean(window[NODE_PICKER_DEBUG_KEY]);
            }
            return raw === "1" || raw === "true";
        } catch (_err) {
            return Boolean(window[NODE_PICKER_DEBUG_KEY]);
        }
    };
    const saveDebugEnabled = (enabled) => {
        try {
            if (enabled) {
                window.localStorage?.setItem(NODE_PICKER_DEBUG_STORAGE_KEY, "1");
            } else {
                window.localStorage?.removeItem(NODE_PICKER_DEBUG_STORAGE_KEY);
            }
        } catch (_err) {
            // Ignore storage failures and keep runtime flag only.
        }
    };
    const applyDebugUiState = () => {
        const enabled = Boolean(debugToggle.checked);
        window[NODE_PICKER_DEBUG_KEY] = enabled;
        diagnostics.style.display = enabled ? "" : "none";
    };
    debugToggle.checked = loadDebugEnabled();
    applyDebugUiState();
    debugToggle.addEventListener("change", () => {
        applyDebugUiState();
        saveDebugEnabled(Boolean(debugToggle.checked));
    });

    const help = document.createElement("div");
    help.className = "alexz-mod-picker-help";
    root.appendChild(help);

    const moduleInfo = document.createElement("div");
    root.appendChild(moduleInfo);

    const nodeList = document.createElement("div");
    root.appendChild(nodeList);

    const catalogByGroup = new Map();
    const moduleCatalogByGroup = new Map();
    const moduleCounts = new Map();
    const moduleOptions = new Map();
    const moduleBadges = new Map();
    const moduleNodeDiffs = new Map();
    const updatedModulesSession = new Set();
    let refreshPollToken = 0;
    let updatePollToken = 0;
    let customModulesNeedUpdate = 0;
    let actionBusy = false;
    let expandedModule = "";

    /**
     * Render ComfyUI update alert block when upstream update is available.
     */
    const renderComfyAlert = (info) => {
        const behind = Number(info?.behind);
        const status = String(info?.update_status || "unknown");
        const updatedBetweenRuns = Boolean(info?.updated_between_runs);
        comfyAlert.classList.remove("alexz-mod-picker-comfy-alert--ok");
        if (status !== "can_update" || !Number.isFinite(behind) || behind <= 0) {
            if (updatedBetweenRuns) {
                const prev = String(info?.startup_prev_commit_short || "unknown");
                const next = String(info?.startup_new_commit_short || "unknown");
                const at = info?.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
                comfyAlertText.textContent = `ComfyUI обновлен между запусками: ${prev} -> ${next}${at}.`;
                comfyUpdateBtn.style.display = "none";
                comfyAlert.style.display = "block";
                comfyAlert.classList.add("alexz-mod-picker-comfy-alert--ok");
                return;
            }
            comfyAlert.style.display = "none";
            comfyAlertText.textContent = "";
            comfyUpdateBtn.style.display = "none";
            return;
        }
        const branch = String(info?.branch || "unknown");
        const local = String(info?.installed_commit_short || "unknown");
        const remote = String(info?.remote_commit_short || "unknown");
        comfyAlertText.textContent = `ComfyUI требует обновления: branch=${branch}, behind=${behind}, local=${local}, remote=${remote}.`;
        comfyUpdateBtn.style.display = "";
        comfyUpdateBtn.disabled = actionBusy;
        comfyAlert.style.display = "block";
    };

    /**
     * Return node catalog entries for currently selected group.
     */
    const getNodesForSelectedGroup = () => {
        const group = groupSelect.value;
        return catalogByGroup.get(group) || [];
    };

    /**
     * Update the one-line status/progress text with optional color tone.
     */
    const setRefreshLine = (text, tone = "neutral") => {
        refreshLine.textContent = text || "";
        refreshLine.classList.remove("alexz-mod-picker-refresh-line--ok", "alexz-mod-picker-refresh-line--warn");
        if (tone === "ok") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--ok");
        } else if (tone === "warn") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--warn");
        }
    };
    /**
     * Render compact diagnostics block for tab-sync troubleshooting.
     */
    const setDiagnosticText = (diag) => {
        const lines = [
            `diag.ts=${new Date().toLocaleTimeString()}`,
            `diag.reason=${diag?.reason || "unknown"}`,
            `diag.active_tab=${diag?.activeTabId || "n/a"}`,
            `diag.last_clicked_tab=${diag?.lastClickedTabId || "n/a"}`,
            `diag.own_btn_found=${diag?.ownBtnFound ? "yes" : "no"}`,
            `diag.own_btn_selected=${diag?.ownBtnSelected === null ? "n/a" : (diag?.ownBtnSelected ? "yes" : "no")}`,
            `diag.root_display=${diag?.rootDisplay || "n/a"}`,
            `diag.child_nodes=${Number(diag?.childNodesCount || 0)}`,
            `diag.child_nodes_short=${diag?.childNodesShort || "n/a"}`,
        ];
        diagnostics.textContent = lines.join("\n");
    };
    if (Boolean(window[NODE_PICKER_SIDEBAR_SYNC_KEY])) {
        bindContainerOwnershipSync(container, root, setDiagnosticText);
    } else {
        bindModuleNodesTabRelay({
            app,
            root,
            sidebarTabId: SIDEBAR_TAB_ID,
            onDiag: setDiagnosticText,
        });
        setDiagnosticText({
            reason: "sync_disabled",
            activeTabId: "n/a",
            lastClickedTabId: "n/a",
            ownBtnFound: false,
            ownBtnSelected: null,
            rootDisplay: root.style.display || "",
            childNodesCount: 1,
            childNodesShort: "ROOT",
        });
    }

    /**
     * Replace help area with plain status/help text.
     */
    const setHelpText = (text) => {
        help.innerHTML = "";
        help.textContent = text || "";
    };

    /**
     * Render expanded-module help summary with insertion hints and legend.
     */
    const setHelpModuleSummary = (moduleName, nodeCount) => {
        help.innerHTML = "";

        const main = document.createElement("div");
        main.className = "alexz-mod-picker-help-main";
        main.append("Модуль ");
        const moduleStrong = document.createElement("strong");
        moduleStrong.textContent = String(moduleName || "unknown");
        main.appendChild(moduleStrong);
        main.append(": нод ");
        const countStrong = document.createElement("strong");
        countStrong.textContent = String(Math.max(0, Number(nodeCount) || 0));
        main.appendChild(countStrong);
        main.append(".");
        help.appendChild(main);

        const hint1 = document.createElement("div");
        hint1.className = "alexz-mod-picker-help-hint";
        hint1.textContent = "Кликните ноду для вставки в граф.";
        help.appendChild(hint1);

        const hint2 = document.createElement("div");
        hint2.className = "alexz-mod-picker-help-hint";
        hint2.textContent = `Метки модулей: ${MODULE_MARK_UPDATED} обновлен между запусками, ${MODULE_MARK_REMOTE_UPDATE} доступно обновление.`;
        help.appendChild(hint2);

        const hint3 = document.createElement("div");
        hint3.className = "alexz-mod-picker-help-hint";
        hint3.textContent = "Рамка ноды: красная = новая, зеленая = обновленная.";
        help.appendChild(hint3);
    };

    /**
     * Render collapsed-module hint shown before node list expansion.
     */
    const setHelpModuleCardHint = (moduleName, nodeCount) => {
        help.innerHTML = "";

        const main = document.createElement("div");
        main.className = "alexz-mod-picker-help-main";
        main.append("Модуль ");
        const moduleStrong = document.createElement("strong");
        moduleStrong.textContent = String(moduleName || "unknown");
        main.appendChild(moduleStrong);
        main.append(": нод ");
        const countStrong = document.createElement("strong");
        countStrong.textContent = String(Math.max(0, Number(nodeCount) || 0));
        main.appendChild(countStrong);
        main.append(".");
        help.appendChild(main);

        const hint = document.createElement("div");
        hint.className = "alexz-mod-picker-help-hint";
        hint.textContent = "Кликните карточку модуля, чтобы показать список нод.";
        help.appendChild(hint);
    };

    /**
     * Convert backend refresh status payload into a one-line progress message.
     */
    const formatRefreshLine = (refresh) => {
        const phase = String(refresh?.phase || "");
        const current = Number(refresh?.current || 0);
        const total = Number(refresh?.total || 0);
        const remaining = Number(refresh?.remaining || 0);
        const modulesNeedUpdate = Number(refresh?.modules_need_update || 0);
        const moduleName = String(refresh?.module || "");
        const error = String(refresh?.error || "");

        if (phase === "sync") {
            if (total > 0) {
                const modulePart = moduleName ? ` (${moduleName})` : "";
                return { text: `Обновление статусов модулей: ${current}/${total}, осталось ${remaining}${modulePart}`, tone: "neutral" };
            }
            return { text: "Обновление статусов модулей: подготовка...", tone: "neutral" };
        }
        if (phase === "snapshots") {
            return { text: "Обновление статусов модулей: пересчет...", tone: "neutral" };
        }
        if (phase === "done") {
            const count = Number.isFinite(modulesNeedUpdate) ? Math.max(0, modulesNeedUpdate) : 0;
            if (count > 0) {
                return { text: `${count} модулей требуют обновления`, tone: "warn" };
            }
            return { text: "обновления не требуются", tone: "ok" };
        }
        if (phase === "error") {
            return { text: `Обновление статусов модулей: ошибка${error ? ` (${error})` : ""}.`, tone: "warn" };
        }
        return { text: "Обновление статусов модулей: запуск...", tone: "neutral" };
    };

    /**
     * Poll refresh status endpoint until job completes or fails.
     */
    const pollRefreshProgress = async () => {
        const token = ++refreshPollToken;
        while (token === refreshPollToken) {
            let payload;
            try {
                payload = await fetchModuleRefreshStatus();
            } catch (err) {
                setRefreshLine(`Обновление статусов модулей: ошибка статуса (${String(err)}).`, "warn");
                return false;
            }
            const refresh = payload?.refresh || {};
            const line = formatRefreshLine(refresh);
            setRefreshLine(line.text, line.tone);
            if (!refresh?.running) {
                return refresh?.phase !== "error";
            }
            await new Promise((resolve) => setTimeout(resolve, 400));
        }
        return false;
    };

    /**
     * Enable/disable actionable UI controls during long-running operations.
     */
    const setActionBusy = (busy) => {
        actionBusy = Boolean(busy);
        refreshBtn.disabled = actionBusy;
        comfyInfoBtn.disabled = actionBusy;
        updateAllBtn.disabled = actionBusy;
        comfyUpdateBtn.disabled = actionBusy || comfyUpdateBtn.style.display === "none";
        for (const btn of moduleInfo.querySelectorAll(".alexz-mod-picker-action-row .alexz-mod-picker-btn-small")) {
            btn.disabled = actionBusy;
        }
    };

    /**
     * Toggle visibility and label of the global custom-nodes update button.
     */
    const syncUpdateAllButton = () => {
        const show = groupSelect.value === "custom" && customModulesNeedUpdate > 0;
        if (!show) {
            updateAllBtn.style.display = "none";
            return;
        }
        updateAllBtn.style.display = "";
        updateAllBtn.textContent = `Update all custom_nodes (${customModulesNeedUpdate})`;
    };

    /**
     * Convert module-update status payload into a one-line progress/result message.
     */
    const formatUpdateLine = (update) => {
        const scope = String(update?.scope || "");
        const phase = String(update?.phase || "");
        const current = Number(update?.current || 0);
        const total = Number(update?.total || 0);
        const remaining = Number(update?.remaining || 0);
        const moduleName = String(update?.module || "");
        const error = String(update?.error || "");
        const updated = Number(update?.updated || 0);
        const failed = Number(update?.failed || 0);
        const requirementsChanged = Boolean(update?.requirements_changed);
        const reqList = Array.isArray(update?.requirements_modules) ? update.requirements_modules : [];

        if (phase === "update") {
            const modulePart = moduleName ? ` (${moduleName})` : "";
            if (total > 0) {
                return { text: `Обновление модулей: ${current}/${total}, осталось ${remaining}${modulePart}`, tone: "neutral" };
            }
            return { text: "Обновление модулей: запуск...", tone: "neutral" };
        }
        if (phase === "done") {
            if (scope === "comfyui") {
                if (failed > 0) {
                    return { text: "Обновление ComfyUI завершено с ошибкой.", tone: "warn" };
                }
                if (updated > 0 && requirementsChanged) {
                    return { text: "ComfyUI обновлен, требуется обновить dependencies.", tone: "warn" };
                }
                if (updated > 0) {
                    return { text: "ComfyUI обновлен.", tone: "ok" };
                }
                return { text: "ComfyUI уже актуален.", tone: "ok" };
            }
            if (total <= 0) {
                return { text: "Обновления не найдены.", tone: "ok" };
            }
            if (failed > 0) {
                return { text: `Обновление завершено: updated=${updated}, failed=${failed}.`, tone: "warn" };
            }
            if (reqList.length > 0) {
                return { text: `Обновление завершено: ${updated} модулей обновлено, требуется обновить dependencies.`, tone: "warn" };
            }
            return { text: `Обновление завершено: ${updated} модулей обновлено.`, tone: "ok" };
        }
        if (phase === "error") {
            return { text: `Обновление модулей: ошибка${error ? ` (${error})` : ""}.`, tone: "warn" };
        }
        return { text: "Обновление модулей: подготовка...", tone: "neutral" };
    };

    /**
     * Poll update status endpoint until module update job finishes.
     */
    const pollUpdateProgress = async () => {
        const token = ++updatePollToken;
        while (token === updatePollToken) {
            let payload;
            try {
                payload = await fetchModuleUpdateStatus();
            } catch (err) {
                setRefreshLine(`Обновление модулей: ошибка статуса (${String(err)}).`, "warn");
                return null;
            }
            const update = payload?.update || {};
            const line = formatUpdateLine(update);
            setRefreshLine(line.text, line.tone);
            if (!update?.running) {
                return update;
            }
            await new Promise((resolve) => setTimeout(resolve, 450));
        }
        return null;
    };

    /**
     * Ask user to install changed requirements.txt files and execute pip install.
     */
    const maybeInstallChangedRequirements = async (update) => {
        const scope = String(update?.scope || "");
        if (scope === "comfyui") {
            if (!Boolean(update?.requirements_changed)) {
                return;
            }
            const answer = window.confirm(
                "В ComfyUI изменился requirements.txt.\n" +
                "Обновить зависимости через pip в текущем окружении ComfyUI?"
            );
            if (!answer) {
                setRefreshLine("ComfyUI requirements.txt изменился, установка зависимостей пропущена.", "warn");
                return;
            }
            setRefreshLine("Установка зависимостей ComfyUI (pip) ...", "neutral");
            const install = await installComfyUIRequirements();
            if (String(install?.status || "") !== "installed") {
                setRefreshLine("Установка зависимостей ComfyUI завершена с ошибкой.", "warn");
                return;
            }
            setRefreshLine("Зависимости ComfyUI обновлены.", "ok");
            return;
        }

        const modules = Array.isArray(update?.requirements_modules) ? update.requirements_modules : [];
        if (!modules.length) {
            return;
        }
        const answer = window.confirm(
            `В модуле(ях) ${modules.join(", ")} изменился requirements.txt.\n` +
            "Обновить зависимости через pip в текущем окружении ComfyUI?"
        );
        if (!answer) {
            setRefreshLine("requirements.txt изменился, установка зависимостей пропущена.", "warn");
            return;
        }
        setRefreshLine("Установка зависимостей (pip) ...", "neutral");
        const install = await installModuleRequirements(modules);
        const failed = Number(install?.failed || 0);
        const installed = Number(install?.installed || 0);
        if (failed > 0) {
            setRefreshLine(`Установка зависимостей завершена с ошибками: ok=${installed}, failed=${failed}.`, "warn");
            return;
        }
        setRefreshLine(`Зависимости обновлены: ${installed} модулей.`, "ok");
    };

    /**
     * Run update flow (backend job + polling + optional requirements install)
     * and then refresh catalog/module state in UI.
     */
    const runModuleUpdate = async (scope, moduleName) => {
        setActionBusy(true);
        try {
            setRefreshLine("Обновление модулей: запуск...", "neutral");
            await startModuleUpdate(scope, moduleName);
            const update = await pollUpdateProgress();
            if (!update) {
                return;
            }
            const currentGroup = String(groupSelect.value || "").trim();
            const currentModule = String(nodeSelect.value || "").trim();
            const updatedNow = Array.isArray(update?.results)
                ? update.results.filter((item) => String(item?.status || "") === "updated")
                : [];
            for (const item of updatedNow) {
                const mod = String(item?.module || "").trim();
                if (mod) {
                    updatedModulesSession.add(mod);
                }
            }
            if (String(update.phase || "") === "done") {
                await maybeInstallChangedRequirements(update);
            }
            let preferredGroup = currentGroup;
            let preferredModule = currentModule;
            let autoExpandModule = "";
            if (scope === "single") {
                preferredGroup = "custom";
                preferredModule = String(moduleName || currentModule || "").trim();
                if (updatedNow.some((item) => String(item?.module || "").trim() === preferredModule)) {
                    autoExpandModule = preferredModule;
                }
            } else if (scope === "all" && currentGroup === "custom" && currentModule) {
                preferredGroup = "custom";
                preferredModule = currentModule;
                if (updatedModulesSession.has(currentModule)) {
                    autoExpandModule = currentModule;
                }
            }
            await loadCatalog({ preferredGroup, preferredModule, autoExpandModule });
            await loadModuleInfo();
        } catch (err) {
            setRefreshLine(`Обновление модулей: ошибка (${String(err)}).`, "warn");
        } finally {
            setActionBusy(false);
            syncUpdateAllButton();
        }
    };

    /**
     * Refresh module select option text for one module after badge updates.
     */
    const setModuleOptionText = (moduleName) => {
        const option = moduleOptions.get(moduleName);
        if (!option) {
            return;
        }
        const count = moduleCounts.get(moduleName) || 0;
        const badges = moduleBadges.get(moduleName) || null;
        option.textContent = formatModuleOption(moduleName, count, badges);
    };

    /**
     * Cache node-level diff markers (new/updated) for selected module.
     */
    const setModuleNodeDiffs = (moduleName, info) => {
        const newNodes = Array.isArray(info?.new_nodes_between_runs) ? info.new_nodes_between_runs : [];
        const updatedNodes = Array.isArray(info?.updated_nodes_between_runs) ? info.updated_nodes_between_runs : [];
        moduleNodeDiffs.set(moduleName, {
            newNodes: new Set(newNodes),
            updatedNodes: new Set(updatedNodes),
            markAllUpdated: Boolean(info?.new_module_between_runs),
        });
    };

    /**
     * Populate module selector for current group with filtering and badge placeholders.
     */
    const fillModuleSelect = (options = {}) => {
        const preferredModule = String(options?.preferredModule || "").trim();
        const autoExpandModule = String(options?.autoExpandModule || "").trim();
        const nodes = getNodesForSelectedGroup();
        const selectedGroup = groupSelect.value;
        const moduleEntries = moduleCatalogByGroup.get(selectedGroup) || [];
        const filterValue = (moduleFilter.value || "").trim().toLowerCase();
        const previousSelectedModule = String(nodeSelect.value || "").trim();
        moduleCounts.clear();
        moduleOptions.clear();
        moduleBadges.clear();
        moduleNodeDiffs.clear();
        nodeSelect.innerHTML = "";
        const grouped = new Map();
        for (const node of nodes) {
            const moduleName = node.module || "unknown";
            if (!grouped.has(moduleName)) {
                grouped.set(moduleName, []);
            }
            grouped.get(moduleName).push(node);
        }
        let modules = [];
        if (moduleEntries.length) {
            modules = moduleEntries
                .map((entry) => String(entry?.module || "unknown"))
                .sort((a, b) => a.localeCompare(b));
        } else {
            modules = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
        }
        if (filterValue) {
            modules = modules.filter((name) => name.toLowerCase().includes(filterValue));
        }
        if (modules.length === 0) {
            const empty = document.createElement("option");
            empty.value = "-1";
            empty.textContent = filterValue ? "Нет модулей по фильтру" : "В этой группе нет модулей";
            nodeSelect.appendChild(empty);
            nodeSelect.value = "-1";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
            setHelpText(filterValue
                ? `Нет модулей по фильтру: "${moduleFilter.value}".`
                : "Модули не найдены для выбранной группы.");
            syncUpdateAllButton();
            return;
        }
        const countMap = new Map();
        const entryMap = new Map();
        for (const entry of moduleEntries) {
            const moduleName = String(entry?.module || "unknown");
            countMap.set(moduleName, Number(entry?.count) || 0);
            entryMap.set(moduleName, entry || {});
        }
        for (const moduleName of modules) {
            const opt = document.createElement("option");
            opt.value = moduleName;
            const count = countMap.has(moduleName)
                ? (countMap.get(moduleName) || 0)
                : (grouped.get(moduleName) || []).length;
            moduleCounts.set(moduleName, count);
            moduleOptions.set(moduleName, opt);
            const entry = entryMap.get(moduleName) || null;
            const badges = moduleBadgesFromModuleEntry(entry);
            if (badges.updatedBetweenRuns || badges.hasRemoteUpdate) {
                moduleBadges.set(moduleName, badges);
            }
            opt.textContent = formatModuleOption(moduleName, count, badges);
            nodeSelect.appendChild(opt);
        }
        if (preferredModule && modules.includes(preferredModule)) {
            nodeSelect.value = preferredModule;
        } else if (previousSelectedModule && modules.includes(previousSelectedModule)) {
            nodeSelect.value = previousSelectedModule;
        } else if (modules.includes(DEFAULT_MODULE)) {
            nodeSelect.value = DEFAULT_MODULE;
        } else {
            nodeSelect.value = modules[0];
        }
        if (autoExpandModule && nodeSelect.value === autoExpandModule) {
            expandedModule = autoExpandModule;
        } else {
            expandedModule = "";
            nodeList.innerHTML = "";
        }
        renderNodeList();
        loadModuleInfo();
        syncUpdateAllButton();
    };

    /**
     * Populate top-level group selector and propagate selection to module list.
     */
    const fillGroupSelect = (groups, options = {}) => {
        const preferredGroup = String(options?.preferredGroup || "").trim();
        const preferredModule = String(options?.preferredModule || "").trim();
        const autoExpandModule = String(options?.autoExpandModule || "").trim();
        const previousGroup = String(groupSelect.value || "").trim();
        groupSelect.innerHTML = "";
        moduleCatalogByGroup.clear();
        groups.forEach((group) => {
            const opt = document.createElement("option");
            opt.value = group.id;
            const label = GROUP_LABELS[group.id] || group.title || group.id;
            opt.textContent = `${label} (${group.count})`;
            groupSelect.appendChild(opt);
            catalogByGroup.set(group.id, group.nodes || []);
            moduleCatalogByGroup.set(group.id, group.modules || []);
        });

        if (preferredGroup && catalogByGroup.has(preferredGroup)) {
            groupSelect.value = preferredGroup;
        } else if (previousGroup && catalogByGroup.has(previousGroup)) {
            groupSelect.value = previousGroup;
        } else if (catalogByGroup.has("custom")) {
            groupSelect.value = "custom";
        } else if (groups.length > 0) {
            groupSelect.value = groups[0].id;
        }
        fillModuleSelect({ preferredModule, autoExpandModule });
    };

    /**
     * Render node cards for currently selected module and bind insertion actions.
     */
    const renderNodeList = () => {
        nodeList.innerHTML = "";
        const selectedModule = nodeSelect.value;
        const nodes = getNodesForSelectedGroup().filter(
            (node) => (node.module || "unknown") === selectedModule
        );
        if (selectedModule === "-1") {
            setHelpText("Выберите модуль, чтобы увидеть список нод.");
            return;
        }
        if (!nodes.length) {
            setHelpText(`Модуль ${selectedModule}: загруженных нод не найдено (возможно, модуль не загрузился).`);
            return;
        }
        if (expandedModule !== selectedModule) {
            setHelpModuleCardHint(selectedModule, nodes.length);
            return;
        }

        setHelpModuleSummary(selectedModule, nodes.length);
        const nodeDiff = moduleNodeDiffs.get(selectedModule) || {
            newNodes: new Set(),
            updatedNodes: new Set(),
            markAllUpdated: false,
        };

        const groupEl = document.createElement("div");
        groupEl.className = "alexz-mod-picker-group";

        const groupTitle = document.createElement("div");
        groupTitle.className = "alexz-mod-picker-group-title";
        groupTitle.textContent = `${selectedModule} (${nodes.length})`;
        groupEl.appendChild(groupTitle);

        for (const nodeInfo of nodes) {
            const item = document.createElement("button");
            item.type = "button";
            item.className = "alexz-mod-picker-node";
            if (nodeDiff.markAllUpdated) {
                item.classList.add("alexz-mod-picker-node--updated");
            } else if (nodeDiff.newNodes.has(nodeInfo.node_name)) {
                item.classList.add("alexz-mod-picker-node--new");
            } else if (nodeDiff.updatedNodes.has(nodeInfo.node_name)) {
                item.classList.add("alexz-mod-picker-node--updated");
            }
            item.onclick = () => {
                const node = createNodeByInfo(nodeInfo);
                if (!node) {
                    setHelpText(`Не удалось создать ноду: ${nodeInfo.display_name}`);
                    return;
                }
                app.graph.add(node);
                centerNode(node);
                app.canvas?.selectNode?.(node, false);
                app.graph.setDirtyCanvas(true, true);
                setHelpText(`Вставлена в граф: ${nodeInfo.display_name}`);
            };

            const nameEl = document.createElement("div");
            nameEl.className = "alexz-mod-picker-node-name";
            nameEl.textContent = nodeInfo.display_name;
            item.appendChild(nameEl);

            const descEl = document.createElement("div");
            descEl.className = "alexz-mod-picker-node-desc";
            descEl.textContent = `${nodeInfo.annotation} [${nodeInfo.category || "unknown"}]`;
            item.appendChild(descEl);

            groupEl.appendChild(item);
        }
        nodeList.appendChild(groupEl);
    };

    /**
     * Render module metadata card, status rows, and per-module action buttons.
     */
    const renderModuleInfo = (info) => {
        moduleInfo.innerHTML = "";
        if (!info || nodeSelect.value === "-1") {
            return;
        }

        const card = document.createElement("div");
        card.className = "alexz-mod-picker-module-card";
        const selectedModule = nodeSelect.value;
        const nodeCount = moduleCounts.get(selectedModule) || 0;
        if (
            updatedModulesSession.has(selectedModule)
            || Boolean(info?.updated_between_runs)
            || Boolean(info?.new_module_between_runs)
        ) {
            card.classList.add("alexz-mod-picker-module-card--updated");
        }
        if (selectedModule !== "-1" && nodeCount > 0) {
            card.classList.add("alexz-mod-picker-module-card--clickable");
            card.title = "Кликните, чтобы показать список нод";
            card.onclick = () => {
                expandedModule = selectedModule;
                renderNodeList();
            };
        }

        const titleEl = document.createElement("div");
        titleEl.className = "alexz-mod-picker-module-title";
        titleEl.textContent = info.title || info.module || nodeSelect.value;
        card.appendChild(titleEl);

        const authorEl = document.createElement("div");
        authorEl.className = "alexz-mod-picker-module-meta";
        if (info.author && info.owner_url) {
            authorEl.innerHTML = `Owner: <a href="${info.owner_url}" target="_blank" rel="noopener noreferrer">${info.author}</a>`;
            const ownerLink = authorEl.querySelector("a");
            ownerLink?.addEventListener("click", (event) => event.stopPropagation());
        } else {
            authorEl.textContent = `Owner: ${info.author || "unknown"}`;
        }
        card.appendChild(authorEl);

        if (info.description) {
            const descEl = document.createElement("div");
            descEl.className = "alexz-mod-picker-module-desc";
            descEl.textContent = info.description;
            card.appendChild(descEl);
        }

        const hasInstalledMeta = Boolean(info.installed_updated_at || info.installed_commit_short);
        if (hasInstalledMeta) {
            const installedRow = document.createElement("div");
            installedRow.className = "alexz-mod-picker-module-row";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Installed:";
            const valueEl = document.createElement("span");
            valueEl.textContent = `${info.installed_commit_short ? `${info.installed_commit_short} · ` : ""}${fmtDate(info.installed_updated_at)}`;
            installedRow.appendChild(labelEl);
            installedRow.appendChild(valueEl);
            card.appendChild(installedRow);
        }

        if (info.remote_updated_at) {
            const remoteRow = document.createElement("div");
            remoteRow.className = "alexz-mod-picker-module-row";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Remote updated:";
            const valueEl = document.createElement("span");
            valueEl.textContent = fmtDate(info.remote_updated_at);
            remoteRow.appendChild(labelEl);
            remoteRow.appendChild(valueEl);
            card.appendChild(remoteRow);
        }

        if (String(info.group || "") === "custom") {
            const statusRow = document.createElement("div");
            statusRow.className = "alexz-mod-picker-module-row";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Status:";
            const valueEl = document.createElement("span");
            const updateStatus = String(info.update_status || "unknown");
            if (updateStatus === "can_update") {
                statusRow.classList.add("warn");
                valueEl.textContent = "модуль требует обновления";
            } else if (updateStatus === "up_to_date") {
                statusRow.classList.add("ok");
                valueEl.textContent = "модуль актуален";
            } else {
                valueEl.textContent = "статус неизвестен";
            }
            statusRow.appendChild(labelEl);
            statusRow.appendChild(valueEl);
            card.appendChild(statusRow);

            const actionRow = document.createElement("div");
            actionRow.className = "alexz-mod-picker-action-row";

            const refreshInfoBtn = document.createElement("button");
            refreshInfoBtn.type = "button";
            refreshInfoBtn.className = "alexz-mod-picker-btn-small";
            refreshInfoBtn.textContent = "Обновить информацию о модуле";
            refreshInfoBtn.disabled = actionBusy;
            refreshInfoBtn.onclick = async (event) => {
                event.stopPropagation();
                if (actionBusy) {
                    return;
                }
                setActionBusy(true);
                setRefreshLine("Обновление информации о модуле...", "neutral");
                try {
                    await loadModuleInfo({ forceRefresh: true, syncUpstream: true, throwOnError: true });
                    setRefreshLine("Информация о модуле обновлена.", "ok");
                } catch (err) {
                    setRefreshLine(`Ошибка обновления информации модуля: ${String(err)}`, "warn");
                } finally {
                    setActionBusy(false);
                    syncUpdateAllButton();
                }
            };
            actionRow.appendChild(refreshInfoBtn);

            if (updateStatus === "can_update") {
                const updateBtn = document.createElement("button");
                updateBtn.type = "button";
                updateBtn.className = "alexz-mod-picker-btn-small";
                updateBtn.textContent = "Update module";
                updateBtn.disabled = actionBusy;
                updateBtn.onclick = async (event) => {
                    event.stopPropagation();
                    if (actionBusy) {
                        return;
                    }
                    const moduleName = String(info.module || nodeSelect.value || "").trim();
                    if (!moduleName) {
                        return;
                    }
                    await runModuleUpdate("single", moduleName);
                };
                actionRow.appendChild(updateBtn);
            }
            card.appendChild(actionRow);
        }
        if (String(info.group || "") !== "custom") {
            const actionRow = document.createElement("div");
            actionRow.className = "alexz-mod-picker-action-row";
            const refreshInfoBtn = document.createElement("button");
            refreshInfoBtn.type = "button";
            refreshInfoBtn.className = "alexz-mod-picker-btn-small";
            refreshInfoBtn.textContent = "Обновить информацию о модуле";
            refreshInfoBtn.disabled = actionBusy;
            refreshInfoBtn.onclick = async (event) => {
                event.stopPropagation();
                if (actionBusy) {
                    return;
                }
                setActionBusy(true);
                setRefreshLine("Обновление информации о модуле...", "neutral");
                try {
                    await loadModuleInfo({ forceRefresh: true, syncUpstream: false, throwOnError: true });
                    setRefreshLine("Информация о модуле обновлена.", "ok");
                } catch (err) {
                    setRefreshLine(`Ошибка обновления информации модуля: ${String(err)}`, "warn");
                } finally {
                    setActionBusy(false);
                    syncUpdateAllButton();
                }
            };
            actionRow.appendChild(refreshInfoBtn);
            card.appendChild(actionRow);
        }

        if (info.new_module_between_runs) {
            const newRow = document.createElement("div");
            newRow.className = "alexz-mod-picker-module-row notice";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Detected between runs:";
            const valueEl = document.createElement("span");
            valueEl.textContent = "new module";
            newRow.appendChild(labelEl);
            newRow.appendChild(valueEl);
            card.appendChild(newRow);
        }

        if (info.updated_between_runs) {
            const updateRow = document.createElement("div");
            updateRow.className = "alexz-mod-picker-module-row notice";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Updated between runs:";
            const valueEl = document.createElement("span");
            const prev = info.startup_prev_commit_short || "unknown";
            const next = info.startup_new_commit_short || "unknown";
            const at = info.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
            if (info.startup_prev_commit_short || info.startup_new_commit_short) {
                valueEl.textContent = `${prev} -> ${next}${at}`;
            } else {
                valueEl.textContent = `local changes detected${at}`;
            }
            updateRow.appendChild(labelEl);
            updateRow.appendChild(valueEl);
            card.appendChild(updateRow);
        }

        const updatedNodes = Array.isArray(info.updated_nodes_between_runs)
            ? info.updated_nodes_between_runs.filter(Boolean)
            : [];
        if (updatedNodes.length) {
            const updatedLine = document.createElement("div");
            updatedLine.className = "alexz-mod-picker-module-note";
            updatedLine.textContent = `Обновлены ноды: ${updatedNodes.join(", ")}`;
            card.appendChild(updatedLine);
        }

        const newNodes = Array.isArray(info.new_nodes_between_runs)
            ? info.new_nodes_between_runs.filter(Boolean)
            : [];
        if (newNodes.length) {
            const newLine = document.createElement("div");
            newLine.className = "alexz-mod-picker-module-note";
            newLine.textContent = `Добавлены ноды: ${newNodes.join(", ")}`;
            card.appendChild(newLine);
        }

        moduleInfo.appendChild(card);
    };

    /**
     * Load and render module info for currently selected group/module.
     */
    const loadModuleInfo = async (options = {}) => {
        const selectedModule = nodeSelect.value;
        const selectedGroup = groupSelect.value;
        const forceRefresh = Boolean(options?.forceRefresh);
        const syncUpstream = Boolean(options?.syncUpstream);
        const throwOnError = Boolean(options?.throwOnError);
        if (!selectedModule || selectedModule === "-1") {
            moduleInfo.innerHTML = "";
            return;
        }
        try {
            const payload = await fetchModuleInfo(selectedGroup, selectedModule, {
                forceRefresh,
                syncUpstream,
            });
            if (nodeSelect.value !== selectedModule || groupSelect.value !== selectedGroup) {
                return;
            }
            const info = payload?.info || null;
            renderModuleInfo(info);
            if (info) {
                const badges = moduleBadgesFromInfo(info);
                if (badges.updatedBetweenRuns || badges.hasRemoteUpdate) {
                    moduleBadges.set(selectedModule, badges);
                } else {
                    moduleBadges.delete(selectedModule);
                }
                setModuleNodeDiffs(selectedModule, info);
                setModuleOptionText(selectedModule);
                renderNodeList();
            }
        } catch (err) {
            moduleInfo.innerHTML = "";
            if (throwOnError) {
                throw err;
            }
        }
    };

    /**
     * Load full node catalog from backend and refresh picker UI state.
     */
    const loadCatalog = async (options = {}) => {
        const preferredGroup = String(options?.preferredGroup || "").trim();
        const preferredModule = String(options?.preferredModule || "").trim();
        const autoExpandModule = String(options?.autoExpandModule || "").trim();
        setHelpText("Загрузка списка нод...");
        try {
            const payload = await fetchNodeCatalog();
            catalogByGroup.clear();
            const groups = payload?.groups || [];
            customModulesNeedUpdate = Number(payload?.custom_modules_need_update || 0);
            renderComfyAlert(payload?.comfyui || null);
            fillGroupSelect(groups, { preferredGroup, preferredModule, autoExpandModule });
            const summary = groups
                .map((group) => {
                    const label = GROUP_LABELS[group.id] || group.title || group.id;
                    return `${label}=${group.count}`;
                })
                .join(", ");
            setHelpText(`Группы: ${summary}.`);
            syncUpdateAllButton();
        } catch (err) {
            setHelpText(`Ошибка загрузки: ${String(err)}`);
            comfyAlert.style.display = "none";
            comfyAlertText.textContent = "";
            comfyUpdateBtn.style.display = "none";
            groupSelect.innerHTML = "";
            nodeSelect.innerHTML = "";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
            syncUpdateAllButton();
        }
    };

    groupSelect.onchange = () => {
        fillModuleSelect();
        syncUpdateAllButton();
    };
    moduleFilter.oninput = () => fillModuleSelect();
    nodeSelect.onchange = () => {
        expandedModule = "";
        nodeList.innerHTML = "";
        loadModuleInfo();
    };
    updateAllBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        await runModuleUpdate("all", "");
    };
    comfyUpdateBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        await runModuleUpdate("comfyui", "");
    };
    comfyInfoBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        setActionBusy(true);
        setRefreshLine("Обновление информации о ComfyUI...", "neutral");
        try {
            const payload = await fetchComfyUIInfo(true);
            renderComfyAlert(payload?.comfyui || null);
            const status = String(payload?.comfyui?.update_status || "unknown");
            if (status === "can_update") {
                setRefreshLine("ComfyUI требует обновления.", "warn");
            } else if (status === "up_to_date") {
                setRefreshLine("ComfyUI актуален.", "ok");
            } else {
                setRefreshLine("Статус ComfyUI обновлен (не удалось определить необходимость обновления).", "neutral");
            }
        } catch (err) {
            setRefreshLine(`Ошибка обновления информации ComfyUI: ${String(err)}`, "warn");
        } finally {
            setActionBusy(false);
            syncUpdateAllButton();
        }
    };
    refreshBtn.onclick = async () => {
        setActionBusy(true);
        setRefreshLine("Обновление статусов модулей: запуск...", "neutral");
        try {
            await refreshModuleRuntimeState();
            const ok = await pollRefreshProgress();
            if (!ok) {
                setRefreshLine("Обновление статусов модулей: завершилось с ошибкой.", "warn");
            } else {
                try {
                    await acknowledgeAllModuleNovelty();
                } catch (err) {
                    setRefreshLine(`Статусы обновлены, но не удалось сбросить метки новизны: ${String(err)}`, "warn");
                }
            }
        } catch (err) {
            setRefreshLine(`Обновление статусов модулей: ошибка (${String(err)}).`, "warn");
        } finally {
            setActionBusy(false);
        }
        await loadCatalog();
    };

    loadCatalog();
}

/**
 * Tear down legacy container-ownership sync listeners and observers.
 */
function unbindContainerOwnershipSync() {
    const state = window[CONTAINER_SYNC_STATE_KEY];
    if (!state) {
        return;
    }
    if (state.containerObserver && typeof state.containerObserver.disconnect === "function") {
        state.containerObserver.disconnect();
    }
    if (state.sidebarObserver && typeof state.sidebarObserver.disconnect === "function") {
        state.sidebarObserver.disconnect();
    }
    if (state.onClick) {
        document.removeEventListener("click", state.onClick, true);
        document.removeEventListener("keyup", state.onClick, true);
    }
    if (state.onPointerDown) {
        document.removeEventListener("pointerdown", state.onPointerDown, true);
    }
    window[CONTAINER_SYNC_STATE_KEY] = null;
}

/**
 * Bind legacy sidebar/container sync logic used in compatibility mode.
 * Keeps picker root hidden when another sidebar tab owns the container.
 */
function bindContainerOwnershipSync(container, root, onDiag) {
    unbindContainerOwnershipSync();
    // Ensure root is visible when binding ownership sync
    root.style.display = "";
    let lastClickedTabId = "";
    let lastClickedTs = 0;
    let observedContainer = container;
    let syncTimerId = 0;
    const getActiveSidebarTabId = () => {
        const manager = app.extensionManager;
        const sidebar = manager?.sidebarTab || manager;
        const active = sidebar?.activeSidebarTabId ?? sidebar?.activeSidebarTab ?? "";
        return String(active || "");
    };
    const tryActivateSidebarTab = (tabId) => {
        if (!tabId) {
            return false;
        }
        const manager = app.extensionManager;
        const sidebar = manager?.sidebarTab || manager;
        const openFn = sidebar && typeof sidebar.activateSidebarTab === "function"
            ? sidebar.activateSidebarTab.bind(sidebar)
            : null;
        if (!openFn) {
            return false;
        }
        try {
            openFn(tabId);
            return true;
        } catch (_err) {
            return false;
        }
    };
    const extractTabIdFromButton = (buttonEl) => {
        if (!(buttonEl instanceof Element)) {
            return "";
        }
        for (const cls of Array.from(buttonEl.classList || [])) {
            if (cls.endsWith("-tab-button")) {
                return cls.slice(0, -"-tab-button".length);
            }
        }
        return "";
    };
    const resolveSidebarButtonFromEvent = (event) => {
        const isTabButtonLike = (el) => {
            if (!(el instanceof Element)) {
                return false;
            }
            for (const cls of Array.from(el.classList || [])) {
                if (cls.endsWith("-tab-button")) {
                    return true;
                }
            }
            return false;
        };
        const directTarget = event?.target;
        if (directTarget instanceof Element) {
            const byClosest = directTarget.closest(".side-bar-button, [class*='-tab-button']");
            if (byClosest) {
                return byClosest;
            }
        }
        if (typeof event?.composedPath === "function") {
            for (const item of event.composedPath()) {
                if (isTabButtonLike(item) || (item instanceof Element && item.classList?.contains("side-bar-button"))) {
                    return item;
                }
            }
        }
        return null;
    };
    const getLiveContainer = () => {
        const parent = root.parentElement;
        if (parent instanceof Element) {
            return parent;
        }
        return observedContainer;
    };
    const isOurTabSelected = () => {
        const ownBtn = document.querySelector(`.${SIDEBAR_TAB_ID}-tab-button`);
        if (!ownBtn) {
            return null;
        }
        return ownBtn.classList.contains("side-bar-button-selected");
    };
    const getSelectedSidebarTabIds = () => {
        const selectedButtons = Array.from(document.querySelectorAll(".side-bar-button-selected"));
        const ids = [];
        for (const btn of selectedButtons) {
            const id = extractTabIdFromButton(btn);
            if (id) {
                ids.push(id);
            }
        }
        return ids;
    };
    const describeChildNodes = (targetContainer) => {
        const out = [];
        for (const node of targetContainer.childNodes) {
            if (node === root) {
                out.push("ROOT");
                continue;
            }
            if (node.nodeType === Node.TEXT_NODE) {
                const trimmed = String(node.textContent || "").trim();
                out.push(trimmed ? `TXT:${trimmed.slice(0, 24)}` : "TXT:blank");
                continue;
            }
            if (node.nodeType === Node.COMMENT_NODE) {
                out.push("COMMENT");
                continue;
            }
            const tag = String(node.nodeName || "NODE");
            const cls = String(node.className || "").trim();
            out.push(cls ? `${tag}.${cls.split(/\s+/).slice(0, 2).join(".")}` : tag);
        }
        return out.slice(0, 10).join(" | ");
    };
    const hasForeignContent = (targetContainer) => {
        for (const node of targetContainer.childNodes) {
            if (node === root) {
                continue;
            }
            if (node.nodeType === Node.TEXT_NODE && !String(node.textContent || "").trim()) {
                continue;
            }
            return true;
        }
        return false;
    };
    let lastDiag = "";
    const isDebugLoggingEnabled = () => Boolean(window[NODE_PICKER_DEBUG_KEY]);
    const sync = () => {
        const liveContainer = getLiveContainer();
        if (!root.isConnected || !liveContainer?.isConnected) {
            onDiag?.({
                reason: !root.isConnected ? "root_disconnected" : "container_disconnected",
                activeTabId: getActiveSidebarTabId(),
                ownBtnFound: Boolean(document.querySelector(`.${SIDEBAR_TAB_ID}-tab-button`)),
                ownBtnSelected: isOurTabSelected(),
                rootDisplay: root.style.display || "",
                childNodesCount: Number(liveContainer?.childNodes?.length || 0),
                childNodesShort: liveContainer ? describeChildNodes(liveContainer) : "n/a",
                lastClickedTabId,
            });
            return;
        }
        if (liveContainer !== observedContainer) {
            if (containerObserver && typeof containerObserver.disconnect === "function") {
                containerObserver.disconnect();
            }
            observedContainer = liveContainer;
            containerObserver.observe(observedContainer, { childList: true, subtree: true });
        }
        const activeTabId = getActiveSidebarTabId();
        const selected = isOurTabSelected();
        const selectedTabIds = getSelectedSidebarTabIds();
        const otherTabSelected = selectedTabIds.some((x) => x && x !== SIDEBAR_TAB_ID);
        const foreign = hasForeignContent(liveContainer);
        let reason = "visible";
        if (otherTabSelected) {
            root.style.display = "none";
            reason = "other_button_selected";
        } else if (activeTabId && activeTabId !== SIDEBAR_TAB_ID) {
            root.style.display = "none";
            reason = "other_active_tab";
        } else if (selected === false) {
            root.style.display = "none";
            reason = "module_tab_not_selected";
        } else if (
            !activeTabId &&
            lastClickedTabId &&
            lastClickedTabId !== SIDEBAR_TAB_ID &&
            Date.now() - lastClickedTs < 1500
        ) {
            root.style.display = "none";
            reason = "recent_other_tab_click";
        } else if (foreign) {
            root.style.display = "none";
            reason = "foreign_content_in_container";
        } else {
            // Default: show if selected is true or null (button not found yet)
            root.style.display = "";
            reason = selected === true ? "visible" : "button_not_found_yet";
        }
        const diag = {
            reason,
            activeTabId,
            ownBtnFound: selected !== null,
            ownBtnSelected: selected,
            rootDisplay: root.style.display || "",
            childNodesCount: liveContainer.childNodes.length,
            childNodesShort: describeChildNodes(liveContainer),
            lastClickedTabId,
        };
        onDiag?.(diag);
        const sig = JSON.stringify(diag);
        if (sig !== lastDiag && isDebugLoggingEnabled()) {
            console.log("ALEXZ_tools Node Picker sync:", diag);
            lastDiag = sig;
        }
    };
    /**
     * Debounce sync execution to avoid excessive DOM work on rapid events.
     */
    const scheduleSync = (clickedTabId = "") => {
        if (syncTimerId) {
            return;
        }
        syncTimerId = window.setTimeout(() => {
            syncTimerId = 0;
            sync();
            // Workaround: some third-party tabs may not switch activeSidebarTabId
            // from this tab; enforce a single delayed activation attempt.
            const activeAfterClick = getActiveSidebarTabId();
            if (
                clickedTabId &&
                clickedTabId !== SIDEBAR_TAB_ID &&
                activeAfterClick === SIDEBAR_TAB_ID
            ) {
                tryActivateSidebarTab(clickedTabId);
            }
            window.setTimeout(sync, 70);
        }, 16);
    };
    const containerObserver = new MutationObserver(() => scheduleSync(""));
    containerObserver.observe(observedContainer, { childList: true });
    const sidebarObserver = null;
    /**
     * Track sidebar interactions and schedule ownership sync updates.
     */
    const onInteraction = (event) => {
        const button = resolveSidebarButtonFromEvent(event);
        if (!button) {
            return;
        }
        lastClickedTabId = extractTabIdFromButton(button);
        lastClickedTs = Date.now();
        if (!lastClickedTabId && !button.classList.contains(`${SIDEBAR_TAB_ID}-tab-button`)) {
            lastClickedTabId = "(unknown-other-tab)";
            root.style.display = "none";
        } else if (lastClickedTabId && lastClickedTabId !== SIDEBAR_TAB_ID) {
            root.style.display = "none";
        }
        scheduleSync(lastClickedTabId);
    };
    const onPointerDown = (event) => {
        onInteraction(event);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keyup", onInteraction, true);

    window[CONTAINER_SYNC_STATE_KEY] = {
        containerObserver,
        sidebarObserver,
        onClick: onInteraction,
        onPointerDown,
    };
    // Always show root on initialization - sync will determine visibility on next tick
    root.style.display = "";
    sync();
}

/**
 * Attach fallback button when Sidebar API is unavailable.
 */
function attachFallbackButton() {
    cleanupFallbackButtons();
    const button = document.createElement("button");
    button.id = FALLBACK_BUTTON_ID;
    button.type = "button";
    button.textContent = "Module Nodes";
    button.title = "Открыть подбор нод";
    button.onclick = () => {
        const manager = app.extensionManager;
        const sidebar = manager?.sidebarTab || manager;
        const openFn = sidebar && typeof sidebar.activateSidebarTab === "function"
            ? sidebar.activateSidebarTab.bind(sidebar)
            : null;
        if (!openFn) {
            button.textContent = "Sidebar API недоступен";
            return;
        }
        try {
            openFn(SIDEBAR_TAB_ID);
        } catch (_err) {
            button.textContent = "Sidebar API недоступен";
        }
    };

    const menuContainer = app.ui?.menuContainer;
    if (menuContainer) {
        button.style.width = "100%";
        button.style.order = 95;
        menuContainer.append(button);
        return;
    }

    button.className = "alexz-mod-picker-floating-btn";
    document.body.appendChild(button);
}

/**
 * Remove all fallback buttons previously created by this extension.
 */
function cleanupFallbackButtons() {
    const byId = document.getElementById(FALLBACK_BUTTON_ID);
    if (byId && byId.parentNode) {
        byId.parentNode.removeChild(byId);
    }
    for (const el of document.querySelectorAll(".alexz-mod-picker-floating-btn")) {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    }
}

if (!window[MODULE_PICKER_GUARD_KEY]) {
    window[MODULE_PICKER_GUARD_KEY] = true;
    app.registerExtension({
        name: EXT_NAME,
        setup() {
            injectStyles();
            cleanupFallbackButtons();

            if (app.extensionManager && typeof app.extensionManager.registerSidebarTab === "function") {
                app.extensionManager.registerSidebarTab({
                    id: SIDEBAR_TAB_ID,
                    icon: "pi pi-th-large",
                    title: "Module Nodes",
                    tooltip: "Выбор и вставка нод по группам Core/Custom",
                    type: "custom",
                    render: (container) => {
                        renderPicker(container);
                    },
                });
                return;
            }

            attachFallbackButton();
        },
    });
}
