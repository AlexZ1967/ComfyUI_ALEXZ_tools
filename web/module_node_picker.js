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
import {
    bindModuleNodesTabRelay,
    unbindModuleNodesTabRelay,
} from "./module_node_picker_tab_relay.js";
import {
    fetchNodeCatalog,
    fetchModuleInfo,
    fetchComfyUIInfo,
    refreshModuleRuntimeState,
    fetchModuleRefreshStatus,
    acknowledgeAllModuleNovelty,
    startModuleUpdate,
    fetchModuleUpdateStatus,
    installModuleRequirements,
    installComfyUIRequirements,
} from "./api/module_node_picker_api.js";
import {
    fmtDate,
    moduleBadgesFromInfo,
    moduleBadgesFromModuleEntry,
    formatModuleOption,
} from "./ui/module_node_picker_formatters.js";
import {
    formatRefreshLine,
    formatUpdateLine,
} from "./ui/module_node_picker_status.js";
import {
    renderHelpText,
    renderHelpModuleSummary,
    renderHelpModuleCardHint,
} from "./ui/module_node_picker_help.js";
import { createModuleNodePickerStore } from "./state/store.js";
import { createModuleDiagnosticsLogger } from "./diagnostics/logger.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const SIDEBAR_TAB_ID = "alexz-module-nodes";
const MODULE_PICKER_GUARD_KEY = "__alexz_module_node_picker_registered__";
const FALLBACK_BUTTON_ID = "alexz-module-nodes-fallback-btn";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const CONTAINER_SYNC_STATE_KEY = "__alexz_module_nodes_container_sync_state__";
const NODE_PICKER_DEBUG_KEY = "__alexz_module_picker_debug__";
const NODE_PICKER_DEBUG_STORAGE_KEY = "alexz_module_picker_debug";
const NODE_PICKER_SELECTED_GROUP_STORAGE_KEY = "alexz_module_picker_selected_group";
const NODE_PICKER_SELECTED_MODULE_STORAGE_KEY = "alexz_module_picker_selected_module";
const COMFYUI_CHECK_MODE_STORAGE_KEY = "alexz_comfyui_check_mode";
const NODE_PICKER_SIDEBAR_SYNC_KEY = "__alexz_module_picker_sidebar_sync__";
const GROUP_LABELS = {
    core: "Core_Nodes",
    core_extras: "Core_Extras_Nodes",
    api: "API_Nodes",
    custom: "Custom_Nodes",
};
const COMFY_GROUP_ORDER = ["core", "core_extras", "api"];
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
    .alexz-mod-picker-title {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.95;
        margin-right: auto;
    }
    .alexz-mod-picker-head-right {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .alexz-mod-picker-debug-card {
        border: 1px dashed var(--border-color, #555);
        border-radius: 7px;
        padding: 6px;
        display: none;
        background: var(--comfy-input-bg, rgba(255,255,255,0.02));
    }
    .alexz-mod-picker-debug-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    .alexz-mod-picker-debug-title {
        font-size: 11px;
        font-weight: 700;
        opacity: 0.92;
    }
    .alexz-mod-picker-divider {
        border-top: 1px solid var(--border-color, #444);
        margin: 2px 0;
    }
    .alexz-mod-picker-update-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .alexz-mod-picker-update-block {
        display: flex;
        flex-direction: column;
        gap: 6px;
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
    .alexz-mod-picker-status-card {
        border: 1px solid var(--border-color, #555);
        background: var(--comfy-input-bg, rgba(255,255,255,0.02));
        color: var(--input-text, inherit);
        border-radius: 7px;
        padding: 7px 8px;
        font-size: 12px;
        line-height: 1.3;
        display: block;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--warn {
        border-color: #b64040;
        background: rgba(180, 64, 64, 0.16);
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--ok {
        border-color: #2e8f61;
        background: rgba(61, 187, 126, 0.16);
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--neutral {
        border-color: var(--border-color, #555);
        background: rgba(120, 120, 120, 0.08);
        color: var(--input-text, inherit);
    }
    .alexz-mod-picker-status-card-actions {
        margin-top: 6px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .alexz-mod-picker-process-inline {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px dashed var(--border-color, #555);
        display: none;
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
    .alexz-mod-picker-module-note.alexz-mod-picker-module-note--ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-module-note.alexz-mod-picker-module-note--warn {
        color: #ff6b6b;
        font-weight: 700;
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

    const headRight = document.createElement("div");
    headRight.className = "alexz-mod-picker-head-right";
    head.appendChild(headRight);

    const debugToggle = document.createElement("button");
    debugToggle.type = "button";
    debugToggle.className = "alexz-mod-picker-btn-small";
    debugToggle.textContent = "Debug";
    headRight.appendChild(debugToggle);

    const debugCard = document.createElement("div");
    debugCard.className = "alexz-mod-picker-debug-card";
    root.appendChild(debugCard);

    const debugCardHeader = document.createElement("div");
    debugCardHeader.className = "alexz-mod-picker-debug-card-header";
    debugCard.appendChild(debugCardHeader);

    const debugTitle = document.createElement("div");
    debugTitle.className = "alexz-mod-picker-debug-title";
    debugTitle.textContent = "Debug diagnostics";
    debugCardHeader.appendChild(debugTitle);

    const debugCopyBtn = document.createElement("button");
    debugCopyBtn.type = "button";
    debugCopyBtn.className = "alexz-mod-picker-btn-small";
    debugCopyBtn.textContent = "Copy ⧉";
    debugCardHeader.appendChild(debugCopyBtn);

    const diagnostics = document.createElement("div");
    diagnostics.className = "alexz-mod-picker-diag";
    diagnostics.textContent = "diag: waiting for sidebar sync...";
    debugCard.appendChild(diagnostics);

    const dividerTop = document.createElement("div");
    dividerTop.className = "alexz-mod-picker-divider";
    root.appendChild(dividerTop);

    const updateGrid = document.createElement("div");
    updateGrid.className = "alexz-mod-picker-update-grid";
    root.appendChild(updateGrid);

    const comfyUpdateBlock = document.createElement("div");
    comfyUpdateBlock.className = "alexz-mod-picker-update-block";
    updateGrid.appendChild(comfyUpdateBlock);

    const customUpdateBlock = document.createElement("div");
    customUpdateBlock.className = "alexz-mod-picker-update-block";
    updateGrid.appendChild(customUpdateBlock);

    const comfyInfoBtn = document.createElement("button");
    comfyInfoBtn.type = "button";
    comfyInfoBtn.textContent = "Refresh ComfyUI Info";
    comfyInfoBtn.className = "alexz-mod-picker-btn-small";
    comfyUpdateBlock.appendChild(comfyInfoBtn);

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.textContent = "Refresh Custom Nodes Info";
    refreshBtn.className = "alexz-mod-picker-btn-small";
    customUpdateBlock.appendChild(refreshBtn);

    const comfyModeSelect = document.createElement("select");
    comfyModeSelect.className = "alexz-mod-picker-btn-small";
    comfyModeSelect.title = "ComfyUI update-check mode";
    const modeReleases = document.createElement("option");
    modeReleases.value = "releases";
    modeReleases.textContent = "ComfyUI check: releases";
    comfyModeSelect.appendChild(modeReleases);
    const modeCommits = document.createElement("option");
    modeCommits.value = "commits";
    modeCommits.textContent = "ComfyUI check: commits";
    comfyModeSelect.appendChild(modeCommits);
    comfyUpdateBlock.appendChild(comfyModeSelect);

    const comfyAlert = document.createElement("div");
    comfyAlert.className = "alexz-mod-picker-status-card alexz-mod-picker-status-card--neutral";
    const comfyAlertText = document.createElement("div");
    comfyAlert.appendChild(comfyAlertText);
    const comfyActions = document.createElement("div");
    comfyActions.className = "alexz-mod-picker-status-card-actions";
    comfyAlert.appendChild(comfyActions);
    const comfyUpdateBtn = document.createElement("button");
    comfyUpdateBtn.type = "button";
    comfyUpdateBtn.className = "alexz-mod-picker-btn-small";
    comfyUpdateBtn.textContent = "Update ComfyUI";
    comfyUpdateBtn.style.display = "none";
    comfyActions.appendChild(comfyUpdateBtn);
    const comfyInstallReqBtn = document.createElement("button");
    comfyInstallReqBtn.type = "button";
    comfyInstallReqBtn.className = "alexz-mod-picker-btn-small";
    comfyInstallReqBtn.textContent = "Install ComfyUI requirements";
    comfyInstallReqBtn.style.display = "none";
    comfyActions.appendChild(comfyInstallReqBtn);
    root.appendChild(comfyAlert);

    const customAlert = document.createElement("div");
    customAlert.className = "alexz-mod-picker-status-card alexz-mod-picker-status-card--neutral";
    const customAlertText = document.createElement("div");
    customAlert.appendChild(customAlertText);
    const customActions = document.createElement("div");
    customActions.className = "alexz-mod-picker-status-card-actions";
    customAlert.appendChild(customActions);
    const updateAllBtn = document.createElement("button");
    updateAllBtn.type = "button";
    updateAllBtn.textContent = "Update Custom Nodes";
    updateAllBtn.className = "alexz-mod-picker-btn-small";
    updateAllBtn.style.display = "none";
    customActions.appendChild(updateAllBtn);
    root.appendChild(customAlert);

    const processHost = document.createElement("div");
    processHost.className = "alexz-mod-picker-process-inline";

    const dividerBottom = document.createElement("div");
    dividerBottom.className = "alexz-mod-picker-divider";
    root.appendChild(dividerBottom);

    const categorySelect = document.createElement("select");
    categorySelect.className = "alexz-mod-picker-select";
    const catComfy = document.createElement("option");
    catComfy.value = "comfy";
    catComfy.textContent = "ComfyUI Nodes";
    categorySelect.appendChild(catComfy);
    const catCustom = document.createElement("option");
    catCustom.value = "custom";
    catCustom.textContent = "Custom Nodes";
    categorySelect.appendChild(catCustom);
    categorySelect.value = "custom";
    root.appendChild(categorySelect);

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
    processHost.appendChild(refreshLine);

    const processActions = document.createElement("div");
    processActions.className = "alexz-mod-picker-status-card-actions";
    processHost.appendChild(processActions);

    const pickerStore = createModuleNodePickerStore({
        defaultSelectedGroup: "custom",
        defaultSelectedModule: DEFAULT_MODULE,
        defaultDebugEnabled: Boolean(window[NODE_PICKER_DEBUG_KEY]),
        selectedGroupStorageKey: NODE_PICKER_SELECTED_GROUP_STORAGE_KEY,
        selectedModuleStorageKey: NODE_PICKER_SELECTED_MODULE_STORAGE_KEY,
        debugStorageKey: NODE_PICKER_DEBUG_STORAGE_KEY,
    });
    const diagnosticsLogger = createModuleDiagnosticsLogger({
        namespace: "ALEXZ_tools Node Picker",
        maxEntries: 200,
        debugEnabled: Boolean(pickerStore.get("debugEnabled")),
    });

    const loadComfyCheckMode = () => {
        try {
            const raw = window.localStorage?.getItem(COMFYUI_CHECK_MODE_STORAGE_KEY);
            return String(raw || "releases").trim().toLowerCase() === "commits" ? "commits" : "releases";
        } catch (_err) {
            return "releases";
        }
    };
    const saveComfyCheckMode = (mode) => {
        const normalized = String(mode || "releases").trim().toLowerCase() === "commits" ? "commits" : "releases";
        try {
            window.localStorage?.setItem(COMFYUI_CHECK_MODE_STORAGE_KEY, normalized);
        } catch (_err) {
            // Ignore storage failures and keep runtime value only.
        }
    };
    comfyModeSelect.value = loadComfyCheckMode();

    let debugEnabled = Boolean(pickerStore.get("debugEnabled"));
    const applyDebugUiState = () => {
        window[NODE_PICKER_DEBUG_KEY] = Boolean(debugEnabled);
        diagnosticsLogger.setDebugEnabled(Boolean(debugEnabled));
        debugCard.hidden = !debugEnabled;
        debugCard.style.display = debugEnabled ? "block" : "none";
        debugToggle.textContent = debugEnabled ? "Debug: ON" : "Debug";
    };
    pickerStore.subscribe("debugEnabled", (value) => {
        debugEnabled = Boolean(value);
        applyDebugUiState();
    });
    applyDebugUiState();
    debugToggle.addEventListener("click", () => {
        pickerStore.set({ debugEnabled: !Boolean(pickerStore.get("debugEnabled")) });
    });
    debugCopyBtn.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(diagnostics.textContent || "");
            setHelpText("Debug diagnostics copied to clipboard.");
        } catch (_err) {
            setHelpText("Failed to copy debug diagnostics.");
        }
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
    const moduleInlineStatus = new Map();
    const updatedModulesSession = new Set();
    let refreshPollToken = 0;
    let updatePollToken = 0;
    let customModulesNeedUpdate = 0;
    let customStatusChecked = false;
    let processTarget = "";
    let actionBusy = false;
    let expandedModule = "";

    /**
     * Store one-line module action result shown inside module card.
     */
    const setModuleInlineStatus = (moduleName, text, tone = "neutral") => {
        const key = String(moduleName || "").trim();
        if (!key) {
            return;
        }
        if (!text) {
            moduleInlineStatus.delete(key);
            return;
        }
        moduleInlineStatus.set(key, {
            text: String(text),
            tone: String(tone || "neutral"),
        });
    };

    /**
     * Mount progress inline block into the selected top status card.
     */
    const setProcessTarget = (target) => {
        const normalized = String(target || "").trim().toLowerCase();
        processTarget = normalized;
        const parent = processHost.parentElement;
        if (parent) {
            parent.removeChild(processHost);
        }
        if (normalized === "comfy") {
            comfyAlert.appendChild(processHost);
            comfyAlert.style.display = "block";
        } else if (normalized === "custom") {
            customAlert.appendChild(processHost);
            customAlert.style.display = "block";
        }
    };

    /**
     * Return true when UI is currently in Custom Nodes mode.
     */
    const isCustomCategory = () => String(categorySelect.value || "") === "custom";

    /**
     * Return effective group id based on category/subgroup selection.
     */
    const getSelectedGroup = () => {
        if (isCustomCategory()) {
            return "custom";
        }
        return String(groupSelect.value || "").trim();
    };

    /**
     * Persist currently selected group/module into picker store.
     */
    const syncPickerSelectionState = () => {
        const partial = {
            selectedGroup: getSelectedGroup() || "custom",
        };
        const selectedModule = String(nodeSelect.value || "").trim();
        if (selectedModule && selectedModule !== "-1") {
            partial.selectedModule = selectedModule;
        }
        pickerStore.set(partial);
    };

    /**
     * Render ComfyUI status card based on selected update-check mode.
     */
    const renderComfyAlert = (info) => {
        const behind = Number(info?.behind);
        const status = String(info?.update_status || "unknown");
        const mode = String(info?.check_mode || comfyModeSelect.value || "releases");
        const branch = String(info?.branch || "unknown");
        const local = String(info?.installed_commit_short || "unknown");
        const remote = String(info?.remote_commit_short || "unknown");
        const releaseTag = String(info?.release_tag || "").trim();
        const canUpdate = status === "can_update" && (!Number.isFinite(behind) || behind > 0);
        const requirementsPending = Boolean(info?.requirements_update_pending);
        const requirementsPendingAt = info?.requirements_pending_updated_at
            ? ` (${fmtDate(info.requirements_pending_updated_at)})`
            : "";

        comfyAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        comfyAlert.style.display = "block";

        if (canUpdate) {
            comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
            if (mode === "releases" && releaseTag) {
                comfyAlertText.textContent = `ComfyUI requires update (releases): release=${releaseTag}, local=${local}, remote=${remote}.`;
            } else if (Number.isFinite(behind) && behind > 0) {
                comfyAlertText.textContent = `ComfyUI requires update (commits): branch=${branch}, behind=${behind}, local=${local}, remote=${remote}.`;
            } else {
                comfyAlertText.textContent = `ComfyUI requires update: mode=${mode}, branch=${branch}, local=${local}, remote=${remote}.`;
            }
            if (requirementsPending) {
                comfyAlertText.textContent += ` requirements.txt install is pending${requirementsPendingAt}.`;
                comfyInstallReqBtn.style.display = "";
                comfyInstallReqBtn.disabled = actionBusy;
            } else {
                comfyInstallReqBtn.style.display = "none";
            }
            comfyUpdateBtn.style.display = "";
            comfyUpdateBtn.disabled = actionBusy;
            return;
        }

        if (requirementsPending) {
            comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
            comfyAlertText.textContent = `ComfyUI requirements.txt install is pending${requirementsPendingAt}.`;
            comfyUpdateBtn.style.display = "none";
            comfyInstallReqBtn.style.display = "";
            comfyInstallReqBtn.disabled = actionBusy;
            return;
        }

        comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
        if (Boolean(info?.updated_between_runs)) {
            const prev = String(info?.startup_prev_commit_short || "unknown");
            const next = String(info?.startup_new_commit_short || "unknown");
            const at = info?.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
            comfyAlertText.textContent = `ComfyUI updated between runs: ${prev} -> ${next}${at}. No updates required.`;
        } else {
            comfyAlertText.textContent = `ComfyUI is up to date (${mode} check).`;
        }
        comfyUpdateBtn.style.display = "none";
        comfyInstallReqBtn.style.display = "none";
    };

    /**
     * Render Custom Nodes status card and global update button.
     */
    const renderCustomAlert = () => {
        if (customModulesNeedUpdate <= 0 && !customStatusChecked) {
            customAlert.style.display = "none";
            updateAllBtn.style.display = "none";
            return;
        }
        customAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        customAlert.style.display = "block";
        if (customModulesNeedUpdate > 0) {
            customAlert.classList.add("alexz-mod-picker-status-card--warn");
            customAlertText.textContent = `${customModulesNeedUpdate} custom modules require update.`;
            updateAllBtn.textContent = `Update Custom Nodes (${customModulesNeedUpdate})`;
            updateAllBtn.style.display = "";
            updateAllBtn.disabled = actionBusy;
            return;
        }
        customAlert.classList.add("alexz-mod-picker-status-card--neutral");
        customAlertText.textContent = "Custom Nodes: no updates required.";
        updateAllBtn.style.display = "none";
    };

    /**
     * Return node catalog entries for currently selected group.
     */
    const getNodesForSelectedGroup = () => {
        const group = getSelectedGroup();
        return catalogByGroup.get(group) || [];
    };

    /**
     * Show a process action row with optional button (e.g., install requirements).
     */
    const setProcessAction = (label, btnText, onClick) => {
        processActions.innerHTML = "";
        if (!label) {
            if (!refreshLine.textContent) {
                processHost.style.display = "none";
            }
            return;
        }
        if (!processHost.parentElement) {
            setProcessTarget(processTarget || "custom");
        }
        processHost.style.display = "";
        const labelEl = document.createElement("div");
        labelEl.textContent = label;
        processActions.appendChild(labelEl);
        if (!btnText || typeof onClick !== "function") {
            return;
        }
        const actionBtn = document.createElement("button");
        actionBtn.type = "button";
        actionBtn.className = "alexz-mod-picker-btn-small";
        actionBtn.textContent = btnText;
        actionBtn.disabled = actionBusy;
        actionBtn.onclick = onClick;
        processActions.appendChild(actionBtn);
    };

    /**
     * Update inline process text with optional color tone.
     */
    const setRefreshLine = (text, tone = "neutral") => {
        const value = String(text || "");
        refreshLine.textContent = value;
        refreshLine.classList.remove("alexz-mod-picker-refresh-line--ok", "alexz-mod-picker-refresh-line--warn");
        if (!value) {
            if (!processActions.children.length) {
                processHost.style.display = "none";
            }
            return;
        }
        if (!processHost.parentElement) {
            setProcessTarget(processTarget || "custom");
        }
        processHost.style.display = "";
        if (tone === "ok") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--ok");
        } else if (tone === "warn") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--warn");
        }
        diagnosticsLogger.info(value, null, { forceConsole: true });
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
        bindContainerOwnershipSync(container, root, setDiagnosticText, diagnosticsLogger);
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
        renderHelpText(help, text);
    };

    /**
     * Render expanded-module help summary with insertion hints and legend.
     */
    const setHelpModuleSummary = (moduleName, nodeCount) => {
        renderHelpModuleSummary(help, moduleName, nodeCount, {
            updatedMark: MODULE_MARK_UPDATED,
            remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
        });
    };

    /**
     * Render collapsed-module hint shown before node list expansion.
     */
    const setHelpModuleCardHint = (moduleName, nodeCount) => {
        renderHelpModuleCardHint(help, moduleName, nodeCount);
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
                setRefreshLine(`Custom Nodes refresh status failed (${String(err)}).`, "warn");
                return false;
            }
            const refresh = payload?.refresh || {};
            const line = formatRefreshLine(refresh);
            setRefreshLine(line.text, line.tone);
            if (processTarget === "custom") {
                customAlert.style.display = "block";
                customAlert.classList.remove(
                    "alexz-mod-picker-status-card--warn",
                    "alexz-mod-picker-status-card--ok",
                    "alexz-mod-picker-status-card--neutral"
                );
                if (line.tone === "warn") {
                    customAlert.classList.add("alexz-mod-picker-status-card--warn");
                } else if (line.tone === "ok") {
                    customAlert.classList.add("alexz-mod-picker-status-card--ok");
                } else {
                    customAlert.classList.add("alexz-mod-picker-status-card--neutral");
                }
                customAlertText.textContent = line.text;
            }
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
        comfyModeSelect.disabled = actionBusy;
        updateAllBtn.disabled = actionBusy;
        comfyUpdateBtn.disabled = actionBusy || comfyUpdateBtn.style.display === "none";
        comfyInstallReqBtn.disabled = actionBusy || comfyInstallReqBtn.style.display === "none";
        for (const btn of moduleInfo.querySelectorAll(".alexz-mod-picker-action-row .alexz-mod-picker-btn-small")) {
            btn.disabled = actionBusy;
        }
        for (const btn of processActions.querySelectorAll(".alexz-mod-picker-btn-small")) {
            btn.disabled = actionBusy;
        }
    };

    /**
     * Toggle visibility and label of the global custom-nodes update button.
     */
    const syncUpdateAllButton = () => {
        renderCustomAlert();
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
                setRefreshLine(`Update status failed (${String(err)}).`, "warn");
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
     * Install ComfyUI requirements and refresh ComfyUI status card.
     */
    const installComfyUIRequirementsFlow = async () => {
        setActionBusy(true);
        setProcessTarget("comfy");
        try {
            setRefreshLine("Installing ComfyUI dependencies (pip)...", "neutral");
            const install = await installComfyUIRequirements();
            if (String(install?.status || "") !== "installed") {
                setRefreshLine("ComfyUI dependencies install failed.", "warn");
                return;
            }
            const comfyPayload = await fetchComfyUIInfo(false, false, comfyModeSelect.value);
            renderComfyAlert(comfyPayload?.comfyui || null);
            setRefreshLine("ComfyUI dependencies installed.", "ok");
            setProcessAction("", "", null);
        } finally {
            setActionBusy(false);
            syncUpdateAllButton();
        }
    };

    /**
     * Offer one-click requirements installation when updated modules changed requirements.txt.
     */
    const maybeInstallChangedRequirements = async (update) => {
        const scope = String(update?.scope || "");
        if (scope === "comfyui") {
            if (!Boolean(update?.requirements_changed)) {
                return;
            }
            setRefreshLine("ComfyUI requirements.txt changed. Install dependencies?", "warn");
            setProcessAction(
                "ComfyUI requirements were updated after pull.",
                "Install ComfyUI requirements",
                installComfyUIRequirementsFlow
            );
            return;
        }

        const modules = Array.isArray(update?.requirements_modules) ? update.requirements_modules : [];
        if (!modules.length) {
            return;
        }
        setRefreshLine("Custom module requirements changed. Install dependencies?", "warn");
        setProcessAction(
            `requirements.txt changed for: ${modules.join(", ")}.`,
            "Install updated requirements",
            async () => {
                setActionBusy(true);
                try {
                    setRefreshLine("Installing updated dependencies (pip)...", "neutral");
                    const install = await installModuleRequirements(modules);
                    const failed = Number(install?.failed || 0);
                    const installed = Number(install?.installed || 0);
                    if (failed > 0) {
                        setRefreshLine(`Dependencies install finished with errors: ok=${installed}, failed=${failed}.`, "warn");
                        return;
                    }
                    setRefreshLine(`Dependencies installed: ${installed} module(s).`, "ok");
                    setProcessAction("", "", null);
                } finally {
                    setActionBusy(false);
                }
            }
        );
    };

    /**
     * Run update flow (backend job + polling + optional requirements install)
     * and then refresh catalog/module state in UI.
     */
    const runModuleUpdate = async (scope, moduleName) => {
        setActionBusy(true);
        try {
            if (String(scope || "") === "comfyui") {
                setProcessTarget("comfy");
            } else {
                setProcessTarget("custom");
            }
            setProcessAction("", "", null);
            setRefreshLine("Starting update...", "neutral");
            await startModuleUpdate(scope, moduleName);
            const update = await pollUpdateProgress();
            if (!update) {
                return;
            }
            const currentGroup = getSelectedGroup();
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
            setRefreshLine(`Update failed (${String(err)}).`, "warn");
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
        option.textContent = formatModuleOption(moduleName, count, badges, {
            updatedMark: MODULE_MARK_UPDATED,
            remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
        });
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
        const selectedGroup = getSelectedGroup();
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
            pickerStore.set({ selectedGroup: getSelectedGroup() || "custom" });
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
            opt.textContent = formatModuleOption(moduleName, count, badges, {
                updatedMark: MODULE_MARK_UPDATED,
                remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
            });
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
        syncPickerSelectionState();
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
        const previousCategory = String(categorySelect.value || "").trim();
        const previousGroup = String(groupSelect.value || "").trim();
        moduleCatalogByGroup.clear();
        groups.forEach((group) => {
            catalogByGroup.set(group.id, group.nodes || []);
            moduleCatalogByGroup.set(group.id, group.modules || []);
        });
        const comfyGroups = COMFY_GROUP_ORDER.filter((groupId) => catalogByGroup.has(groupId));
        groupSelect.innerHTML = "";
        for (const groupId of comfyGroups) {
            const opt = document.createElement("option");
            const nodes = catalogByGroup.get(groupId) || [];
            opt.value = groupId;
            opt.textContent = `${GROUP_LABELS[groupId] || groupId} (${nodes.length})`;
            groupSelect.appendChild(opt);
        }
        if (preferredGroup === "custom") {
            categorySelect.value = "custom";
        } else if (preferredGroup && COMFY_GROUP_ORDER.includes(preferredGroup) && catalogByGroup.has(preferredGroup)) {
            categorySelect.value = "comfy";
        } else if (previousCategory === "comfy" || previousCategory === "custom") {
            categorySelect.value = previousCategory;
        } else if (catalogByGroup.has("custom")) {
            categorySelect.value = "custom";
        } else {
            categorySelect.value = "comfy";
        }
        if (!isCustomCategory()) {
            if (preferredGroup && comfyGroups.includes(preferredGroup)) {
                groupSelect.value = preferredGroup;
            } else if (previousGroup && comfyGroups.includes(previousGroup)) {
                groupSelect.value = previousGroup;
            } else if (comfyGroups.length > 0) {
                groupSelect.value = comfyGroups[0];
            }
        }
        groupSelect.style.display = isCustomCategory() ? "none" : "";
        pickerStore.set({ selectedGroup: getSelectedGroup() || "custom" });
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
            const requirementsPending = Boolean(info?.requirements_update_pending);
            const requirementsPendingAt = info?.requirements_pending_updated_at
                ? ` (${fmtDate(info.requirements_pending_updated_at)})`
                : "";

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

            if (requirementsPending) {
                const reqRow = document.createElement("div");
                reqRow.className = "alexz-mod-picker-module-row warn";
                const reqLabel = document.createElement("span");
                reqLabel.className = "alexz-mod-picker-module-label";
                reqLabel.textContent = "Requirements:";
                const reqValue = document.createElement("span");
                reqValue.textContent = `requirements.txt install pending${requirementsPendingAt}`;
                reqRow.appendChild(reqLabel);
                reqRow.appendChild(reqValue);
                card.appendChild(reqRow);
            }

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
                const moduleName = String(info.module || nodeSelect.value || "").trim();
                setProcessTarget("");
                setRefreshLine("", "neutral");
                setProcessAction("", "", null);
                setModuleInlineStatus(moduleName, "Refreshing module info...", "neutral");
                setActionBusy(true);
                try {
                    await loadModuleInfo({ forceRefresh: true, syncUpstream: true, throwOnError: true });
                    setModuleInlineStatus(moduleName, "Module info updated.", "ok");
                } catch (err) {
                    setModuleInlineStatus(moduleName, `Failed to refresh module info: ${String(err)}`, "warn");
                } finally {
                    await loadModuleInfo({ forceRefresh: false, syncUpstream: false });
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

            if (requirementsPending) {
                const installReqBtn = document.createElement("button");
                installReqBtn.type = "button";
                installReqBtn.className = "alexz-mod-picker-btn-small";
                installReqBtn.textContent = "Install module requirements";
                installReqBtn.disabled = actionBusy;
                installReqBtn.onclick = async (event) => {
                    event.stopPropagation();
                    if (actionBusy) {
                        return;
                    }
                    const moduleName = String(info.module || nodeSelect.value || "").trim();
                    if (!moduleName) {
                        return;
                    }
                    setProcessTarget("custom");
                    setRefreshLine(`Installing requirements for ${moduleName}...`, "neutral");
                    setProcessAction("", "", null);
                    setModuleInlineStatus(moduleName, "Installing module requirements...", "neutral");
                    setActionBusy(true);
                    try {
                        const install = await installModuleRequirements([moduleName]);
                        const failed = Number(install?.failed || 0);
                        const installed = Number(install?.installed || 0);
                        if (failed > 0 || installed <= 0) {
                            setModuleInlineStatus(moduleName, "Module requirements install failed.", "warn");
                        } else {
                            setModuleInlineStatus(moduleName, "Module requirements installed.", "ok");
                        }
                    } catch (err) {
                        setModuleInlineStatus(moduleName, `Module requirements install failed: ${String(err)}`, "warn");
                    } finally {
                        await loadModuleInfo({ forceRefresh: false, syncUpstream: false });
                        setActionBusy(false);
                        syncUpdateAllButton();
                    }
                };
                actionRow.appendChild(installReqBtn);
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
                const moduleName = String(info.module || nodeSelect.value || "").trim();
                setProcessTarget("");
                setRefreshLine("", "neutral");
                setProcessAction("", "", null);
                setModuleInlineStatus(moduleName, "Refreshing module info...", "neutral");
                setActionBusy(true);
                try {
                    await loadModuleInfo({ forceRefresh: true, syncUpstream: false, throwOnError: true });
                    setModuleInlineStatus(moduleName, "Module info updated.", "ok");
                } catch (err) {
                    setModuleInlineStatus(moduleName, `Failed to refresh module info: ${String(err)}`, "warn");
                } finally {
                    await loadModuleInfo({ forceRefresh: false, syncUpstream: false });
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

        const inlineStatus = moduleInlineStatus.get(selectedModule);
        if (inlineStatus && inlineStatus.text) {
            const statusLine = document.createElement("div");
            statusLine.className = "alexz-mod-picker-module-note";
            if (inlineStatus.tone === "ok") {
                statusLine.classList.add("alexz-mod-picker-module-note--ok");
            } else if (inlineStatus.tone === "warn") {
                statusLine.classList.add("alexz-mod-picker-module-note--warn");
            }
            statusLine.textContent = inlineStatus.text;
            card.appendChild(statusLine);
        }

        moduleInfo.appendChild(card);
    };

    /**
     * Load and render module info for currently selected group/module.
     */
    const loadModuleInfo = async (options = {}) => {
        const selectedModule = nodeSelect.value;
        const selectedGroup = getSelectedGroup();
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
            if (nodeSelect.value !== selectedModule || getSelectedGroup() !== selectedGroup) {
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
            const payload = await fetchNodeCatalog(comfyModeSelect.value);
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
            comfyAlert.classList.remove("alexz-mod-picker-status-card--warn", "alexz-mod-picker-status-card--ok");
            comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
            comfyAlert.style.display = "block";
            comfyAlertText.textContent = "ComfyUI status unavailable (catalog load failed).";
            comfyUpdateBtn.style.display = "none";
            customModulesNeedUpdate = 0;
            groupSelect.innerHTML = "";
            nodeSelect.innerHTML = "";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
            syncUpdateAllButton();
        }
    };

    groupSelect.onchange = () => {
        if (isCustomCategory()) {
            return;
        }
        pickerStore.set({ selectedGroup: getSelectedGroup() || "custom" });
        fillModuleSelect();
        syncUpdateAllButton();
    };
    categorySelect.onchange = () => {
        groupSelect.style.display = isCustomCategory() ? "none" : "";
        pickerStore.set({ selectedGroup: getSelectedGroup() || "custom" });
        fillModuleSelect();
        syncUpdateAllButton();
    };
    moduleFilter.oninput = () => fillModuleSelect();
    nodeSelect.onchange = () => {
        expandedModule = "";
        nodeList.innerHTML = "";
        syncPickerSelectionState();
        loadModuleInfo();
    };
    updateAllBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        customStatusChecked = true;
        setProcessTarget("custom");
        await runModuleUpdate("all", "");
    };
    comfyUpdateBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        setProcessTarget("comfy");
        await runModuleUpdate("comfyui", "");
    };
    comfyInstallReqBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        await installComfyUIRequirementsFlow();
    };
    comfyInfoBtn.onclick = async () => {
        if (actionBusy) {
            return;
        }
        setActionBusy(true);
        setProcessTarget("comfy");
        setProcessAction("", "", null);
        setRefreshLine("Refreshing ComfyUI info...", "neutral");
        comfyAlert.style.display = "block";
        comfyAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
        comfyAlertText.textContent = "Refreshing ComfyUI info...";
        try {
            const payload = await fetchComfyUIInfo(true, true, comfyModeSelect.value);
            renderComfyAlert(payload?.comfyui || null);
        } catch (err) {
            comfyAlert.classList.remove(
                "alexz-mod-picker-status-card--warn",
                "alexz-mod-picker-status-card--ok",
                "alexz-mod-picker-status-card--neutral"
            );
            comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
            comfyAlert.style.display = "block";
            comfyAlertText.textContent = `Failed to refresh ComfyUI info: ${String(err)}`;
            comfyUpdateBtn.style.display = "none";
            comfyInstallReqBtn.style.display = "none";
        } finally {
            setActionBusy(false);
            syncUpdateAllButton();
        }
    };
    comfyModeSelect.onchange = async () => {
        saveComfyCheckMode(comfyModeSelect.value);
        await loadCatalog();
    };
    refreshBtn.onclick = async () => {
        customStatusChecked = true;
        setActionBusy(true);
        setProcessTarget("custom");
        setProcessAction("", "", null);
        setRefreshLine("Refreshing Custom Nodes info...", "neutral");
        customAlert.style.display = "block";
        customAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        customAlert.classList.add("alexz-mod-picker-status-card--neutral");
        customAlertText.textContent = "Refreshing Custom Nodes info...";
        try {
            await refreshModuleRuntimeState();
            const ok = await pollRefreshProgress();
            if (!ok) {
                setRefreshLine("Custom Nodes refresh finished with errors.", "warn");
            } else {
                try {
                    await acknowledgeAllModuleNovelty();
                } catch (err) {
                    setRefreshLine(`Refresh completed, but novelty reset failed: ${String(err)}`, "warn");
                }
            }
        } catch (err) {
            setRefreshLine(`Custom Nodes refresh error: ${String(err)}`, "warn");
        } finally {
            setActionBusy(false);
        }
        await loadCatalog();
    };

    const startupGroup = String(pickerStore.get("selectedGroup") || "custom").trim();
    const startupModule = String(pickerStore.get("selectedModule") || DEFAULT_MODULE).trim();
    loadCatalog({
        preferredGroup: startupGroup || "custom",
        preferredModule: startupModule || DEFAULT_MODULE,
    });
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
function bindContainerOwnershipSync(container, root, onDiag, diagnosticsLogger = null) {
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
    const isDebugLoggingEnabled = () => {
        if (diagnosticsLogger && typeof diagnosticsLogger.isDebugEnabled === "function") {
            return diagnosticsLogger.isDebugEnabled();
        }
        return Boolean(window[NODE_PICKER_DEBUG_KEY]);
    };
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
            if (diagnosticsLogger && typeof diagnosticsLogger.info === "function") {
                diagnosticsLogger.info("sync", diag);
            } else {
                console.log("ALEXZ_tools Node Picker sync:", diag);
            }
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
