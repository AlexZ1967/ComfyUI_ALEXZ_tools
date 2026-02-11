/**
 * Module: web/module_node_picker.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
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
    renderHelpHintText,
    renderHelpHintTextWithTone,
    renderHelpModuleSummary,
    renderHelpModuleCardHint,
} from "./ui/module_node_picker_help.js";
import {
    renderNodeListPanel,
    renderModuleInfoCard,
} from "./ui/module_node_picker_renderers.js";
import {
    renderComfyAlertCard,
    renderCustomAlertCard,
} from "./ui/module_node_picker_alerts.js";
import { createProcessUiController } from "./ui/module_node_picker_process.js";
import {
    fillModuleSelectUi,
    fillGroupSelectUi,
} from "./ui/module_node_picker_catalog.js";
import {
    pollRefreshProgressLoop,
    pollUpdateProgressLoop,
    runInstallComfyUIRequirementsFlow,
    maybeInstallChangedRequirementsFlow,
    runModuleUpdateFlow,
} from "./orchestration/module_node_picker_update_flow.js";
import {
    updateModuleOptionText,
    cacheModuleNodeDiffs,
    loadModuleInfoFlow,
    loadCatalogFlow,
} from "./orchestration/module_node_picker_data_flow.js";
import {
    runRefreshModuleInfoAction,
    runInstallSingleModuleRequirementsAction,
    runRefreshComfyUIInfoAction,
    runRefreshCustomNodesInfoAction,
} from "./orchestration/module_node_picker_actions.js";
import {
    bindModuleNodePickerEvents,
    runModuleNodePickerStartupLoad,
} from "./orchestration/module_node_picker_bindings.js";
import { createModuleNodePickerStore } from "./state/store.js";
import { createModuleDiagnosticsLogger } from "./diagnostics/logger.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const SIDEBAR_TAB_ID = "alexz-module-nodes";
const MODULE_PICKER_GUARD_KEY = "__alexz_module_node_picker_registered__";
const FALLBACK_BUTTON_ID = "alexz-module-nodes-fallback-btn";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const NODE_PICKER_DEBUG_KEY = "__alexz_module_picker_debug__";
const NODE_PICKER_DEBUG_STORAGE_KEY = "alexz_module_picker_debug";
const NODE_PICKER_SELECTED_GROUP_STORAGE_KEY = "alexz_module_picker_selected_group";
const NODE_PICKER_SELECTED_MODULE_STORAGE_KEY = "alexz_module_picker_selected_module";
const COMFYUI_CHECK_MODE_STORAGE_KEY = "alexz_comfyui_check_mode";
const PICKER_CLEANUP_KEY = "__alexz_module_node_picker_cleanup__";
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
        justify-content: flex-end;
    }
    .alexz-mod-picker-module-info-wrap {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .alexz-mod-picker-help-main {
        font-size: 13px;
        line-height: 1.3;
        opacity: 0.95;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .alexz-mod-picker-help-main strong {
        font-weight: 700;
    }
    .alexz-mod-picker-help-hint {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
        margin-bottom: 0;
    }
    .alexz-mod-picker-help-hint--warn {
        color: #ff6b6b;
        opacity: 0.95;
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
    .alexz-mod-picker-node-legend {
        padding: 0;
        margin-top: 2px;
        margin-bottom: 6px;
        border: none;
        background: transparent;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .alexz-mod-picker-node-legend-row {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
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
    const prevCleanup = container?.[PICKER_CLEANUP_KEY];
    if (typeof prevCleanup === "function") {
        try {
            prevCleanup();
        } catch (_err) {
            // Ignore stale cleanup errors from previous picker instance.
        }
    }
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
    comfyAlert.style.display = "none";
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

    const moduleHintDivider = document.createElement("div");
    moduleHintDivider.className = "alexz-mod-picker-divider";
    root.appendChild(moduleHintDivider);

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

    const moduleInfoWrap = document.createElement("div");
    moduleInfoWrap.className = "alexz-mod-picker-module-info-wrap";
    root.appendChild(moduleInfoWrap);

    const help = document.createElement("div");
    help.className = "alexz-mod-picker-help";
    moduleInfoWrap.appendChild(help);

    const moduleInfo = document.createElement("div");
    moduleInfoWrap.appendChild(moduleInfo);

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
    let catalogLoadToken = 0;
    let moduleInfoLoadToken = 0;
    let refreshPollToken = 0;
    let updatePollToken = 0;
    let customModulesNeedUpdate = 0;
    let customStatusChecked = false;
    let actionBusy = false;
    let expandedModule = "";

    let pickerDisposed = false;
    const isPickerAlive = () => !pickerDisposed && root.isConnected;
    const disposePickerInstance = () => {
        if (pickerDisposed) {
            return;
        }
        pickerDisposed = true;
        catalogLoadToken += 1;
        moduleInfoLoadToken += 1;
        refreshPollToken += 1;
        updatePollToken += 1;
        unbindModuleNodesTabRelay();
    };
    container[PICKER_CLEANUP_KEY] = disposePickerInstance;

    /**
     * Store one-line module action result shown inside module card.
     */
    const setModuleInlineStatus = (moduleName, text, tone = "neutral") => {
        if (!isPickerAlive()) {
            return;
        }
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

    const processUi = createProcessUiController({
        processHost,
        refreshLine,
        processActions,
        comfyAlert,
        customAlert,
        diagnosticsLogger,
        defaultTarget: () => "custom",
    });

    /**
     * Mount progress inline block into the selected top status card.
     */
    const setProcessTarget = (target) => {
        if (!isPickerAlive()) {
            return;
        }
        processUi.setTarget(target);
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
        if (!isPickerAlive()) {
            return;
        }
        renderComfyAlertCard({
            info,
            comfyMode: comfyModeSelect.value,
            actionBusy,
            fmtDate,
            comfyAlert,
            comfyAlertText,
            comfyUpdateBtn,
            comfyInstallReqBtn,
        });
    };

    /**
     * Render Custom Nodes status card and global update button.
     */
    const renderCustomAlert = () => {
        if (!isPickerAlive()) {
            return;
        }
        renderCustomAlertCard({
            customModulesNeedUpdate,
            customStatusChecked,
            actionBusy,
            customAlert,
            customAlertText,
            updateAllBtn,
        });
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
        if (!isPickerAlive()) {
            return;
        }
        processUi.setAction(label, btnText, onClick, actionBusy);
    };

    /**
     * Update inline process text with optional color tone.
     */
    const setRefreshLine = (text, tone = "neutral") => {
        if (!isPickerAlive()) {
            return;
        }
        processUi.setLine(text, tone);
    };
    /**
     * Render compact diagnostics block for tab-sync troubleshooting.
     */
    const setDiagnosticText = (diag) => {
        if (!isPickerAlive()) {
            return;
        }
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
    bindModuleNodesTabRelay({
        app,
        root,
        sidebarTabId: SIDEBAR_TAB_ID,
        onDiag: setDiagnosticText,
    });

    /**
     * Replace help area with plain status/help text.
     */
    const setHelpText = (text) => {
        if (!isPickerAlive()) {
            return;
        }
        renderHelpText(help, text);
    };

    /**
     * Replace help area with compact hint-like message.
     */
    const setHelpHintText = (text, tone = "neutral") => {
        if (!isPickerAlive()) {
            return;
        }
        if (String(tone || "").toLowerCase() === "warn") {
            renderHelpHintTextWithTone(help, text, "warn");
            return;
        }
        renderHelpHintText(help, text);
    };

    /**
     * Render expanded-module help summary with insertion hints and legend.
     */
    const setHelpModuleSummary = (moduleName, nodeCount) => {
        if (!isPickerAlive()) {
            return;
        }
        renderHelpModuleSummary(help, moduleName, nodeCount, {
            updatedMark: MODULE_MARK_UPDATED,
            remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
        });
    };

    /**
     * Render collapsed-module hint shown before node list expansion.
     */
    const setHelpModuleCardHint = (moduleName, nodeCount) => {
        if (!isPickerAlive()) {
            return;
        }
        renderHelpModuleCardHint(help, moduleName, nodeCount);
    };

    /**
     * Poll refresh status endpoint until job completes or fails.
     */
    const pollRefreshProgress = async () => {
        const token = ++refreshPollToken;
        return pollRefreshProgressLoop({
            isTokenActive: () => token === refreshPollToken,
            fetchModuleRefreshStatus,
            formatRefreshLine,
            setRefreshLine,
            getProcessTarget: () => processUi.getTarget(),
            customAlert,
            customAlertText,
            sleepMs: 400,
        });
    };

    /**
     * Enable/disable actionable UI controls during long-running operations.
     */
    const setActionBusy = (busy) => {
        if (!isPickerAlive()) {
            return;
        }
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
        processUi.setButtonsDisabled(actionBusy);
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
        return pollUpdateProgressLoop({
            isTokenActive: () => token === updatePollToken,
            fetchModuleUpdateStatus,
            formatUpdateLine,
            setRefreshLine,
            sleepMs: 450,
        });
    };

    /**
     * Install ComfyUI requirements and refresh ComfyUI status card.
     */
    const installComfyUIRequirementsFlow = async () => {
        return runInstallComfyUIRequirementsFlow({
            setActionBusy,
            setProcessTarget,
            setRefreshLine,
            installComfyUIRequirements,
            fetchComfyUIInfo,
            getComfyMode: () => comfyModeSelect.value,
            renderComfyAlert,
            setProcessAction,
            syncUpdateAllButton,
        });
    };

    /**
     * Offer one-click requirements installation when updated modules changed requirements.txt.
     */
    const maybeInstallChangedRequirements = async (update) => {
        return maybeInstallChangedRequirementsFlow(update, {
            setRefreshLine,
            setProcessAction,
            installComfyUIRequirementsFlow,
            installModuleRequirements,
            setActionBusy,
        });
    };

    /**
     * Run update flow (backend job + polling + optional requirements install)
     * and then refresh catalog/module state in UI.
     */
    const runModuleUpdate = async (scope, moduleName) => {
        return runModuleUpdateFlow(scope, moduleName, {
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            startModuleUpdate,
            pollUpdateProgress,
            getSelectedGroup,
            getSelectedModule: () => String(nodeSelect.value || "").trim(),
            onMarkUpdatedModule: (mod) => updatedModulesSession.add(mod),
            isModuleMarkedUpdated: (mod) => updatedModulesSession.has(String(mod || "").trim()),
            maybeInstallChangedRequirements,
            loadCatalog,
            loadModuleInfo,
            syncUpdateAllButton,
        });
    };

    /**
     * Refresh module select option text for one module after badge updates.
     */
    const setModuleOptionText = (moduleName) => {
        updateModuleOptionText(
            {
                moduleOptions,
                moduleCounts,
                moduleBadges,
                formatModuleOption,
                marks: {
                    updatedMark: MODULE_MARK_UPDATED,
                    remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
                },
            },
            moduleName
        );
    };

    /**
     * Cache node-level diff markers (new/updated) for selected module.
     */
    const setModuleNodeDiffs = (moduleName, info) => {
        cacheModuleNodeDiffs(
            {
                moduleNodeDiffs,
            },
            moduleName,
            info
        );
    };

    /**
     * Load and render module info for currently selected group/module.
     */
    const loadModuleInfo = async (options = {}) => {
        if (!isPickerAlive()) {
            return;
        }
        const token = ++moduleInfoLoadToken;
        return loadModuleInfoFlow(options, {
            isRequestActive: () => token === moduleInfoLoadToken && isPickerAlive(),
            getSelectedModule: () => String(nodeSelect.value || ""),
            getSelectedGroup,
            fetchModuleInfo,
            clearModuleInfo: () => {
                moduleInfo.innerHTML = "";
            },
            renderModuleInfo,
            moduleBadgesFromInfo,
            moduleBadges,
            setModuleNodeDiffs,
            setModuleOptionText,
            renderNodeList,
        });
    };

    /**
     * Load full node catalog from backend and refresh picker UI state.
     */
    const loadCatalog = async (options = {}) => {
        if (!isPickerAlive()) {
            return;
        }
        const token = ++catalogLoadToken;
        return loadCatalogFlow(options, {
            isRequestActive: () => token === catalogLoadToken && isPickerAlive(),
            fetchNodeCatalog,
            getComfyMode: () => comfyModeSelect.value,
            catalogByGroup,
            setCustomModulesNeedUpdate: (value) => {
                customModulesNeedUpdate = Number(value || 0);
            },
            renderComfyAlert,
            fillGroupSelect,
            groupLabels: GROUP_LABELS,
            setHelpText,
            syncUpdateAllButton,
            comfyAlert,
            comfyAlertText,
            comfyUpdateBtn,
            comfyInstallReqBtn,
            groupSelect,
            nodeSelect,
            clearModuleInfo: () => {
                moduleInfo.innerHTML = "";
            },
            nodeList,
        });
    };

    /**
     * Populate module selector for current group with filtering and badge placeholders.
     */
    const fillModuleSelect = (options = {}) => {
        fillModuleSelectUi({
            options,
            nodes: getNodesForSelectedGroup(),
            selectedGroup: getSelectedGroup(),
            moduleEntries: moduleCatalogByGroup.get(getSelectedGroup()) || [],
            moduleFilterValue: moduleFilter.value,
            moduleFilterRaw: moduleFilter.value,
            previousSelectedModule: nodeSelect.value,
            moduleCounts,
            moduleOptions,
            moduleBadges,
            moduleNodeDiffs,
            nodeSelect,
            moduleInfo,
            nodeList,
            pickerStore,
            getSelectedGroup,
            setHelpText,
            syncUpdateAllButton,
            moduleBadgesFromModuleEntry,
            formatModuleOption,
            marks: {
                updatedMark: MODULE_MARK_UPDATED,
                remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
            },
            defaultModule: DEFAULT_MODULE,
            setExpandedModule: (value) => {
                expandedModule = String(value || "").trim();
            },
            syncPickerSelectionState,
            renderNodeList,
            loadModuleInfo,
        });
    };

    /**
     * Populate top-level group selector and propagate selection to module list.
     */
    const fillGroupSelect = (groups, options = {}) => {
        fillGroupSelectUi({
            groups,
            options,
            previousCategory: categorySelect.value,
            previousGroup: groupSelect.value,
            catalogByGroup,
            moduleCatalogByGroup,
            comfyGroupOrder: COMFY_GROUP_ORDER,
            groupLabels: GROUP_LABELS,
            groupSelect,
            categorySelect,
            isCustomCategory,
            pickerStore,
            getSelectedGroup,
            fillModuleSelect,
        });
    };

    /**
     * Refresh one module card metadata and keep the result inline in the card.
     */
    const refreshModuleInfoFlow = async (moduleName, syncUpstream) => {
        return runRefreshModuleInfoAction(moduleName, syncUpstream, {
            setProcessTarget,
            setRefreshLine,
            setProcessAction,
            setModuleInlineStatus,
            setActionBusy,
            loadModuleInfo,
            syncUpdateAllButton,
        });
    };

    /**
     * Install requirements for a single custom module and refresh card state.
     */
    const installSingleModuleRequirementsFlow = async (moduleName) => {
        return runInstallSingleModuleRequirementsAction(moduleName, {
            setProcessTarget,
            setRefreshLine,
            setProcessAction,
            setModuleInlineStatus,
            setActionBusy,
            installModuleRequirements,
            loadModuleInfo,
            syncUpdateAllButton,
        });
    };

    /**
     * Render node cards for currently selected module and bind insertion actions.
     */
    const renderNodeList = () => {
        if (!isPickerAlive()) {
            return;
        }
        renderNodeListPanel({
            nodeListEl: nodeList,
            selectedModule: nodeSelect.value,
            getNodesForSelectedGroup,
            expandedModule,
            setHelpText,
            setHelpHintText,
            setHelpModuleCardHint,
            setHelpModuleSummary,
            moduleNodeDiffs,
            marks: {
                updatedMark: MODULE_MARK_UPDATED,
                remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
            },
            createNodeByInfo,
            app,
            centerNode,
        });
    };

    /**
     * Render module metadata card, status rows, and per-module action buttons.
     */
    const renderModuleInfo = (info) => {
        if (!isPickerAlive()) {
            return;
        }
        const selectedModule = String(nodeSelect.value || "").trim();
        const nodeCount = moduleCounts.get(selectedModule) || 0;
        renderModuleInfoCard({
            moduleInfoEl: moduleInfo,
            info,
            selectedModule,
            nodeCount,
            isModuleUpdated:
                updatedModulesSession.has(selectedModule)
                || Boolean(info?.updated_between_runs)
                || Boolean(info?.new_module_between_runs),
            actionBusy,
            inlineStatus: moduleInlineStatus.get(selectedModule) || null,
            fmtDate,
            onExpandModule: (moduleName) => {
                const normalized = String(moduleName || "").trim();
                expandedModule = expandedModule === normalized ? "" : normalized;
                renderNodeList();
            },
            onRefreshModuleInfo: refreshModuleInfoFlow,
            onUpdateModule: async (moduleName) => runModuleUpdate("single", moduleName),
            onInstallModuleRequirements: installSingleModuleRequirementsFlow,
        });
    };

    const refreshComfyUIInfoFlow = async () => {
        return runRefreshComfyUIInfoAction({
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            comfyAlert,
            comfyAlertText,
            comfyUpdateBtn,
            comfyInstallReqBtn,
            fetchComfyUIInfo,
            getComfyMode: () => comfyModeSelect.value,
            renderComfyAlert,
            syncUpdateAllButton,
        });
    };

    const refreshCustomNodesInfoFlow = async () => {
        return runRefreshCustomNodesInfoAction({
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            customAlert,
            customAlertText,
            refreshModuleRuntimeState,
            pollRefreshProgress,
            acknowledgeAllModuleNovelty,
            loadCatalog,
        });
    };

    bindModuleNodePickerEvents({
        groupSelect,
        categorySelect,
        moduleFilter,
        nodeSelect,
        nodeList,
        updateAllBtn,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        comfyInfoBtn,
        comfyModeSelect,
        refreshBtn,
        isCustomCategory,
        pickerStore,
        getSelectedGroup,
        fillModuleSelect,
        syncUpdateAllButton,
        syncPickerSelectionState,
        loadModuleInfo,
        isActionBusy: () => actionBusy,
        setCustomStatusChecked: (value) => {
            customStatusChecked = Boolean(value);
        },
        setProcessTarget,
        runModuleUpdate,
        installComfyUIRequirementsFlow,
        refreshComfyUIInfoFlow,
        saveComfyCheckMode,
        loadCatalog,
        refreshCustomNodesInfoFlow,
        setExpandedModule: (value) => {
            expandedModule = String(value || "").trim();
        },
    });

    runModuleNodePickerStartupLoad({
        pickerStore,
        defaultModule: DEFAULT_MODULE,
        loadCatalog,
    });
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
