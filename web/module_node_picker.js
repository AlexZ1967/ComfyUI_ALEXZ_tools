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
import { createModuleNodePickerLayout } from "./ui/module_node_picker_layout.js";
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
import { isCanceledRequestError } from "./orchestration/module_node_picker_error_utils.js";
import {
    resumePendingCustomRefreshFlow as resumePendingCustomRefreshFlowImpl,
    resumePendingModuleUpdateFlow as resumePendingModuleUpdateFlowImpl,
    resumePendingComfyInfoRefreshFlow as resumePendingComfyInfoRefreshFlowImpl,
} from "./orchestration/module_node_picker_resume_flow.js";
import { createBusyUiController } from "./orchestration/module_node_picker_busy_ui.js";
import { runStartupCoordinator } from "./orchestration/module_node_picker_startup_flow.js";
import { createModuleNodePickerStore } from "./state/store.js";
import {
    getRuntimePickerState,
    clearLegacyPersistentFlags,
    createRuntimeStatusAccessors,
    loadComfyCheckMode,
    saveComfyCheckMode as persistComfyCheckMode,
} from "./state/module_node_picker_runtime_state.js";
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
const MODULE_PICKER_RUNTIME_STATE_KEY = "__alexz_module_picker_runtime_state__";
const LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY = "alexz_module_picker_custom_status_checked";
const LEGACY_PENDING_CUSTOM_REFRESH_STORAGE_KEY = "alexz_module_picker_pending_custom_refresh";
const LEGACY_PENDING_UPDATE_STORAGE_KEY = "alexz_module_picker_pending_update";
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

    const {
        root,
        debugToggle,
        debugCard,
        debugCopyBtn,
        diagnostics,
        comfyInfoBtn,
        refreshBtn,
        comfyModeSelect,
        comfyAlert,
        comfyAlertText,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        customAlert,
        customAlertText,
        updateAllBtn,
        processHost,
        categorySelect,
        groupSelect,
        nodeSelect,
        moduleFilter,
        refreshLine,
        processActions,
        help,
        moduleInfo,
        nodeList,
    } = createModuleNodePickerLayout(container);

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
    const runtimePickerState = getRuntimePickerState(window, MODULE_PICKER_RUNTIME_STATE_KEY);
    clearLegacyPersistentFlags(window, {
        customStatusCheckedKey: LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY,
        pendingCustomRefreshKey: LEGACY_PENDING_CUSTOM_REFRESH_STORAGE_KEY,
        pendingUpdateKey: LEGACY_PENDING_UPDATE_STORAGE_KEY,
    });

    const {
        loadCustomStatusChecked,
        saveCustomStatusChecked,
        loadComfyStatusChecked,
        saveComfyStatusChecked,
        loadComfyInfoSnapshot,
        saveComfyInfoSnapshot,
        hasPendingCustomRefresh,
        setPendingCustomRefresh,
        clearPendingCustomRefresh,
        hasPendingUpdate,
        setPendingUpdate,
        clearPendingUpdate,
        hasPendingComfyInfoRefresh,
        setPendingComfyInfoRefresh,
        clearPendingComfyInfoRefresh,
    } = createRuntimeStatusAccessors(runtimePickerState);

    comfyModeSelect.value = loadComfyCheckMode(window, COMFYUI_CHECK_MODE_STORAGE_KEY);
    const saveComfyCheckMode = (mode) => {
        persistComfyCheckMode(window, COMFYUI_CHECK_MODE_STORAGE_KEY, mode);
    };

    let debugEnabled = Boolean(pickerStore.get("debugEnabled"));
    const unsubscribeDebug = pickerStore.subscribe("debugEnabled", (value) => {
        debugEnabled = Boolean(value);
        applyDebugUiState();
    });
    const applyDebugUiState = () => {
        window[NODE_PICKER_DEBUG_KEY] = Boolean(debugEnabled);
        diagnosticsLogger.setDebugEnabled(Boolean(debugEnabled));
        debugCard.hidden = !debugEnabled;
        debugCard.style.display = debugEnabled ? "block" : "none";
        debugToggle.textContent = debugEnabled ? "Debug: ON" : "Debug";
    };
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

    const catalogByGroup = new Map();
    const moduleCatalogByGroup = new Map();
    const moduleCounts = new Map();
    const moduleOptions = new Map();
    const moduleBadges = new Map();
    const moduleNodeDiffs = new Map();
    const moduleInlineStatus = new Map();
    const updatedModulesSession = new Set();
    let catalogLoadToken = 0;
    let catalogLoadBusyCount = 0;
    let moduleInfoLoadToken = 0;
    let refreshPollToken = 0;
    let updatePollToken = 0;
    let customModulesNeedUpdate = 0;
    let customStatusChecked = loadCustomStatusChecked();
    let comfyStatusChecked = loadComfyStatusChecked();
    let expandedModule = "";
    let unbindPickerEvents = () => {};
    let processUi = null;
    let cancelStartupLoad = () => {};

    let pickerDisposed = false;
    const apiAbortController = typeof AbortController === "function"
        ? new AbortController()
        : null;
    // Keep async/UI flows active for this picker instance even if the root is
    // temporarily detached during sidebar transitions; lifecycle is governed by
    // explicit dispose, not transient DOM attachment state.
    const isPickerAlive = () => !pickerDisposed;
    const apiSignal = () => apiAbortController?.signal;
    const fetchNodeCatalogApi = (comfyMode) =>
        fetchNodeCatalog(comfyMode, { signal: apiSignal() });
    const fetchModuleInfoApi = (group, moduleName, options = {}) =>
        fetchModuleInfo(group, moduleName, { ...(options || {}), signal: apiSignal() });
    const fetchComfyUIInfoApi = (forceRefresh = true, acknowledge = true, comfyMode = "releases", options = {}) =>
        fetchComfyUIInfo(forceRefresh, acknowledge, comfyMode, { ...(options || {}), signal: apiSignal() });
    const refreshModuleRuntimeStateApi = (options = {}) =>
        refreshModuleRuntimeState({ ...(options || {}), signal: apiSignal() });
    const fetchModuleRefreshStatusApi = (options = {}) =>
        fetchModuleRefreshStatus({ ...(options || {}), signal: apiSignal() });
    const acknowledgeAllModuleNoveltyApi = (options = {}) =>
        acknowledgeAllModuleNovelty({ ...(options || {}), signal: apiSignal() });
    const startModuleUpdateApi = (scope, moduleName, options = {}) =>
        startModuleUpdate(scope, moduleName, { ...(options || {}), signal: apiSignal() });
    const fetchModuleUpdateStatusApi = (options = {}) =>
        fetchModuleUpdateStatus({ ...(options || {}), signal: apiSignal() });
    const installModuleRequirementsApi = (modules, options = {}) =>
        installModuleRequirements(modules, { ...(options || {}), signal: apiSignal() });
    const installComfyUIRequirementsApi = (options = {}) =>
        installComfyUIRequirements({ ...(options || {}), signal: apiSignal() });
    const disposePickerInstance = () => {
        if (pickerDisposed) {
            return;
        }
        pickerDisposed = true;
        catalogLoadToken += 1;
        moduleInfoLoadToken += 1;
        refreshPollToken += 1;
        updatePollToken += 1;
        try {
            unbindPickerEvents?.();
        } catch (_err) {
            // Ignore stale event-unbind errors.
        }
        try {
            cancelStartupLoad?.();
        } catch (_err) {
            // Ignore stale startup-load cancellation errors.
        }
        try {
            unsubscribeDebug?.();
        } catch (_err) {
            // Ignore stale store-unsubscribe errors.
        }
        try {
            processUi?.dispose?.();
        } catch (_err) {
            // Ignore stale process-ui dispose errors.
        }
        try {
            apiAbortController?.abort?.();
        } catch (_err) {
            // Ignore stale abort-controller errors.
        }
        unbindModuleNodesTabRelay();
        if (container?.[PICKER_CLEANUP_KEY] === disposePickerInstance) {
            container[PICKER_CLEANUP_KEY] = null;
        }
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

    processUi = createProcessUiController({
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

    const busyUi = createBusyUiController({
        shouldContinue: isPickerAlive,
        controls: {
            refreshBtn,
            comfyInfoBtn,
            comfyModeSelect,
            categorySelect,
            groupSelect,
            nodeSelect,
            moduleFilter,
            updateAllBtn,
            comfyUpdateBtn,
            comfyInstallReqBtn,
            moduleInfo,
            nodeList,
        },
        getProcessUi: () => processUi,
    });
    const syncBusyUiState = () => busyUi.syncBusyUiState();
    const setCatalogControlsLoading = (loading) => busyUi.setCatalogControlsLoading(loading);
    const setActionBusy = (busy) => busyUi.setActionBusy(busy);
    const setStartupBusy = (busy) => busyUi.setStartupBusy(busy);

    /**
     * Render ComfyUI status card based on selected update-check mode.
     */
    const renderComfyAlert = (info) => {
        if (!isPickerAlive()) {
            return;
        }
        if (info && typeof info === "object") {
            comfyStatusChecked = true;
            saveComfyStatusChecked(true);
            saveComfyInfoSnapshot(info);
        }
        renderComfyAlertCard({
            info,
            comfyMode: comfyModeSelect.value,
            actionBusy: busyUi.getActionBusy(),
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
            actionBusy: busyUi.getActionBusy(),
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
        processUi.setAction(label, btnText, onClick, busyUi.getActionBusy());
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
     * Mirror refresh-line tone/text into Custom Nodes status card.
     */
    const setCustomRefreshCardLine = (text, tone = "neutral") => {
        if (!isPickerAlive()) {
            return;
        }
        if (!customAlert || !customAlertText) {
            return;
        }
        customAlert.style.display = "block";
        customAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        if (tone === "warn") {
            customAlert.classList.add("alexz-mod-picker-status-card--warn");
        } else if (tone === "ok") {
            customAlert.classList.add("alexz-mod-picker-status-card--ok");
        } else {
            customAlert.classList.add("alexz-mod-picker-status-card--neutral");
        }
        customAlertText.textContent = String(text || "");
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
            shouldContinue: isPickerAlive,
            isTokenActive: () => token === refreshPollToken,
            fetchModuleRefreshStatus: fetchModuleRefreshStatusApi,
            formatRefreshLine,
            setRefreshLine,
            getProcessTarget: () => processUi.getTarget(),
            customAlert,
            customAlertText,
            sleepMs: 400,
        });
    };

    /**
     * Toggle visibility and label of the global custom-nodes update button.
     */
    const syncUpdateAllButton = () => {
        renderCustomAlert();
    };

    /**
     * Set persisted flag that Custom Nodes status was explicitly checked.
     */
    const setCustomStatusChecked = (checked) => {
        customStatusChecked = Boolean(checked);
        saveCustomStatusChecked(customStatusChecked);
        renderCustomAlert();
    };

    /**
     * Set persisted-in-session flag that ComfyUI status was explicitly checked.
     */
    const setComfyStatusChecked = (checked) => {
        comfyStatusChecked = Boolean(checked);
        saveComfyStatusChecked(comfyStatusChecked);
    };


    /**
     * Poll update status endpoint until module update job finishes.
     */
    const pollUpdateProgress = async () => {
        const token = ++updatePollToken;
        return pollUpdateProgressLoop({
            shouldContinue: isPickerAlive,
            isTokenActive: () => token === updatePollToken,
            fetchModuleUpdateStatus: fetchModuleUpdateStatusApi,
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
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessTarget,
            setRefreshLine,
            installComfyUIRequirements: installComfyUIRequirementsApi,
            fetchComfyUIInfo: fetchComfyUIInfoApi,
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
            shouldContinue: isPickerAlive,
            setRefreshLine,
            setProcessAction,
            installComfyUIRequirementsFlow,
            installModuleRequirements: installModuleRequirementsApi,
            setActionBusy,
        });
    };

    /**
     * Run update flow (backend job + polling + optional requirements install)
     * and then refresh catalog/module state in UI.
     */
    const runModuleUpdate = async (scope, moduleName) => {
        return runModuleUpdateFlow(scope, moduleName, {
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            startModuleUpdate: startModuleUpdateApi,
            pollUpdateProgress,
            getSelectedGroup,
            getSelectedModule: () => String(nodeSelect.value || "").trim(),
            onMarkUpdatedModule: (mod) => updatedModulesSession.add(mod),
            isModuleMarkedUpdated: (mod) => updatedModulesSession.has(String(mod || "").trim()),
            maybeInstallChangedRequirements,
            loadCatalog,
            loadModuleInfo,
            syncUpdateAllButton,
            setPendingUpdate,
            clearPendingUpdate,
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
            fetchModuleInfo: fetchModuleInfoApi,
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
        catalogLoadBusyCount += 1;
        if (catalogLoadBusyCount === 1) {
            setCatalogControlsLoading(true);
        }
        try {
            return await loadCatalogFlow(options, {
                isRequestActive: () => token === catalogLoadToken && isPickerAlive(),
                fetchNodeCatalog: fetchNodeCatalogApi,
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
        } finally {
            catalogLoadBusyCount = Math.max(0, catalogLoadBusyCount - 1);
            if (catalogLoadBusyCount === 0) {
                setCatalogControlsLoading(false);
            }
        }
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
            shouldContinue: isPickerAlive,
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
            shouldContinue: isPickerAlive,
            setProcessTarget,
            setRefreshLine,
            setProcessAction,
            setModuleInlineStatus,
            setActionBusy,
            installModuleRequirements: installModuleRequirementsApi,
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
            actionBusy: busyUi.getActionBusy(),
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
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            comfyAlert,
            comfyAlertText,
            comfyUpdateBtn,
            comfyInstallReqBtn,
            fetchComfyUIInfo: fetchComfyUIInfoApi,
            getComfyMode: () => comfyModeSelect.value,
            renderComfyAlert,
            syncUpdateAllButton,
            setComfyStatusChecked,
            setPendingComfyInfoRefresh,
            clearPendingComfyInfoRefresh,
        });
    };

    const refreshCustomNodesInfoFlow = async () => {
        return runRefreshCustomNodesInfoAction({
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            customAlert,
            customAlertText,
            refreshModuleRuntimeState: refreshModuleRuntimeStateApi,
            pollRefreshProgress,
            acknowledgeAllModuleNovelty: acknowledgeAllModuleNoveltyApi,
            loadCatalog,
            setCustomStatusChecked,
            setPendingCustomRefresh,
            clearPendingCustomRefresh,
        });
    };

    /**
     * Restore in-flight Custom Nodes refresh after picker re-open/re-render.
     */
    const resumePendingCustomRefreshFlow = async () => {
        return resumePendingCustomRefreshFlowImpl({
            hasPendingCustomRefresh,
            shouldContinue: isPickerAlive,
            setCustomStatusChecked,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            setCustomRefreshCardLine,
            fetchModuleRefreshStatus: fetchModuleRefreshStatusApi,
            pollRefreshProgress,
            acknowledgeAllModuleNovelty: acknowledgeAllModuleNoveltyApi,
            loadCatalog,
            clearPendingCustomRefresh,
            formatRefreshLine,
            isCanceledRequestError,
        });
    };

    /**
     * Restore in-flight module-update job after picker re-open/re-render.
     */
    const resumePendingModuleUpdateFlow = async () => {
        return resumePendingModuleUpdateFlowImpl({
            hasPendingUpdate,
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessAction,
            setRefreshLine,
            fetchModuleUpdateStatus: fetchModuleUpdateStatusApi,
            setProcessTarget,
            formatUpdateLine,
            pollUpdateProgress,
            clearPendingUpdate,
            maybeInstallChangedRequirements,
            loadCatalog,
            loadModuleInfo,
            isCanceledRequestError,
        });
    };

    /**
     * Restore interrupted ComfyUI info refresh after picker re-open/re-render.
     */
    const resumePendingComfyInfoRefreshFlow = async () => {
        return resumePendingComfyInfoRefreshFlowImpl({
            hasPendingComfyInfoRefresh,
            shouldContinue: isPickerAlive,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            comfyAlert,
            comfyAlertText,
            fetchComfyUIInfo: fetchComfyUIInfoApi,
            getComfyMode: () => comfyModeSelect.value,
            renderComfyAlert,
            clearPendingComfyInfoRefresh,
            syncUpdateAllButton,
            isCanceledRequestError,
        });
    };

    unbindPickerEvents = bindModuleNodePickerEvents({
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
        isActionBusy: () => busyUi.isActionBusy(),
        setCustomStatusChecked,
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
    }) || (() => {});

    // Restore last ComfyUI status card across widget switches in current session.
    if (comfyStatusChecked && !hasPendingComfyInfoRefresh()) {
        const lastComfyInfo = loadComfyInfoSnapshot();
        if (lastComfyInfo) {
            renderComfyAlert(lastComfyInfo);
        }
    }

    cancelStartupLoad = runStartupCoordinator({
        shouldContinue: isPickerAlive,
        setStartupBusy,
        startCatalogStartupLoad: (options = {}) => runModuleNodePickerStartupLoad({
            pickerStore,
            defaultModule: DEFAULT_MODULE,
            loadCatalog,
            shouldContinue: isPickerAlive,
            startupRetries: 2,
            startupRetryDelayMs: 250,
            onSettled: options?.onSettled,
        }),
        hasPendingCustomRefresh,
        hasPendingUpdate,
        hasPendingComfyInfoRefresh,
        resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow,
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
