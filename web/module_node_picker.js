/**
 * Frontend module: `module_node_picker.js`.
 * Handles ComfyUI web-side UI behavior for ALEXZ tools.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const SIDEBAR_TAB_ID = "alexz-module-nodes";
const MODULE_PICKER_GUARD_KEY = "__alexz_module_node_picker_registered__";
const FALLBACK_BUTTON_ID = "alexz-module-nodes-fallback-btn";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const CONTAINER_SYNC_STATE_KEY = "__alexz_module_nodes_container_sync_state__";
const NODE_PICKER_DEBUG_KEY = "__alexz_module_picker_debug__";
const NODE_PICKER_SIDEBAR_SYNC_KEY = "__alexz_module_picker_sidebar_sync__";
const NODE_PICKER_TAB_RELAY_STATE_KEY = "__alexz_module_picker_tab_relay_state__";
const GROUP_LABELS = {
    core: "Core_Nodes",
    core_extras: "Core_Extras_Nodes",
    api: "API_Nodes",
    custom: "Custom_Nodes",
};
const MODULE_MARK_UPDATED = "✅";
const MODULE_MARK_REMOTE_UPDATE = "🟥";

/** Handle `injectStyles` workflow step. */
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

/** Parse sidebar tab id from button-like element classes. */
function extractTabIdFromButtonElement(buttonEl) {
    if (!(buttonEl instanceof Element)) {
        return "";
    }
    for (const cls of Array.from(buttonEl.classList || [])) {
        if (cls.endsWith("-tab-button")) {
            return cls.slice(0, -"-tab-button".length);
        }
    }
    return "";
}

/** Resolve sidebar tab button from a DOM event using closest/path fallback. */
function resolveSidebarButtonFromEventObject(event) {
    const directTarget = event?.target;
    if (directTarget instanceof Element) {
        const byClosest = directTarget.closest(".side-bar-button, [class*='-tab-button']");
        if (byClosest) {
            return byClosest;
        }
    }
    if (typeof event?.composedPath === "function") {
        for (const item of event.composedPath()) {
            if (!(item instanceof Element)) {
                continue;
            }
            if (item.classList?.contains("side-bar-button")) {
                return item;
            }
            const tabId = extractTabIdFromButtonElement(item);
            if (tabId) {
                return item;
            }
        }
    }
    return null;
}

/** Remove lightweight tab relay listeners. */
function unbindMinimalTabRelay() {
    const state = window[NODE_PICKER_TAB_RELAY_STATE_KEY];
    if (!state) {
        return;
    }
    const timerId = Number(state.relayTimer || 0);
    if (timerId) {
        window.clearTimeout(timerId);
    }
    const intervalId = Number(state.visibilityInterval || 0);
    if (intervalId) {
        window.clearInterval(intervalId);
    }
    const bindIntervalId = Number(state.bindButtonsInterval || 0);
    if (bindIntervalId) {
        window.clearInterval(bindIntervalId);
    }
    const rootCheckId = Number(state.rootCheckInterval || 0);
    if (rootCheckId) {
        window.clearInterval(rootCheckId);
    }
    if (state.onPointerDown) {
        document.removeEventListener("pointerdown", state.onPointerDown, true);
    }
    if (state.onMouseDown) {
        document.removeEventListener("mousedown", state.onMouseDown, true);
    }
    if (state.onClick) {
        document.removeEventListener("click", state.onClick, true);
    }
    if (state.onKeyUp) {
        document.removeEventListener("keyup", state.onKeyUp, true);
    }
    if (Array.isArray(state.boundButtons)) {
        for (const item of state.boundButtons) {
            if (!item?.el || !item?.handler) {
                continue;
            }
            item.el.removeEventListener("pointerdown", item.handler, true);
            item.el.removeEventListener("click", item.handler, true);
            item.el.removeEventListener("mousedown", item.handler, true);
        }
    }
    window[NODE_PICKER_TAB_RELAY_STATE_KEY] = null;
}

/** Bind lightweight tab relay to ensure other sidebar tabs activate reliably. */
function bindMinimalTabRelay(root, onDiag) {
    unbindMinimalTabRelay();
    let relayTimer = 0;
    let visibilityInterval = 0;
    let bindButtonsInterval = 0;
    let lastDiagSig = "";
    let lastClickedTabId = "";
    let lastClickedTs = 0;
    const boundButtons = [];

    // Save container reference - will be updated if root changes parent
    let savedContainer = root.parentElement;
    const getActiveSidebarTabId = () => {
        const manager = app.extensionManager;
        const sidebar = manager?.sidebarTab || manager;
        const active = sidebar?.activeSidebarTabId ?? sidebar?.activeSidebarTab ?? "";
        return String(active || "");
    };
    const isOurButtonSelected = () => {
        const ownBtn = document.querySelector(`.${SIDEBAR_TAB_ID}-tab-button`);
        if (!ownBtn) {
            return null;
        }
        return ownBtn.classList.contains("side-bar-button-selected");
    };
    const getLiveContainer = () => {
        // Always use the saved container from bind time, not parent of root
        // because Vue might move root around
        if (savedContainer && savedContainer.isConnected) {
            return savedContainer;
        }
        // Fallback: if saved container is gone, use root's parent
        const parent = root.parentElement;
        if (parent instanceof Element) {
            return parent;
        }
        return null;
    };
    const getContainerState = () => {
        const container = getLiveContainer();
        if (!container) {
            return {
                childCount: 0,
                childShort: "n/a",
                hasForeignContent: false,
            };
        }
        const out = [];
        let foreign = false;
        let childCount = 0;
        for (const node of container.childNodes) {
            childCount += 1;
            if (node === root) {
                out.push("ROOT");
                continue;
            }
            if (node.nodeType === Node.TEXT_NODE) {
                const txt = String(node.textContent || "").trim();
                if (!txt) {
                    out.push("TXT:blank");
                    continue;
                }
                foreign = true;
                out.push(`TXT:${txt.slice(0, 20)}`);
                continue;
            }
            foreign = true;
            if (node instanceof Element) {
                const cls = String(node.className || "").trim();
                out.push(cls ? `${node.nodeName}.${cls.split(/\s+/).slice(0, 2).join(".")}` : node.nodeName);
            } else {
                out.push(String(node.nodeName || "NODE"));
            }
        }
        return {
            childCount,
            childShort: out.slice(0, 10).join(" | ") || "n/a",
            hasForeignContent: foreign,
        };
    };
    const emitDiag = (reason, clickedTabId = "") => {
        const ownSelected = isOurButtonSelected();
        const containerState = getContainerState();
        const diag = {
            reason,
            activeTabId: getActiveSidebarTabId() || "n/a",
            lastClickedTabId: clickedTabId || lastClickedTabId || "n/a",
            ownBtnFound: ownSelected !== null,
            ownBtnSelected: ownSelected,
            rootDisplay: root.style.display || "",
            childNodesCount: containerState.childCount,
            childNodesShort: containerState.childShort,
        };
        const sig = JSON.stringify(diag);
        if (sig === lastDiagSig) {
            return;
        }
        lastDiagSig = sig;
        onDiag?.(diag);
    };
    const syncRootVisibility = (reason, clickedTabId = "") => {
        const ownSelected = isOurButtonSelected();
        const containerState = getContainerState();
        const activeTabId = getActiveSidebarTabId();
        const clickedRecently = Boolean(
            lastClickedTabId &&
            lastClickedTabId !== SIDEBAR_TAB_ID &&
            Date.now() - lastClickedTs < 1600
        );

        // Ensure root is in the DOM if it got disconnected OR moved to wrong parent
        if (!root.isConnected || root.parentElement !== getLiveContainer()) {
            console.warn("ALEXZ: syncRootVisibility - root is NOT in correct location, attempting to reconnect", {
                isConnected: root.isConnected,
                currentParent: root.parentElement?.className || root.parentElement?.tagName,
                expectedParent: getLiveContainer()?.className,
            });
            const liveContainer = getLiveContainer();
            if (liveContainer && liveContainer.isConnected) {
                // Remove from wherever it is
                if (root.parentElement) {
                    try {
                        root.parentElement.removeChild(root);
                    } catch (e) {
                        // Already removed
                    }
                }
                // Add to correct location
                liveContainer.appendChild(root);
                console.log("ALEXZ: syncRootVisibility - root re-appended to correct location");
            }
        }

        if (
            reason === "relay_tick" &&
            ownSelected === true &&
            activeTabId === SIDEBAR_TAB_ID &&
            clickedRecently
        ) {
            const manager = app.extensionManager;
            const sidebar = manager?.sidebarTab || manager;
            if (sidebar && typeof sidebar.activateSidebarTab === "function") {
                try {
                    sidebar.activateSidebarTab(lastClickedTabId);
                    emitDiag("relay_tick_forced", lastClickedTabId);
                    return;
                } catch (_err) {
                    // no-op: keep default behavior below
                }
            }
        }
        if (containerState.hasForeignContent) {
            root.style.display = "none";
            emitDiag("relay_foreign_content", clickedTabId);
            return;
        }
        // Ensure visibility takes priority: if we're the selected button, always show
        if (ownSelected === null) {
            // Button not found yet - show by default to ensure widget appears
            root.style.display = "";
        } else if (ownSelected === false) {
            root.style.display = "none";
        } else if (ownSelected === true) {
            root.style.display = "";
        }
        emitDiag(reason, clickedTabId);
    };
    const processTabButton = (button) => {
        if (!(button instanceof Element)) {
            return;
        }
        const tabId = extractTabIdFromButtonElement(button);
        if (!tabId || tabId === SIDEBAR_TAB_ID) {
            return;
        }
        lastClickedTabId = tabId;
        lastClickedTs = Date.now();
        // Only relay when our tab is currently selected. Otherwise we might
        // interfere with normal navigation between other tabs.
        if (isOurButtonSelected() !== true) {
            return;
        }
        if (relayTimer) {
            window.clearTimeout(relayTimer);
        }
        relayTimer = window.setTimeout(() => {
            relayTimer = 0;
            const activeTabId = getActiveSidebarTabId();
            const ownSelected = isOurButtonSelected();
            // If Comfy switched tab normally, do nothing.
            if (activeTabId === tabId || ownSelected === false) {
                syncRootVisibility("relay_native_ok", tabId);
                return;
            }
            const manager = app.extensionManager;
            const sidebar = manager?.sidebarTab || manager;
            if (sidebar && typeof sidebar.activateSidebarTab === "function") {
                try {
                    sidebar.activateSidebarTab(tabId);
                    window.setTimeout(() => syncRootVisibility("relay_forced", tabId), 40);
                } catch (_err) {
                    syncRootVisibility("relay_force_failed", tabId);
                }
            }
            const liveState = window[NODE_PICKER_TAB_RELAY_STATE_KEY];
            if (liveState) {
                liveState.relayTimer = 0;
            }
        }, 45);
        const liveState = window[NODE_PICKER_TAB_RELAY_STATE_KEY];
        if (liveState) {
            liveState.relayTimer = relayTimer;
        }
    };
    const processTabClickLikeEvent = (event) => {
        const button = resolveSidebarButtonFromEventObject(event);
        if (!button) {
            return;
        }
        processTabButton(button);
    };
    const bindDirectTabButtonListeners = () => {
        const buttons = Array.from(document.querySelectorAll(".side-bar-button, [class*='-tab-button']"));
        for (const button of buttons) {
            if (!(button instanceof Element)) {
                continue;
            }
            const tabId = extractTabIdFromButtonElement(button);
            if (!tabId || tabId === SIDEBAR_TAB_ID) {
                continue;
            }
            if (boundButtons.some((item) => item.el === button)) {
                continue;
            }
            const handler = () => {
                processTabButton(button);
            };
            button.addEventListener("pointerdown", handler, true);
            button.addEventListener("mousedown", handler, true);
            button.addEventListener("click", handler, true);
            boundButtons.push({ el: button, handler });
        }
    };
    const onPointerDown = (event) => {
        processTabClickLikeEvent(event);
    };
    const onMouseDown = (event) => {
        processTabClickLikeEvent(event);
    };
    const onClick = (event) => {
        processTabClickLikeEvent(event);
    };
    const onKeyUp = () => {
        syncRootVisibility("relay_keyup");
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("click", onClick, true);
    document.addEventListener("keyup", onKeyUp, true);

    // Periodically check if root got moved to wrong parent and re-attach if needed
    const rootCheckInterval = window.setInterval(() => {
        const liveContainer = getLiveContainer();
        const isDetached = !root.isConnected || (liveContainer && root.parentElement !== liveContainer);
        if (isDetached && liveContainer && liveContainer.isConnected) {
            // Root was detached or moved to wrong container, re-attach it immediately
            if (root.parentElement && root.parentElement !== liveContainer) {
                try {
                    root.parentElement.removeChild(root);
                } catch (e) {
                    // Already removed
                }
            }
            // Insert at the beginning to ensure visibility
            if (liveContainer.firstChild && liveContainer.firstChild !== root) {
                liveContainer.insertBefore(root, liveContainer.firstChild);
                console.log("ALEXZ: rootCheckInterval - root moved to beginning");
            } else if (!liveContainer.firstChild) {
                liveContainer.appendChild(root);
                console.log("ALEXZ: rootCheckInterval - root appended to empty container");
            }
            // Force display
            root.style.display = "";
            // Update saved container in case it changed
            if (savedContainer !== liveContainer) {
                savedContainer = liveContainer;
            }
        }
    }, 50);  // Check every 50ms to catch Vue re-renders quickly

    visibilityInterval = window.setInterval(() => {
        if (!root.isConnected) {
            return;
        }
        syncRootVisibility("relay_tick");
    }, 200);
    bindDirectTabButtonListeners();
    bindButtonsInterval = window.setInterval(() => {
        bindDirectTabButtonListeners();
    }, 800);
    syncRootVisibility("relay_init");
    window[NODE_PICKER_TAB_RELAY_STATE_KEY] = {
        onPointerDown,
        onMouseDown,
        onClick,
        onKeyUp,
        relayTimer: 0,
        visibilityInterval,
        bindButtonsInterval,
        boundButtons,
        rootCheckInterval,
    };
}

/** Handle `centerNode` workflow step. */
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

/** Handle `createNodeByInfo` workflow step. */
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

/** Handle `fetchNodeCatalog` workflow step. */
async function fetchNodeCatalog() {
    const resp = await api.fetchApi("/alexz_tools/node_catalog", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/** Handle `fetchModuleInfo` workflow step. */
async function fetchModuleInfo(group, moduleName, options = {}) {
    const forceRefresh = Boolean(options?.forceRefresh);
    const syncUpstream = Boolean(options?.syncUpstream);
    const resp = await api.fetchApi(
        `/alexz_tools/module_info?group=${encodeURIComponent(group || "")}` +
        `&module=${encodeURIComponent(moduleName || "")}` +
        `&refresh=${forceRefresh ? "1" : "0"}` +
        `&sync_upstream=${syncUpstream ? "1" : "0"}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/** Handle `fetchComfyUIInfo` workflow step. */
async function fetchComfyUIInfo(forceRefresh = true) {
    const resp = await api.fetchApi(
        `/alexz_tools/comfyui_info?refresh=${forceRefresh ? "1" : "0"}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/** Handle `refreshModuleRuntimeState` workflow step. */
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

/** Handle `fetchModuleRefreshStatus` workflow step. */
async function fetchModuleRefreshStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/** Handle `startModuleUpdate` workflow step. */
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

/** Handle `fetchModuleUpdateStatus` workflow step. */
async function fetchModuleUpdateStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_update_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/** Handle `installModuleRequirements` workflow step. */
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

/** Handle `installComfyUIRequirements` workflow step. */
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

/** Handle `fmtDate` workflow step. */
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

/** Handle `moduleBadgesFromInfo` workflow step. */
function moduleBadgesFromInfo(info) {
    const behind = Number(info?.git_behind);
    return {
        updatedBetweenRuns: Boolean(info?.updated_between_runs),
        hasRemoteUpdate: Number.isFinite(behind) && behind > 0,
    };
}

/** Handle `formatModuleOption` workflow step. */
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

/** Handle `renderPicker` workflow step. */
function renderPicker(container) {
    // Safety: clear any previous global tab-sync hooks before re-rendering.
    unbindContainerOwnershipSync();
    unbindMinimalTabRelay();

    // Remove any old root elements from anywhere in the DOM
    const oldRoots = document.querySelectorAll(".alexz-mod-picker");
    for (const oldRoot of oldRoots) {
        if (oldRoot.parentElement) {
            oldRoot.parentElement.removeChild(oldRoot);
        }
    }

    // DON'T use container.innerHTML = "" because it can trigger Vue re-renders
    // Instead, remove children one by one to be safe
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    // Explicitly ensure container is visible on re-render to fix tab switching bug
    container.style.display = "";

    const root = document.createElement("div");
    root.className = "alexz-mod-picker";
    root.style.display = "";  // Ensure root is visible initially
    root.style.minHeight = "100px";  // Ensure it has minimum height

    console.log("ALEXZ: renderPicker - about to append root to container", {
        containerClass: container.className,
        containerParent: container.parentElement?.className,
    });

    container.appendChild(root);

    console.log("ALEXZ: renderPicker - root appended", {
        rootParent: root.parentElement?.className || root.parentElement?.tagName,
        rootConnected: root.isConnected,
        containerCheck: root.parentElement === container,
    });

    // Verify root is actually in the container
    if (root.parentElement !== container) {
        console.error("ALEXZ_tools: ERROR - root parent mismatch!", {
            rootParent: root.parentElement?.className || root.parentElement?.tagName,
            expectedContainer: container.className,
            containerParent: container.parentElement?.className,
        });
        // Try to move it to correct location
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        container.appendChild(root);
    }

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
    let moduleBadgeLoadToken = 0;
    let refreshPollToken = 0;
    let updatePollToken = 0;
    let customModulesNeedUpdate = 0;
    let actionBusy = false;
    let expandedModule = "";

    /** Handle `renderComfyAlert` workflow step. */
    const renderComfyAlert = (info) => {
        const behind = Number(info?.behind);
        const status = String(info?.update_status || "unknown");
        if (status !== "can_update" || !Number.isFinite(behind) || behind <= 0) {
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

    /** Handle `getNodesForSelectedGroup` workflow step. */
    const getNodesForSelectedGroup = () => {
        const group = groupSelect.value;
        return catalogByGroup.get(group) || [];
    };

    const setRefreshLine = (text, tone = "neutral") => {
        refreshLine.textContent = text || "";
        refreshLine.classList.remove("alexz-mod-picker-refresh-line--ok", "alexz-mod-picker-refresh-line--warn");
        if (tone === "ok") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--ok");
        } else if (tone === "warn") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--warn");
        }
    };
    const setDiagnosticText = (diag) => {
        const rootComputed = window.getComputedStyle(root);
        const containerComputed = window.getComputedStyle(container);
        const parentComputed = container.parentElement ? window.getComputedStyle(container.parentElement) : null;

        // Find where root actually is in DOM
        let rootActualParent = "?";
        let rootActualPath = "?";
        try {
            if (root.parentElement) {
                rootActualParent = root.parentElement.className || root.parentElement.tagName;
            }
            const rootNode = root.getRootNode?.();
            if (rootNode && rootNode.nodeType === Node.DOCUMENT_NODE) {
                rootActualPath = "in-document";
            } else if (rootNode && rootNode.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
                rootActualPath = "in-fragment";
            } else {
                rootActualPath = "unknown";
            }
        } catch (e) {
            rootActualPath = "error";
        }

        const lines = [
            `diag.ts=${new Date().toLocaleTimeString()}`,
            `diag.reason=${diag?.reason || "unknown"}`,
            `diag.active_tab=${diag?.activeTabId || "n/a"}`,
            `diag.own_btn_selected=${diag?.ownBtnSelected === null ? "n/a" : (diag?.ownBtnSelected ? "yes" : "no")}`,
            `diag.root: display=${rootComputed.display}, visibility=${rootComputed.visibility}`,
            `diag.root: height=${rootComputed.height}, overflow=${rootComputed.overflow}`,
            `diag.root_connected=${root.isConnected ? "yes" : "NO"}`,
            `diag.root_parent=${rootActualParent}`,
            `diag.root_path=${rootActualPath}`,
            `diag.container: display=${containerComputed.display}`,
            `diag.container_parent=${container.parentElement?.className || "no_parent"}`,
        ];
        diagnostics.textContent = lines.join("\n");
    };
    if (Boolean(window[NODE_PICKER_SIDEBAR_SYNC_KEY])) {
        // Temporarily disabled: use bindMinimalTabRelay instead for better reliability
        // bindContainerOwnershipSync(container, root, setDiagnosticText);
        root.style.display = "";
        bindMinimalTabRelay(root, setDiagnosticText);
    } else {
        // Initialize root visibility BEFORE binding Tab Relay
        root.style.display = "";
        bindMinimalTabRelay(root, setDiagnosticText);
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

    /** Handle `setHelpText` workflow step. */
    const setHelpText = (text) => {
        help.innerHTML = "";
        help.textContent = text || "";
    };

    /** Handle `setHelpModuleSummary` workflow step. */
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

    /** Handle `formatRefreshLine` workflow step. */
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

    /** Handle `pollRefreshProgress` workflow step. */
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

    /** Handle `setActionBusy` workflow step. */
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

    /** Handle `syncUpdateAllButton` workflow step. */
    const syncUpdateAllButton = () => {
        const show = groupSelect.value === "custom" && customModulesNeedUpdate > 0;
        if (!show) {
            updateAllBtn.style.display = "none";
            return;
        }
        updateAllBtn.style.display = "";
        updateAllBtn.textContent = `Update all custom_nodes (${customModulesNeedUpdate})`;
    };

    /** Handle `formatUpdateLine` workflow step. */
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

    /** Handle `pollUpdateProgress` workflow step. */
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

    /** Handle `maybeInstallChangedRequirements` workflow step. */
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

    /** Handle `runModuleUpdate` workflow step. */
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

    /** Handle `setModuleOptionText` workflow step. */
    const setModuleOptionText = (moduleName) => {
        const option = moduleOptions.get(moduleName);
        if (!option) {
            return;
        }
        const count = moduleCounts.get(moduleName) || 0;
        const badges = moduleBadges.get(moduleName) || null;
        option.textContent = formatModuleOption(moduleName, count, badges);
    };

    /** Handle `setModuleNodeDiffs` workflow step. */
    const setModuleNodeDiffs = (moduleName, info) => {
        const newNodes = Array.isArray(info?.new_nodes_between_runs) ? info.new_nodes_between_runs : [];
        const updatedNodes = Array.isArray(info?.updated_nodes_between_runs) ? info.updated_nodes_between_runs : [];
        moduleNodeDiffs.set(moduleName, {
            newNodes: new Set(newNodes),
            updatedNodes: new Set(updatedNodes),
            markAllUpdated: Boolean(info?.new_module_between_runs),
        });
    };

    /** Handle `loadModuleBadges` workflow step. */
    const loadModuleBadges = async (group, modules) => {
        const token = ++moduleBadgeLoadToken;
        if (!modules.length) {
            return;
        }

        const queue = [...modules];
        const workers = Array.from({ length: Math.min(4, queue.length) }, async () => {
            while (queue.length && token === moduleBadgeLoadToken) {
                const moduleName = queue.shift();
                if (!moduleName) {
                    break;
                }
                try {
                    const payload = await fetchModuleInfo(group, moduleName);
                    if (token !== moduleBadgeLoadToken || groupSelect.value !== group) {
                        return;
                    }
                    const badges = moduleBadgesFromInfo(payload?.info || {});
                    if (badges.updatedBetweenRuns || badges.hasRemoteUpdate) {
                        moduleBadges.set(moduleName, badges);
                    } else {
                        moduleBadges.delete(moduleName);
                    }
                    setModuleOptionText(moduleName);
                } catch (err) {
                    // Ignore per-module errors and keep list usable.
                }
            }
        });

        await Promise.all(workers);
    };

    /** Handle `fillModuleSelect` workflow step. */
    const fillModuleSelect = (options = {}) => {
        const preferredModule = String(options?.preferredModule || "").trim();
        const autoExpandModule = String(options?.autoExpandModule || "").trim();
        const nodes = getNodesForSelectedGroup();
        const selectedGroup = groupSelect.value;
        const moduleEntries = moduleCatalogByGroup.get(selectedGroup) || [];
        const filterValue = (moduleFilter.value || "").trim().toLowerCase();
        const previousSelectedModule = String(nodeSelect.value || "").trim();
        moduleBadgeLoadToken += 1;
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
        for (const entry of moduleEntries) {
            const moduleName = String(entry?.module || "unknown");
            countMap.set(moduleName, Number(entry?.count) || 0);
        }
        for (const moduleName of modules) {
            const opt = document.createElement("option");
            opt.value = moduleName;
            const count = countMap.has(moduleName)
                ? (countMap.get(moduleName) || 0)
                : (grouped.get(moduleName) || []).length;
            moduleCounts.set(moduleName, count);
            moduleOptions.set(moduleName, opt);
            opt.textContent = formatModuleOption(moduleName, count, null);
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
        loadModuleBadges(selectedGroup, modules);
        syncUpdateAllButton();
    };

    /** Handle `fillGroupSelect` workflow step. */
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

    /** Handle `renderNodeList` workflow step. */
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

    /** Handle `renderModuleInfo` workflow step. */
    const renderModuleInfo = (info) => {
        moduleInfo.innerHTML = "";
        if (!info || nodeSelect.value === "-1") {
            return;
        }

        const card = document.createElement("div");
        card.className = "alexz-mod-picker-module-card";
        const selectedModule = nodeSelect.value;
        const nodeCount = moduleCounts.get(selectedModule) || 0;
        if (updatedModulesSession.has(selectedModule)) {
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

        if (info.updated_between_runs && (info.startup_prev_commit_short || info.startup_new_commit_short)) {
            const updateRow = document.createElement("div");
            updateRow.className = "alexz-mod-picker-module-row notice";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Updated between runs:";
            const valueEl = document.createElement("span");
            const prev = info.startup_prev_commit_short || "unknown";
            const next = info.startup_new_commit_short || "unknown";
            const at = info.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
            valueEl.textContent = `${prev} -> ${next}${at}`;
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

    /** Handle `loadModuleInfo` workflow step. */
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

    /** Handle `loadCatalog` workflow step. */
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

/** Handle `unbindContainerOwnershipSync` workflow step. */
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
    if (state.rootCheckInterval) {
        window.clearInterval(state.rootCheckInterval);
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

/** Handle `bindContainerOwnershipSync` workflow step. */
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

    // Periodically check if root got detached from container and re-attach if needed
    const rootCheckInterval = window.setInterval(() => {
        const isDetached = !root.isConnected || root.parentElement !== container;
        if (isDetached) {
            // Root was detached, re-attach it to container immediately
            if (root.parentElement && root.parentElement !== container) {
                try {
                    root.parentElement.removeChild(root);
                } catch (e) {
                    // Already removed or parent changed
                }
            }
            if (container && container.isConnected) {
                // Insert at the beginning to ensure visibility
                if (container.firstChild && container.firstChild !== root) {
                    container.insertBefore(root, container.firstChild);
                } else if (!container.firstChild) {
                    container.appendChild(root);
                }
                // Force display
                root.style.display = "";
            }
        }
    }, 100);

    window[CONTAINER_SYNC_STATE_KEY] = {
        containerObserver,
        sidebarObserver,
        onClick: onInteraction,
        onPointerDown,
        rootCheckInterval,
    };
    // Always show root on initialization - sync will determine visibility on next tick
    root.style.display = "";
    sync();
}

/** Handle `attachFallbackButton` workflow step. */
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

/** Handle `cleanupFallbackButtons` workflow step. */
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
