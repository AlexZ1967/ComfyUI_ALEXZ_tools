/**
 * Module: web/module_node_picker_tab_relay.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module Node Picker tab relay helper.
 *
 * Purpose:
 *   Synchronizes panel attachment/visibility with sidebar tab state and reports diagnostics.
 */

const TAB_RELAY_STATE_KEY = "__alexz_module_picker_tab_relay_state_v2__";

/**
 * Return Sidebar API object for current ComfyUI build shape.
 */
function getSidebarApi(app) {
    const manager = app?.extensionManager;
    return manager?.sidebarTab || manager || null;
}

/**
 * Read currently active sidebar tab id in a version-tolerant way.
 */
function getActiveSidebarTabId(app) {
    const sidebar = getSidebarApi(app);
    if (!sidebar) {
        return "";
    }
    const active = sidebar.activeSidebarTabId ?? sidebar.activeSidebarTab ?? "";
    return String(active || "");
}

/**
 * Extract logical tab id from sidebar button attributes/classes.
 */
function extractTabIdFromButton(buttonEl) {
    if (!(buttonEl instanceof Element)) {
        return "";
    }
    const idAttr = String(buttonEl.getAttribute("id") || "").trim();
    if (idAttr.endsWith("-tab-button")) {
        return idAttr.slice(0, -"-tab-button".length);
    }
    const dataTabId = String(
        buttonEl.getAttribute("data-tab-id")
        || buttonEl.getAttribute("data-sidebar-tab")
        || buttonEl.getAttribute("data-tab")
        || ""
    ).trim();
    if (dataTabId) {
        return dataTabId;
    }
    for (const cls of Array.from(buttonEl.classList || [])) {
        if (cls.endsWith("-tab-button")) {
            return cls.slice(0, -"-tab-button".length);
        }
    }
    return "";
}

/**
 * Resolve sidebar tab-like button element from DOM event target/path.
 */
function resolveSidebarButtonFromEvent(event) {
    const direct = event?.target;
    if (direct instanceof Element) {
        const closest = direct.closest(
            ".side-bar-button, [class*='-tab-button'], [role='tab'], button[aria-label], button[title]"
        );
        if (closest) {
            return closest;
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
            if (extractTabIdFromButton(item)) {
                return item;
            }
            if (String(item.getAttribute("role") || "").toLowerCase() === "tab") {
                return item;
            }
        }
    }
    return null;
}

/**
 * Heuristic check whether event originates from sidebar-related UI context.
 */
function isSidebarContextEvent(event) {
    const direct = event?.target;
    if (direct instanceof Element) {
        const sidebarMatch = direct.closest(
            ".side-bar, .sidebar, .comfy-sidebar, [class*='sidebar'], [class*='side-bar']"
        );
        if (sidebarMatch) {
            return true;
        }
    }
    if (typeof event?.composedPath === "function") {
        for (const item of event.composedPath()) {
            if (!(item instanceof Element)) {
                continue;
            }
            const cls = String(item.className || "").toLowerCase();
            const id = String(item.id || "").toLowerCase();
            if (
                cls.includes("sidebar")
                || cls.includes("side-bar")
                || id.includes("sidebar")
                || id.includes("side-bar")
            ) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Return selected state for this extension sidebar button.
 */
function isOwnButtonSelected(sidebarTabId) {
    const ownBtn = document.querySelector(`.${sidebarTabId}-tab-button`);
    if (!ownBtn) {
        return null;
    }
    return ownBtn.classList.contains("side-bar-button-selected");
}

/**
 * Collect known sidebar tab descriptors from multiple ComfyUI API shapes.
 */
function collectSidebarTabDescriptors(app) {
    const sidebar = getSidebarApi(app);
    const pools = [
        sidebar?.tabs,
        sidebar?.sidebarTabs,
        sidebar?.tabConfigs,
        app?.extensionManager?.tabs,
        app?.extensionManager?.sidebarTabs,
    ];
    const out = [];
    for (const pool of pools) {
        if (!Array.isArray(pool)) {
            continue;
        }
        for (const item of pool) {
            if (!item || typeof item !== "object") {
                continue;
            }
            const id = String(item.id || item.tabId || item.key || "").trim();
            const title = String(item.title || item.label || item.tooltip || "").trim();
            if (id) {
                out.push({ id, title });
            }
        }
    }
    return out;
}

/**
 * Check whether sidebar exposes tab with the requested id.
 */
function hasSidebarTabId(app, tabId) {
    if (!tabId) {
        return false;
    }
    const descriptors = collectSidebarTabDescriptors(app);
    return descriptors.some((x) => String(x.id || "") === String(tabId));
}

/**
 * Infer tab id when clicked button has no stable id/class markers.
 */
function inferFallbackTabIdFromContext(app, event, sidebarTabId) {
    const knownNodesMapId = "easyuse_nodes_map";
    const descriptors = collectSidebarTabDescriptors(app);
    const composed = typeof event?.composedPath === "function" ? event.composedPath() : [];
    const chunks = [];
    for (const item of composed) {
        if (!(item instanceof Element)) {
            continue;
        }
        const cls = String(item.className || "").toLowerCase();
        const text = String(item.textContent || "").toLowerCase().trim();
        const title = String(item.getAttribute("title") || item.getAttribute("aria-label") || "").toLowerCase().trim();
        if (cls) {
            chunks.push(cls);
        }
        if (text) {
            chunks.push(text);
        }
        if (title) {
            chunks.push(title);
        }
    }
    const hay = chunks.join(" | ");
    if (hay.includes("nodesmap") || hay.includes("nodes map")) {
        if (hasSidebarTabId(app, knownNodesMapId)) {
            return knownNodesMapId;
        }
    }
    if (hay.includes("pi-sitemap") && hasSidebarTabId(app, knownNodesMapId)) {
        return knownNodesMapId;
    }
    // Generic fallback by title match from descriptors.
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (!id || id === sidebarTabId) {
            continue;
        }
        if (title.includes("nodesmap") || title.includes("nodes map")) {
            return id;
        }
    }
    return "";
}

/**
 * Infer target tab id from button metadata, title, and known variants.
 */
function inferTabIdFromButton(app, button) {
    const explicit = extractTabIdFromButton(button);
    if (explicit) {
        return explicit;
    }
    const label = String(
        button.getAttribute("title")
        || button.getAttribute("aria-label")
        || button.textContent
        || ""
    ).trim().toLowerCase();
    if (!label) {
        return "";
    }
    const descriptors = collectSidebarTabDescriptors(app);
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (title && (title === label || title.includes(label) || label.includes(title))) {
            return id;
        }
    }
    if (label.includes("nodesmap") || label.includes("nodes map")) {
        // Known EasyUse tab id variants seen in different builds.
        return "easyuse_nodes_map";
    }
    return "";
}

/**
 * Return compact DOM snapshot of picker container children for diagnostics.
 */
function getContainerState(root) {
    const container = root?.parentElement;
    if (!(container instanceof Element)) {
        return { childCount: 0, childShort: "n/a" };
    }
    const out = [];
    let childCount = 0;
    for (const node of container.childNodes) {
        childCount += 1;
        if (node === root) {
            out.push("ROOT");
            continue;
        }
        if (node.nodeType === Node.TEXT_NODE) {
            const txt = String(node.textContent || "").trim();
            out.push(txt ? `TXT:${txt.slice(0, 20)}` : "TXT:blank");
            continue;
        }
        if (node instanceof Element) {
            const cls = String(node.className || "").trim();
            out.push(cls ? `${node.nodeName}.${cls.split(/\s+/).slice(0, 2).join(".")}` : node.nodeName);
        } else {
            out.push(String(node.nodeName || "NODE"));
        }
    }
    return { childCount, childShort: out.slice(0, 10).join(" | ") || "n/a" };
}

/**
 * Unbind all relay listeners/intervals and clear global relay state.
 */
export function unbindModuleNodesTabRelay() {
    const state = window[TAB_RELAY_STATE_KEY];
    if (!state) {
        return;
    }
    if (state.relayTimer) {
        window.clearTimeout(state.relayTimer);
    }
    if (state.tickInterval) {
        window.clearInterval(state.tickInterval);
    }
    if (state.bindButtonsInterval) {
        window.clearInterval(state.bindButtonsInterval);
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
    if (state.onFocusIn) {
        document.removeEventListener("focusin", state.onFocusIn, true);
    }
    if (Array.isArray(state.boundButtons)) {
        for (const item of state.boundButtons) {
            if (!item?.el || !item?.handler) {
                continue;
            }
            item.el.removeEventListener("pointerdown", item.handler, true);
            item.el.removeEventListener("mousedown", item.handler, true);
            item.el.removeEventListener("click", item.handler, true);
        }
    }
    window[TAB_RELAY_STATE_KEY] = null;
}

/**
 * Bind lightweight tab relay that keeps picker root attached only when its
 * sidebar tab is active, while reporting diagnostics to callback.
 */
export function bindModuleNodesTabRelay({ app, root, sidebarTabId, onDiag }) {
    unbindModuleNodesTabRelay();

    let lastClickedTabId = "";
    let pendingForeignTabId = "";
    let pendingForeignTabAt = 0;
    let lastDiagSig = "";
    const boundButtons = [];
    let relayTimer = 0;
    const homeContainer = root.parentElement instanceof Element ? root.parentElement : null;
    const FOREIGN_TAB_HIDE_MS = 1600;

    /**
     * Emit deduplicated diagnostics payload to panel callback.
     */
    const emitDiag = (reason, clickedTabId = "") => {
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const containerState = getContainerState(root);
        const diag = {
            reason,
            activeTabId: getActiveSidebarTabId(app) || "n/a",
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

    /**
     * Re-attach picker root into home container when needed.
     */
    const ensureRootAttached = () => {
        if (root.isConnected) {
            return true;
        }
        if (homeContainer && homeContainer.isConnected) {
            homeContainer.appendChild(root);
            return true;
        }
        return false;
    };

    /**
     * Detach picker root from DOM when another sidebar tab should own panel area.
     */
    const ensureRootDetached = () => {
        if (!root.isConnected) {
            return true;
        }
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        return true;
    };

    /**
     * Start temporary foreign-tab protection window to avoid stale ownership flicker.
     */
    const markForeignTabIntent = (tabId) => {
        const normalized = String(tabId || "").trim() || "(unknown-other-tab)";
        pendingForeignTabId = normalized;
        pendingForeignTabAt = Date.now();
        lastClickedTabId = normalized;
        ensureRootDetached();
    };

    /**
     * Reset temporary foreign-tab protection window.
     */
    const clearForeignTabIntent = () => {
        pendingForeignTabId = "";
        pendingForeignTabAt = 0;
    };

    /**
     * Return true while we intentionally keep root detached after foreign click.
     */
    const isForeignIntentActive = () => {
        if (!pendingForeignTabId) {
            return false;
        }
        if (Date.now() - pendingForeignTabAt > FOREIGN_TAB_HIDE_MS) {
            clearForeignTabIntent();
            return false;
        }
        return true;
    };

    /**
     * Synchronize picker root visibility/attachment with current sidebar state.
     */
    const syncVisibility = (reason, clickedTabId = "") => {
        const activeTabId = getActiveSidebarTabId(app);
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const switchedAway = (activeTabId && activeTabId !== sidebarTabId) || ownSelected === false;
        if (switchedAway) {
            clearForeignTabIntent();
        }
        const foreignIntentActive = isForeignIntentActive();
        let shouldShow = ownSelected === true || activeTabId === sidebarTabId;
        if (shouldShow && foreignIntentActive) {
            shouldShow = false;
        }
        if (shouldShow) {
            ensureRootAttached();
            root.style.display = "";
        } else {
            ensureRootDetached();
        }
        const effectiveReason = foreignIntentActive && reason !== "relay_own_tab_click"
            ? "relay_wait_foreign_tab"
            : reason;
        emitDiag(effectiveReason, clickedTabId || pendingForeignTabId);
    };

    /**
     * Process sidebar button interaction and schedule relay correction if needed.
     */
    const processTabButton = (button) => {
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId) {
            return;
        }
        if (tabId === sidebarTabId) {
            clearForeignTabIntent();
            lastClickedTabId = sidebarTabId;
            syncVisibility("relay_own_tab_click", sidebarTabId);
            return;
        }
        markForeignTabIntent(tabId);
        syncVisibility("relay_foreign_tab_click", tabId);
        if (isOwnButtonSelected(sidebarTabId) !== true) {
            return;
        }
        if (relayTimer) {
            window.clearTimeout(relayTimer);
        }
        relayTimer = window.setTimeout(() => {
            relayTimer = 0;
            const liveState = window[TAB_RELAY_STATE_KEY];
            if (liveState) {
                liveState.relayTimer = 0;
            }
            const activeTabId = getActiveSidebarTabId(app);
            const ownSelected = isOwnButtonSelected(sidebarTabId);
            if (activeTabId === tabId || ownSelected === false) {
                clearForeignTabIntent();
                syncVisibility("relay_native_ok", tabId);
                return;
            }
            // Do not force tab activation from relay. Only re-evaluate visibility.
            syncVisibility("relay_pending_switch", tabId);
        }, 60);
        const liveState = window[TAB_RELAY_STATE_KEY];
        if (liveState) {
            liveState.relayTimer = relayTimer;
        }
    };

    /**
     * Global event handler used to detect sidebar tab interactions.
     */
    const handleEvent = (event) => {
        const direct = event?.target;
        if (direct instanceof Element && root.contains(direct)) {
            return;
        }
        const button = resolveSidebarButtonFromEvent(event);
        if (!button) {
            if (isOwnButtonSelected(sidebarTabId) === true && isSidebarContextEvent(event)) {
                const fallbackTabId = inferFallbackTabIdFromContext(app, event, sidebarTabId);
                markForeignTabIntent(fallbackTabId || "(unknown-other-tab)");
                syncVisibility("relay_unknown_tab_click", fallbackTabId || "(unknown-other-tab)");
            }
            return;
        }
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId && isOwnButtonSelected(sidebarTabId) === true) {
            const ownMarker = `${sidebarTabId}-tab-button`;
            const isOwn = button.classList?.contains(ownMarker)
                || String(button.getAttribute("id") || "") === ownMarker;
            if (!isOwn) {
                const fallbackTabId = inferFallbackTabIdFromContext(app, event, sidebarTabId);
                markForeignTabIntent(fallbackTabId || "(unknown-other-tab)");
                syncVisibility("relay_unknown_tab_click", fallbackTabId || "(unknown-other-tab)");
            }
            return;
        }
        processTabButton(button);
    };

    /**
     * Bind direct handlers to discovered tab buttons (supports dynamic sidebars).
     */
    const bindDirectButtonListeners = () => {
        const buttons = Array.from(document.querySelectorAll(".side-bar-button, [class*='-tab-button']"));
        for (const button of buttons) {
            if (!(button instanceof Element)) {
                continue;
            }
            const tabId = inferTabIdFromButton(app, button);
            if (!tabId || tabId === sidebarTabId) {
                continue;
            }
            if (boundButtons.some((x) => x.el === button)) {
                continue;
            }
            const handler = () => processTabButton(button);
            button.addEventListener("pointerdown", handler, true);
            button.addEventListener("mousedown", handler, true);
            button.addEventListener("click", handler, true);
            boundButtons.push({ el: button, handler });
        }
    };

    const onPointerDown = (event) => handleEvent(event);
    const onMouseDown = (event) => handleEvent(event);
    const onClick = (event) => handleEvent(event);
    const onKeyUp = () => syncVisibility("relay_keyup");
    const onFocusIn = (event) => handleEvent(event);

    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("click", onClick, true);
    document.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("focusin", onFocusIn, true);

    bindDirectButtonListeners();
    const bindButtonsInterval = window.setInterval(bindDirectButtonListeners, 1000);
    const tickInterval = window.setInterval(() => {
        syncVisibility("relay_tick");
    }, 220);

    syncVisibility("relay_init");
    window[TAB_RELAY_STATE_KEY] = {
        relayTimer,
        tickInterval,
        bindButtonsInterval,
        onPointerDown,
        onMouseDown,
        onClick,
        onKeyUp,
        onFocusIn,
        boundButtons,
    };
}
