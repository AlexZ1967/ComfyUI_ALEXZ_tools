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

import {
    getActiveSidebarTabId,
    resolveSidebarButtonFromEvent,
    isSidebarContextEvent,
    isOwnButtonSelected,
    inferFallbackTabIdFromContext,
    inferTabIdFromButton,
    getContainerState,
} from "./orchestration/module_node_picker_tab_relay_helpers.js";

const TAB_RELAY_STATE_KEY = "__alexz_module_picker_tab_relay_state_v2__";

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
