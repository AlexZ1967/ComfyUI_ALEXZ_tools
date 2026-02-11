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
    isOwnButtonSelected,
    inferTabIdFromButton,
} from "./orchestration/module_node_picker_tab_relay_helpers.js";
import { createModuleNodePickerTabRelayRuntime } from "./orchestration/module_node_picker_tab_relay_runtime.js";

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
    if (state.onPointerDown) {
        document.removeEventListener("pointerdown", state.onPointerDown, true);
    }
    if (state.onMouseDown) {
        document.removeEventListener("mousedown", state.onMouseDown, true);
    }
    if (state.onKeyUp) {
        document.removeEventListener("keyup", state.onKeyUp, true);
    }
    if (state.onVisibilityChange) {
        document.removeEventListener("visibilitychange", state.onVisibilityChange, true);
    }
    if (state.onPageShow) {
        window.removeEventListener("pageshow", state.onPageShow, true);
    }
    if (typeof state.dispose === "function") {
        state.dispose();
    }
    window[TAB_RELAY_STATE_KEY] = null;
}

/**
 * Bind lightweight tab relay that keeps picker root attached only when its
 * sidebar tab is active, while reporting diagnostics to callback.
 */
export function bindModuleNodesTabRelay({ app, root, sidebarTabId, onDiag }) {
    unbindModuleNodesTabRelay();

    let relayTimer = 0;
    let passiveTickBudget = 0;
    const relayRuntime = createModuleNodePickerTabRelayRuntime({
        app,
        root,
        sidebarTabId,
        onDiag,
    });

    /**
     * Process sidebar button interaction and schedule relay correction if needed.
     */
    const processTabButton = (button) => {
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId) {
            return;
        }
        if (tabId === sidebarTabId) {
            relayRuntime.onOwnTabClick(sidebarTabId);
            return;
        }
        relayRuntime.onForeignTabClick(tabId);
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
                relayRuntime.clearForeignIntent();
                relayRuntime.syncVisibility("relay_native_ok", tabId);
                return;
            }
            // Do not force tab activation from relay. Only re-evaluate visibility.
            relayRuntime.syncVisibility("relay_pending_switch", tabId);
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
        // Ignore any interaction that originates inside picker root,
        // including text-node targets emitted by some browser/event paths.
        if (direct instanceof Node && root.contains(direct)) {
            return;
        }
        if (typeof event?.composedPath === "function" && event.composedPath().includes(root)) {
            return;
        }
        const button = resolveSidebarButtonFromEvent(event);
        if (!button) {
            return;
        }
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId) {
            return;
        }
        processTabButton(button);
    };

    const supportsPointer = typeof window !== "undefined" && "PointerEvent" in window;
    const onPointerDown = supportsPointer ? ((event) => handleEvent(event)) : null;
    const onMouseDown = supportsPointer ? null : ((event) => handleEvent(event));
    const onKeyUp = () => relayRuntime.syncVisibility("relay_keyup");
    const onVisibilityChange = () => relayRuntime.syncVisibility("relay_visibility");
    const onPageShow = () => relayRuntime.syncVisibility("relay_pageshow");

    if (onPointerDown) {
        document.addEventListener("pointerdown", onPointerDown, true);
    }
    if (onMouseDown) {
        document.addEventListener("mousedown", onMouseDown, true);
    }
    document.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("visibilitychange", onVisibilityChange, true);
    window.addEventListener("pageshow", onPageShow, true);

    const tickInterval = window.setInterval(() => {
        if (document.visibilityState === "hidden") {
            return;
        }
        const ownTabSelected = isOwnButtonSelected(sidebarTabId) === true;
        const keepFastTick = ownTabSelected || relayRuntime.hasPendingForeignIntent();
        if (!keepFastTick) {
            passiveTickBudget += 1;
            // When picker tab is inactive, run a sparse maintenance tick
            // instead of syncing on every timer pulse.
            if (passiveTickBudget < 6) {
                return;
            }
            passiveTickBudget = 0;
        } else {
            passiveTickBudget = 0;
        }
        relayRuntime.syncVisibility("relay_tick");
    }, 500);

    relayRuntime.syncVisibility("relay_init");
    window[TAB_RELAY_STATE_KEY] = {
        relayTimer,
        tickInterval,
        onPointerDown,
        onMouseDown,
        onKeyUp,
        onVisibilityChange,
        onPageShow,
        dispose: () => relayRuntime.dispose?.(),
    };
}
