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

import { createModuleNodePickerTabRelayRuntime } from "./orchestration/relay/module_node_picker_tab_relay_runtime.js";
import { startModuleNodePickerRelayTickLoop } from "./orchestration/relay/module_node_picker_tab_relay_tick.js";
import { createModuleNodePickerRelayIntentController } from "./orchestration/relay/module_node_picker_tab_relay_intent.js";

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
    if (state.tickTimer) {
        window.clearTimeout(state.tickTimer);
    }
    if (typeof state.stopTick === "function") {
        state.stopTick();
    }
    if (typeof state.stopIntent === "function") {
        state.stopIntent();
    }
    // Backward-compat cleanup for previous relay state shape.
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

    const bindToken = Symbol("alexz_module_picker_relay_bind");
    const relayRuntime = createModuleNodePickerTabRelayRuntime({
        app,
        root,
        sidebarTabId,
        onDiag,
    });
    const isCurrentBinding = () => {
        const state = window[TAB_RELAY_STATE_KEY];
        return Boolean(state && state.bindToken === bindToken);
    };

    const relayIntent = createModuleNodePickerRelayIntentController({
        app,
        root,
        sidebarTabId,
        relayRuntime,
        isCurrentBinding,
        getLiveState: () => window[TAB_RELAY_STATE_KEY],
    });

    const supportsPointer = typeof window !== "undefined" && "PointerEvent" in window;
    const onPointerDown = supportsPointer ? ((event) => relayIntent.handleEvent(event)) : null;
    const onMouseDown = supportsPointer ? null : ((event) => relayIntent.handleEvent(event));
    const onKeyUp = (event) => relayIntent.onKeyUp(event);
    const onVisibilityChange = () => relayIntent.onVisibilityChange();
    const onPageShow = () => relayIntent.onPageShow();

    const relayState = {
        bindToken,
        relayTimer: 0,
        tickTimer: 0,
        onPointerDown,
        onMouseDown,
        onKeyUp,
        onVisibilityChange,
        onPageShow,
        stopIntent: () => relayIntent.dispose?.(),
        stopTick: () => {},
        dispose: () => relayRuntime.dispose?.(),
    };
    window[TAB_RELAY_STATE_KEY] = relayState;

    if (onPointerDown) {
        document.addEventListener("pointerdown", onPointerDown, true);
    }
    if (onMouseDown) {
        document.addEventListener("mousedown", onMouseDown, true);
    }
    document.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("visibilitychange", onVisibilityChange, true);
    window.addEventListener("pageshow", onPageShow, true);

    relayState.stopTick = startModuleNodePickerRelayTickLoop({
        isCurrentBinding,
        relayRuntime,
        sidebarTabId,
        setTickTimer: (timer) => {
            const liveState = window[TAB_RELAY_STATE_KEY];
            if (liveState) {
                liveState.tickTimer = timer;
            }
        },
    });
    relayRuntime.syncVisibility("relay_init");
}
