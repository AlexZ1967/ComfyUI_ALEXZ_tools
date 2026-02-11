/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_facade.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module Node Picker tab relay facade implementation.
 *
 * Purpose:
 *   Encapsulates relay bind/unbind orchestration and global relay runtime
 *   state management while keeping the root entrypoint module lightweight.
 */

import { createModuleNodePickerTabRelayRuntime } from "./module_node_picker_tab_relay_runtime.js";
import { startModuleNodePickerRelayTickLoop } from "./module_node_picker_tab_relay_tick.js";
import { createModuleNodePickerRelayIntentController } from "./module_node_picker_tab_relay_intent.js";
import {
    getModuleNodePickerRelayState,
    setModuleNodePickerRelayState,
    clearModuleNodePickerRelayState,
} from "./module_node_picker_tab_relay_state.js";
import { RELAY_REASON_INIT } from "./module_node_picker_tab_relay_constants.js";
import {
    createRelayDomEventHandlers,
    bindRelayDomEvents,
} from "./module_node_picker_tab_relay_events.js";
import {
    createRelayBindState,
    disposeRelayBindState,
} from "./module_node_picker_tab_relay_lifecycle.js";

/**
 * Unbind all relay listeners/intervals and clear global relay state.
 */
export function unbindModuleNodesTabRelayFacade() {
    if (typeof window === "undefined" || typeof document === "undefined") {
        return;
    }
    const state = getModuleNodePickerRelayState();
    if (!state) {
        return;
    }
    disposeRelayBindState(state);
    clearModuleNodePickerRelayState();
}

/**
 * Bind lightweight tab relay that keeps picker root attached only when its
 * sidebar tab is active, while reporting diagnostics to callback.
 */
export function bindModuleNodesTabRelayFacade({ app, root, mountHost, sidebarTabId, onDiag }) {
    if (typeof window === "undefined" || typeof document === "undefined") {
        return;
    }
    const relayRoot = root instanceof Element ? root : null;
    if (!relayRoot) {
        return;
    }
    const relayMountHost = mountHost instanceof Element
        ? mountHost
        : (relayRoot.parentElement instanceof Element ? relayRoot.parentElement : null);
    unbindModuleNodesTabRelayFacade();

    const bindToken = Symbol("alexz_module_picker_relay_bind");
    const relayRuntime = createModuleNodePickerTabRelayRuntime({
        app,
        root: relayRoot,
        mountHost: relayMountHost,
        sidebarTabId,
        onDiag,
    });
    const isCurrentBinding = () => {
        const state = getModuleNodePickerRelayState();
        return Boolean(state && state.bindToken === bindToken);
    };

    const relayIntent = createModuleNodePickerRelayIntentController({
        app,
        root: relayRoot,
        sidebarTabId,
        relayRuntime,
        isCurrentBinding,
        getLiveState: () => getModuleNodePickerRelayState(),
    });

    const handlers = createRelayDomEventHandlers(relayIntent);

    const relayState = createRelayBindState({
        bindToken,
        handlers,
        stopIntent: () => relayIntent.dispose?.(),
        dispose: () => relayRuntime.dispose?.(),
    });
    setModuleNodePickerRelayState(relayState);
    bindRelayDomEvents(handlers);

    relayState.stopTick = startModuleNodePickerRelayTickLoop({
        isCurrentBinding,
        relayRuntime,
        sidebarTabId,
        setTickTimer: (timer) => {
            const liveState = getModuleNodePickerRelayState();
            if (liveState) {
                liveState.tickTimer = timer;
            }
        },
    });
    relayRuntime.syncVisibility(RELAY_REASON_INIT);
}
