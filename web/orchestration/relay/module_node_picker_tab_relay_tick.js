/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_tick.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Passive tick-loop orchestration for Module Node Picker tab relay.
 *
 * Purpose:
 *   Encapsulates adaptive relay tick scheduling and inactive-tab throttling
 *   so tab relay wiring stays focused on event bindings and state ownership.
 */

import { isOwnButtonSelected } from "./module_node_picker_tab_relay_helpers.js";

/**
 * Start adaptive relay tick loop and return stop callback.
 */
export function startModuleNodePickerRelayTickLoop(context = {}) {
    const isCurrentBinding = typeof context?.isCurrentBinding === "function"
        ? context.isCurrentBinding
        : () => false;
    const relayRuntime = context?.relayRuntime || null;
    const sidebarTabId = String(context?.sidebarTabId || "");
    const setTickTimer = typeof context?.setTickTimer === "function"
        ? context.setTickTimer
        : () => {};

    let passiveTickBudget = 0;
    let timer = 0;

    const schedule = (delayMs) => {
        timer = window.setTimeout(runTick, delayMs);
        setTickTimer(timer);
    };

    const runTick = () => {
        if (!isCurrentBinding()) {
            return;
        }
        let nextDelayMs = 500;
        if (document.visibilityState === "hidden") {
            nextDelayMs = 900;
            schedule(nextDelayMs);
            return;
        }
        const ownTabSelected = isOwnButtonSelected(sidebarTabId) === true;
        const keepFastTick = ownTabSelected || relayRuntime?.hasPendingForeignIntent?.();
        if (!keepFastTick) {
            passiveTickBudget += 1;
            // When picker tab is inactive, run a sparse maintenance tick
            // instead of syncing on every timer pulse.
            if (passiveTickBudget < 6) {
                nextDelayMs = 900;
                schedule(nextDelayMs);
                return;
            }
            passiveTickBudget = 0;
            nextDelayMs = 900;
        } else {
            passiveTickBudget = 0;
        }
        relayRuntime?.syncVisibility?.("relay_tick");
        schedule(nextDelayMs);
    };

    schedule(500);
    return () => {
        if (timer) {
            window.clearTimeout(timer);
            timer = 0;
            setTickTimer(0);
        }
    };
}
