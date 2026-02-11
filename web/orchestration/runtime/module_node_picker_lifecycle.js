/**
 * Module: web/orchestration/runtime/module_node_picker_lifecycle.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Picker instance lifecycle and disposal controller.
 *
 * Purpose:
 *   Centralizes dispose-time cleanup for Module Node Picker so stale async/UI
 *   work cannot leak across sidebar re-renders.
 */

/**
 * Create lifecycle controller for one picker render instance.
 */
export function createModuleNodePickerLifecycle(context = {}) {
    const getCatalogController = typeof context?.getCatalogController === "function"
        ? context.getCatalogController
        : () => null;
    const getPollingController = typeof context?.getPollingController === "function"
        ? context.getPollingController
        : () => null;
    const getUnbindPickerEvents = typeof context?.getUnbindPickerEvents === "function"
        ? context.getUnbindPickerEvents
        : () => () => {};
    const getCancelStartupLoad = typeof context?.getCancelStartupLoad === "function"
        ? context.getCancelStartupLoad
        : () => () => {};
    const getDebugUi = typeof context?.getDebugUi === "function"
        ? context.getDebugUi
        : () => null;
    const getProcessUi = typeof context?.getProcessUi === "function"
        ? context.getProcessUi
        : () => null;
    const getApiClient = typeof context?.getApiClient === "function"
        ? context.getApiClient
        : () => null;
    const unbindTabRelay = typeof context?.unbindTabRelay === "function"
        ? context.unbindTabRelay
        : () => {};
    const container = context?.container;
    const cleanupKey = String(context?.cleanupKey || "");

    let pickerDisposed = false;

    const isPickerAlive = () => !pickerDisposed;

    const dispose = () => {
        if (pickerDisposed) {
            return;
        }
        pickerDisposed = true;
        getCatalogController()?.bumpRequestTokens?.();
        getPollingController()?.invalidate?.();
        try {
            getUnbindPickerEvents()?.();
        } catch (_err) {
            // Ignore stale event-unbind errors.
        }
        try {
            getCancelStartupLoad()?.();
        } catch (_err) {
            // Ignore stale startup-load cancellation errors.
        }
        try {
            getDebugUi()?.dispose?.();
        } catch (_err) {
            // Ignore stale debug-ui dispose errors.
        }
        try {
            getProcessUi()?.dispose?.();
        } catch (_err) {
            // Ignore stale process-ui dispose errors.
        }
        getApiClient()?.dispose?.();
        unbindTabRelay();
        if (container && cleanupKey && container?.[cleanupKey] === dispose) {
            container[cleanupKey] = null;
        }
    };

    if (container && cleanupKey) {
        container[cleanupKey] = dispose;
    }

    return {
        isPickerAlive,
        dispose,
        isDisposed: () => pickerDisposed,
    };
}
