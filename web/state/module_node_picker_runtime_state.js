/**
 * Module: web/state/module_node_picker_runtime_state.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Session/runtime state helpers for Module Node Picker frontend.
 *
 * Purpose:
 *   Centralizes in-memory runtime markers and storage-backed preferences
 *   used by picker restore flows, while keeping behavior deterministic.
 */

/**
 * Get per-page runtime picker state shared across picker re-renders.
 *
 * The state lives in browser memory (`window`) and is reset on full reload.
 */
export function getRuntimePickerState(windowObj, runtimeStateKey) {
    const existing = windowObj?.[runtimeStateKey];
    if (existing && typeof existing === "object") {
        return existing;
    }
    const created = {
        customStatusChecked: false,
        pendingCustomRefresh: false,
        pendingUpdate: false,
        pendingComfyInfoRefresh: false,
        comfyStatusChecked: false,
        comfyLastInfo: null,
    };
    if (windowObj && runtimeStateKey) {
        windowObj[runtimeStateKey] = created;
    }
    return created;
}

/**
 * Remove legacy localStorage flags from earlier persistence strategy.
 */
export function clearLegacyPersistentFlags(windowObj, keys = {}) {
    try {
        windowObj?.localStorage?.removeItem(keys.customStatusCheckedKey || "");
        windowObj?.localStorage?.removeItem(keys.pendingCustomRefreshKey || "");
        windowObj?.localStorage?.removeItem(keys.pendingUpdateKey || "");
    } catch (_err) {
        // Ignore storage failures; legacy flags are best-effort cleanup.
    }
}

/**
 * Create runtime status accessors bound to runtime picker state object.
 */
export function createRuntimeStatusAccessors(runtimePickerState) {
    const state = runtimePickerState && typeof runtimePickerState === "object"
        ? runtimePickerState
        : {};

    const loadCustomStatusChecked = () => Boolean(state.customStatusChecked);
    const saveCustomStatusChecked = (checked) => {
        state.customStatusChecked = Boolean(checked);
    };
    const loadComfyStatusChecked = () => Boolean(state.comfyStatusChecked);
    const saveComfyStatusChecked = (checked) => {
        state.comfyStatusChecked = Boolean(checked);
    };
    const loadComfyInfoSnapshot = () => {
        const info = state.comfyLastInfo;
        if (!info || typeof info !== "object") {
            return null;
        }
        return { ...info };
    };
    const saveComfyInfoSnapshot = (info) => {
        state.comfyLastInfo = (info && typeof info === "object") ? { ...info } : null;
    };
    const hasPendingCustomRefresh = () => Boolean(state.pendingCustomRefresh);
    const setPendingCustomRefresh = (pending) => {
        state.pendingCustomRefresh = Boolean(pending);
    };
    const clearPendingCustomRefresh = () => setPendingCustomRefresh(false);
    const hasPendingUpdate = () => Boolean(state.pendingUpdate);
    const setPendingUpdate = (pending) => {
        state.pendingUpdate = Boolean(pending);
    };
    const clearPendingUpdate = () => setPendingUpdate(false);
    const hasPendingComfyInfoRefresh = () => Boolean(state.pendingComfyInfoRefresh);
    const setPendingComfyInfoRefresh = (pending) => {
        state.pendingComfyInfoRefresh = Boolean(pending);
    };
    const clearPendingComfyInfoRefresh = () => setPendingComfyInfoRefresh(false);

    return {
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
    };
}

/**
 * Read ComfyUI check mode from localStorage (`releases` default).
 */
export function loadComfyCheckMode(windowObj, storageKey) {
    try {
        const raw = windowObj?.localStorage?.getItem(storageKey);
        return String(raw || "releases").trim().toLowerCase() === "commits"
            ? "commits"
            : "releases";
    } catch (_err) {
        return "releases";
    }
}

/**
 * Persist ComfyUI check mode to localStorage (sanitized to `releases|commits`).
 */
export function saveComfyCheckMode(windowObj, storageKey, mode) {
    const normalized = String(mode || "releases").trim().toLowerCase() === "commits"
        ? "commits"
        : "releases";
    try {
        windowObj?.localStorage?.setItem(storageKey, normalized);
    } catch (_err) {
        // Ignore storage failures and keep runtime value only.
    }
}
