/**
 * Module: web/state/store.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Lightweight state store for Module Node Picker.
 *
 * Purpose:
 *   Provides a single source of truth for minimal picker UI state and
 *   persistence for selected group/module and debug flag.
 */

const DEFAULT_SELECTED_GROUP_KEY = "alexz_module_picker_selected_group";
const DEFAULT_SELECTED_MODULE_KEY = "alexz_module_picker_selected_module";
const DEFAULT_DEBUG_KEY = "alexz_module_picker_debug";

/**
 * Safe read helper for localStorage string values.
 */
function readStoredString(key, fallbackValue) {
    try {
        const raw = window.localStorage?.getItem(key);
        if (raw === null || raw === undefined || raw === "") {
            return String(fallbackValue || "");
        }
        return String(raw);
    } catch (_err) {
        return String(fallbackValue || "");
    }
}

/**
 * Safe read helper for localStorage boolean values.
 */
function readStoredBool(key, fallbackValue) {
    try {
        const raw = window.localStorage?.getItem(key);
        if (raw === null || raw === undefined) {
            return Boolean(fallbackValue);
        }
        const text = String(raw).trim().toLowerCase();
        return text === "1" || text === "true" || text === "yes";
    } catch (_err) {
        return Boolean(fallbackValue);
    }
}

/**
 * Write string value to localStorage and ignore runtime storage failures.
 */
function writeStoredString(key, value) {
    try {
        window.localStorage?.setItem(key, String(value || ""));
    } catch (_err) {
        // Ignore storage failures and keep runtime-only state.
    }
}

/**
 * Write boolean value to localStorage with compact 1/0 representation.
 */
function writeStoredBool(key, value) {
    try {
        if (value) {
            window.localStorage?.setItem(key, "1");
        } else {
            window.localStorage?.removeItem(key);
        }
    } catch (_err) {
        // Ignore storage failures and keep runtime-only state.
    }
}

/**
 * Create module picker store with subscribe/unsubscribe API.
 */
export function createModuleNodePickerStore(options = {}) {
    const selectedGroupKey = String(options.selectedGroupStorageKey || DEFAULT_SELECTED_GROUP_KEY);
    const selectedModuleKey = String(options.selectedModuleStorageKey || DEFAULT_SELECTED_MODULE_KEY);
    const debugKey = String(options.debugStorageKey || DEFAULT_DEBUG_KEY);

    const state = {
        selectedGroup: readStoredString(selectedGroupKey, options.defaultSelectedGroup || "custom"),
        selectedModule: readStoredString(selectedModuleKey, options.defaultSelectedModule || ""),
        debugEnabled: readStoredBool(debugKey, Boolean(options.defaultDebugEnabled)),
    };

    const listenersByKey = new Map();

    const notify = (keys) => {
        for (const key of keys) {
            const listeners = listenersByKey.get(key);
            if (!listeners || !listeners.size) {
                continue;
            }
            for (const listener of Array.from(listeners)) {
                try {
                    listener(state[key], key, { ...state });
                } catch (err) {
                    console.error("[ALEXZ_tools] picker store listener error:", err);
                }
            }
        }
    };

    const persistKey = (key) => {
        if (key === "selectedGroup") {
            writeStoredString(selectedGroupKey, state.selectedGroup);
            return;
        }
        if (key === "selectedModule") {
            writeStoredString(selectedModuleKey, state.selectedModule);
            return;
        }
        if (key === "debugEnabled") {
            writeStoredBool(debugKey, state.debugEnabled);
        }
    };

    const api = {
        /**
         * Get state value by key.
         */
        get(key) {
            return state[key];
        },

        /**
         * Return shallow snapshot of the whole store state.
         */
        getState() {
            return { ...state };
        },

        /**
         * Set one or many state keys and notify corresponding listeners.
         */
        set(partial) {
            if (!partial || typeof partial !== "object") {
                return [];
            }
            const changed = [];
            for (const [key, value] of Object.entries(partial)) {
                if (!(key in state)) {
                    continue;
                }
                if (state[key] === value) {
                    continue;
                }
                state[key] = value;
                persistKey(key);
                changed.push(key);
            }
            if (changed.length) {
                notify(changed);
            }
            return changed;
        },

        /**
         * Subscribe to updates for one key or list of keys.
         */
        subscribe(keys, listener) {
            if (typeof listener !== "function") {
                return () => {};
            }
            const keyList = Array.isArray(keys) ? keys : [keys];
            for (const key of keyList) {
                if (!(key in state)) {
                    continue;
                }
                if (!listenersByKey.has(key)) {
                    listenersByKey.set(key, new Set());
                }
                listenersByKey.get(key).add(listener);
            }
            return () => {
                for (const key of keyList) {
                    const bucket = listenersByKey.get(key);
                    if (!bucket) {
                        continue;
                    }
                    bucket.delete(listener);
                    if (!bucket.size) {
                        listenersByKey.delete(key);
                    }
                }
            };
        },
    };

    return api;
}
