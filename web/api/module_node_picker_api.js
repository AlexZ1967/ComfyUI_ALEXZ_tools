/**
 * Module: web/api/module_node_picker_api.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Backend API wrappers for Module Node Picker.
 *
 * Purpose:
 *   Centralizes HTTP calls to picker backend routes so UI code can focus on
 *   rendering and interaction logic.
 */

import { api } from "../../../../scripts/api.js";

const DEFAULT_API_TIMEOUT_MS = 30000;

/**
 * Fetch JSON from backend API with timeout and unified error shape.
 */
async function fetchApiJson(path, options = {}, timeoutMs = DEFAULT_API_TIMEOUT_MS) {
    const timeout = Math.max(1000, Number(timeoutMs || DEFAULT_API_TIMEOUT_MS));
    const externalSignal = options?.signal;
    const requestOptions = { ...(options || {}) };
    delete requestOptions.signal;
    const controller = new AbortController();
    let timedOut = false;
    const onExternalAbort = () => {
        controller.abort();
    };
    if (externalSignal) {
        if (externalSignal.aborted) {
            controller.abort();
        } else {
            externalSignal.addEventListener("abort", onExternalAbort, { once: true });
        }
    }
    const guardedTimer = window.setTimeout(() => {
        timedOut = true;
        controller.abort();
    }, timeout);
    try {
        const resp = await api.fetchApi(path, {
            ...requestOptions,
            signal: controller.signal,
        });
        if (!resp.ok) {
            throw new Error(`API ${resp.status}`);
        }
        return await resp.json();
    } catch (err) {
        if (err?.name === "AbortError") {
            if (timedOut) {
                throw new Error(`API timeout after ${timeout}ms`);
            }
            if (externalSignal?.aborted) {
                throw new Error("API request canceled");
            }
            throw new Error("API request aborted");
        }
        throw err;
    } finally {
        window.clearTimeout(guardedTimer);
        if (externalSignal) {
            externalSignal.removeEventListener("abort", onExternalAbort);
        }
    }
}

/**
 * Normalize ComfyUI check mode to supported values.
 */
function normalizeComfyMode(comfyMode) {
    return String(comfyMode || "releases").trim().toLowerCase() === "commits"
        ? "commits"
        : "releases";
}

/**
 * Fetch grouped node catalog data from backend API.
 */
export async function fetchNodeCatalog(comfyMode = "releases", options = {}) {
    const mode = normalizeComfyMode(comfyMode);
    return fetchApiJson(`/alexz_tools/node_catalog?cache_only=1&comfyui_mode=${mode}`, {
        cache: "no-store",
        signal: options?.signal,
    });
}

/**
 * Fetch detailed info for a specific module.
 */
export async function fetchModuleInfo(group, moduleName, options = {}) {
    const forceRefresh = Boolean(options?.forceRefresh);
    const syncUpstream = Boolean(options?.syncUpstream);
    const cacheOnly = options?.cacheOnly === undefined
        ? (!forceRefresh && !syncUpstream)
        : Boolean(options?.cacheOnly);
    return fetchApiJson(
        `/alexz_tools/module_info?group=${encodeURIComponent(group || "")}` +
        `&module=${encodeURIComponent(moduleName || "")}` +
        `&refresh=${forceRefresh ? "1" : "0"}` +
        `&sync_upstream=${syncUpstream ? "1" : "0"}` +
        `&cache_only=${cacheOnly ? "1" : "0"}`,
        {
            cache: "no-store",
            signal: options?.signal,
        }
    );
}

/**
 * Fetch ComfyUI repository status and update metadata.
 */
export async function fetchComfyUIInfo(
    forceRefresh = true,
    acknowledge = true,
    comfyMode = "releases",
    options = {}
) {
    const mode = normalizeComfyMode(comfyMode);
    return fetchApiJson(
        `/alexz_tools/comfyui_info?refresh=${forceRefresh ? "1" : "0"}&acknowledge=${acknowledge ? "1" : "0"}&mode=${mode}`,
        {
            cache: "no-store",
            signal: options?.signal,
        }
    );
}

/**
 * Start backend refresh job that recomputes module/runtime snapshots.
 */
export async function refreshModuleRuntimeState(options = {}) {
    return fetchApiJson("/alexz_tools/module_refresh", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            log_mode: String(options?.logMode || "summary"),
        }),
        signal: options?.signal,
    });
}

/**
 * Poll refresh job status from backend.
 */
export async function fetchModuleRefreshStatus(options = {}) {
    return fetchApiJson("/alexz_tools/module_refresh_status", {
        cache: "no-store",
        signal: options?.signal,
    }, 15000);
}

/**
 * Acknowledge/clear novelty markers for all modules after global refresh action.
 */
export async function acknowledgeAllModuleNovelty(options = {}) {
    return fetchApiJson("/alexz_tools/module_acknowledge_all", {
        method: "POST",
        cache: "no-store",
        signal: options?.signal,
    });
}

/**
 * Start backend update job for a single module, all modules, or ComfyUI.
 */
export async function startModuleUpdate(scope, moduleName, options = {}) {
    return fetchApiJson("/alexz_tools/module_update", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scope: scope || "single",
            module: moduleName || "",
            log_mode: String(options?.logMode || "summary"),
        }),
        signal: options?.signal,
    }, 60000);
}

/**
 * Poll module-update job status from backend.
 */
export async function fetchModuleUpdateStatus(options = {}) {
    return fetchApiJson("/alexz_tools/module_update_status", {
        cache: "no-store",
        signal: options?.signal,
    }, 15000);
}

/**
 * Install requirements.txt for selected custom modules.
 */
export async function installModuleRequirements(modules, options = {}) {
    return fetchApiJson("/alexz_tools/module_install_requirements", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: Array.isArray(modules) ? modules : [] }),
        signal: options?.signal,
    }, 120000);
}

/**
 * Install ComfyUI requirements.txt in current runtime environment.
 */
export async function installComfyUIRequirements(options = {}) {
    return fetchApiJson("/alexz_tools/comfyui_install_requirements", {
        method: "POST",
        cache: "no-store",
        signal: options?.signal,
    }, 120000);
}
