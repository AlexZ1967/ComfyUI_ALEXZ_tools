/**
 * Module: web/api/module_node_picker_api.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
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
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
    try {
        const resp = await api.fetchApi(path, {
            ...options,
            signal: controller.signal,
        });
        if (!resp.ok) {
            throw new Error(`API ${resp.status}`);
        }
        return await resp.json();
    } catch (err) {
        if (err?.name === "AbortError") {
            throw new Error(`API timeout after ${timeout}ms`);
        }
        throw err;
    } finally {
        window.clearTimeout(timer);
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
export async function fetchNodeCatalog(comfyMode = "releases") {
    const mode = normalizeComfyMode(comfyMode);
    return fetchApiJson(`/alexz_tools/node_catalog?cache_only=1&comfyui_mode=${mode}`, {
        cache: "no-store",
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
        { cache: "no-store" }
    );
}

/**
 * Fetch ComfyUI repository status and update metadata.
 */
export async function fetchComfyUIInfo(forceRefresh = true, acknowledge = true, comfyMode = "releases") {
    const mode = normalizeComfyMode(comfyMode);
    return fetchApiJson(
        `/alexz_tools/comfyui_info?refresh=${forceRefresh ? "1" : "0"}&acknowledge=${acknowledge ? "1" : "0"}&mode=${mode}`,
        { cache: "no-store" }
    );
}

/**
 * Start backend refresh job that recomputes module/runtime snapshots.
 */
export async function refreshModuleRuntimeState() {
    return fetchApiJson("/alexz_tools/module_refresh", {
        method: "POST",
        cache: "no-store",
    });
}

/**
 * Poll refresh job status from backend.
 */
export async function fetchModuleRefreshStatus() {
    return fetchApiJson("/alexz_tools/module_refresh_status", {
        cache: "no-store",
    }, 15000);
}

/**
 * Acknowledge/clear novelty markers for all modules after global refresh action.
 */
export async function acknowledgeAllModuleNovelty() {
    return fetchApiJson("/alexz_tools/module_acknowledge_all", {
        method: "POST",
        cache: "no-store",
    });
}

/**
 * Start backend update job for a single module, all modules, or ComfyUI.
 */
export async function startModuleUpdate(scope, moduleName) {
    return fetchApiJson("/alexz_tools/module_update", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scope: scope || "single",
            module: moduleName || "",
        }),
    }, 60000);
}

/**
 * Poll module-update job status from backend.
 */
export async function fetchModuleUpdateStatus() {
    return fetchApiJson("/alexz_tools/module_update_status", {
        cache: "no-store",
    }, 15000);
}

/**
 * Install requirements.txt for selected custom modules.
 */
export async function installModuleRequirements(modules) {
    return fetchApiJson("/alexz_tools/module_install_requirements", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: Array.isArray(modules) ? modules : [] }),
    }, 120000);
}

/**
 * Install ComfyUI requirements.txt in current runtime environment.
 */
export async function installComfyUIRequirements() {
    return fetchApiJson("/alexz_tools/comfyui_install_requirements", {
        method: "POST",
        cache: "no-store",
    }, 120000);
}
