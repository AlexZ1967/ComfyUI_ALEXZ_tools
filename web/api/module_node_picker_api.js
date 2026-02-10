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
    const resp = await api.fetchApi(`/alexz_tools/node_catalog?cache_only=1&comfyui_mode=${mode}`, {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
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
    const resp = await api.fetchApi(
        `/alexz_tools/module_info?group=${encodeURIComponent(group || "")}` +
        `&module=${encodeURIComponent(moduleName || "")}` +
        `&refresh=${forceRefresh ? "1" : "0"}` +
        `&sync_upstream=${syncUpstream ? "1" : "0"}` +
        `&cache_only=${cacheOnly ? "1" : "0"}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Fetch ComfyUI repository status and update metadata.
 */
export async function fetchComfyUIInfo(forceRefresh = true, acknowledge = true, comfyMode = "releases") {
    const mode = normalizeComfyMode(comfyMode);
    const resp = await api.fetchApi(
        `/alexz_tools/comfyui_info?refresh=${forceRefresh ? "1" : "0"}&acknowledge=${acknowledge ? "1" : "0"}&mode=${mode}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Start backend refresh job that recomputes module/runtime snapshots.
 */
export async function refreshModuleRuntimeState() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Poll refresh job status from backend.
 */
export async function fetchModuleRefreshStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Acknowledge/clear novelty markers for all modules after global refresh action.
 */
export async function acknowledgeAllModuleNovelty() {
    const resp = await api.fetchApi("/alexz_tools/module_acknowledge_all", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Start backend update job for a single module, all modules, or ComfyUI.
 */
export async function startModuleUpdate(scope, moduleName) {
    const resp = await api.fetchApi("/alexz_tools/module_update", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scope: scope || "single",
            module: moduleName || "",
        }),
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Poll module-update job status from backend.
 */
export async function fetchModuleUpdateStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_update_status", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Install requirements.txt for selected custom modules.
 */
export async function installModuleRequirements(modules) {
    const resp = await api.fetchApi("/alexz_tools/module_install_requirements", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: Array.isArray(modules) ? modules : [] }),
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Install ComfyUI requirements.txt in current runtime environment.
 */
export async function installComfyUIRequirements() {
    const resp = await api.fetchApi("/alexz_tools/comfyui_install_requirements", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}
