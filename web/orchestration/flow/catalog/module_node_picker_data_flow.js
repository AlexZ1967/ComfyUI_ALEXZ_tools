/**
 * Module: web/orchestration/flow/catalog/module_node_picker_data_flow.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Data loading flows for Module Node Picker (catalog and module info).
 *
 * Purpose:
 *   Keeps async fetch/render orchestration separate from UI wiring code.
 */

/**
 * Refresh module select option text for one module after badge updates.
 */
export function updateModuleOptionText(context, moduleName) {
    const moduleOptions = context?.moduleOptions;
    const moduleCounts = context?.moduleCounts;
    const moduleBadges = context?.moduleBadges;
    const formatModuleOption = context?.formatModuleOption;
    const marks = context?.marks || {};
    const option = moduleOptions?.get?.(moduleName);
    if (!option) {
        return;
    }
    const count = moduleCounts?.get?.(moduleName) || 0;
    const badges = moduleBadges?.get?.(moduleName) || null;
    option.textContent = typeof formatModuleOption === "function"
        ? formatModuleOption(moduleName, count, badges, marks)
        : `${moduleName} (${count})`;
}

/**
 * Cache node-level diff markers (new/updated) for selected module.
 */
export function cacheModuleNodeDiffs(context, moduleName, info) {
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    if (!moduleNodeDiffs?.set) {
        return;
    }
    const newNodes = Array.isArray(info?.new_nodes_between_runs) ? info.new_nodes_between_runs : [];
    const updatedNodes = Array.isArray(info?.updated_nodes_between_runs) ? info.updated_nodes_between_runs : [];
    moduleNodeDiffs.set(moduleName, {
        newNodes: new Set(newNodes),
        updatedNodes: new Set(updatedNodes),
        markAllUpdated: Boolean(info?.new_module_between_runs),
    });
}

/**
 * Load and render module info for currently selected group/module.
 */
export async function loadModuleInfoFlow(options, context) {
    const forceRefresh = Boolean(options?.forceRefresh);
    const syncUpstream = Boolean(options?.syncUpstream);
    const throwOnError = Boolean(options?.throwOnError);
    const isRequestActive = context?.isRequestActive;
    const selectedModule = String(context?.getSelectedModule?.() || "");
    const selectedGroup = String(context?.getSelectedGroup?.() || "");
    if (!selectedModule || selectedModule === "-1") {
        context?.clearModuleInfo?.();
        return;
    }
    try {
        const payload = await context?.fetchModuleInfo?.(selectedGroup, selectedModule, {
            forceRefresh,
            syncUpstream,
        });
        if (typeof isRequestActive === "function" && !isRequestActive()) {
            return;
        }
        if (
            String(context?.getSelectedModule?.() || "") !== selectedModule
            || String(context?.getSelectedGroup?.() || "") !== selectedGroup
        ) {
            return;
        }
        const info = payload?.info || null;
        context?.renderModuleInfo?.(info);
        if (info) {
            const badges = context?.moduleBadgesFromInfo?.(info) || null;
            if (badges?.updatedBetweenRuns || badges?.hasRemoteUpdate) {
                context?.moduleBadges?.set?.(selectedModule, badges);
            } else {
                context?.moduleBadges?.delete?.(selectedModule);
            }
            context?.setModuleNodeDiffs?.(selectedModule, info);
            context?.setModuleOptionText?.(selectedModule);
            context?.renderNodeList?.();
        }
    } catch (err) {
        if (typeof isRequestActive === "function" && !isRequestActive()) {
            return;
        }
        context?.clearModuleInfo?.();
        if (throwOnError) {
            throw err;
        }
    }
}

/**
 * Load full node catalog from backend and refresh picker UI state.
 */
export async function loadCatalogFlow(options, context) {
    const preferredGroup = String(options?.preferredGroup || "").trim();
    const preferredModule = String(options?.preferredModule || "").trim();
    const autoExpandModule = String(options?.autoExpandModule || "").trim();
    const isRequestActive = context?.isRequestActive;
    try {
        const payload = await context?.fetchNodeCatalog?.(context?.getComfyMode?.());
        if (typeof isRequestActive === "function" && !isRequestActive()) {
            return;
        }
        context?.catalogByGroup?.clear?.();
        const groups = payload?.groups || [];
        const totalNodes = groups.reduce((acc, group) => acc + Number(group?.count || 0), 0);
        const totalModules = groups.reduce(
            (acc, group) => acc + Number(
                group?.module_count
                ?? (Array.isArray(group?.modules) ? group.modules.length : 0)
            ),
            0
        );
        context?.setCustomModulesNeedUpdate?.(Number(payload?.custom_modules_need_update || 0));
        context?.fillGroupSelect?.(groups, { preferredGroup, preferredModule, autoExpandModule });
        context?.syncUpdateAllButton?.();
        const runtimeWarmup = payload?.runtime_warmup || null;
        return {
            ok: true,
            groupCount: groups.length,
            totalNodes,
            totalModules,
            runtimeWarmup,
        };
    } catch (err) {
        if (typeof isRequestActive === "function" && !isRequestActive()) {
            return;
        }
        context?.setHelpText?.(`Ошибка загрузки: ${String(err)}`);
        const comfyAlert = context?.comfyAlert;
        const comfyUpdateBtn = context?.comfyUpdateBtn;
        const comfyInstallReqBtn = context?.comfyInstallReqBtn;
        if (comfyAlert) {
            comfyAlert.style.display = "none";
        }
        if (comfyUpdateBtn) {
            comfyUpdateBtn.style.display = "none";
        }
        if (comfyInstallReqBtn) {
            comfyInstallReqBtn.style.display = "none";
        }
        context?.setCustomModulesNeedUpdate?.(0);
        if (context?.groupSelect) {
            context.groupSelect.innerHTML = "";
        }
        if (context?.nodeSelect) {
            context.nodeSelect.innerHTML = "";
        }
        context?.clearModuleInfo?.();
        if (context?.nodeList) {
            context.nodeList.innerHTML = "";
        }
        context?.syncUpdateAllButton?.();
        return {
            ok: false,
            error: String(err),
            groupCount: 0,
            totalNodes: 0,
            totalModules: 0,
            runtimeWarmup: null,
        };
    }
}
