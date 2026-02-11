/**
 * Module: web/orchestration/module_node_picker_startup_flow.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Startup coordinator for Module Node Picker restore/bootstrap phase.
 *
 * Purpose:
 *   Runs pending resume flows and catalog startup bootstrap under one
 *   lifecycle-aware orchestrator, and keeps startup busy-state consistent.
 */

/**
 * Run picker startup coordinator and return cancel callback.
 */
export function runStartupCoordinator(context) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const setStartupBusy = typeof context?.setStartupBusy === "function"
        ? context.setStartupBusy
        : () => {};
    const startCatalogStartupLoad = typeof context?.startCatalogStartupLoad === "function"
        ? context.startCatalogStartupLoad
        : () => () => {};
    const hasPendingCustomRefresh = typeof context?.hasPendingCustomRefresh === "function"
        ? context.hasPendingCustomRefresh
        : () => false;
    const hasPendingUpdate = typeof context?.hasPendingUpdate === "function"
        ? context.hasPendingUpdate
        : () => false;
    const hasPendingComfyInfoRefresh = typeof context?.hasPendingComfyInfoRefresh === "function"
        ? context.hasPendingComfyInfoRefresh
        : () => false;
    const resumePendingCustomRefreshFlow = typeof context?.resumePendingCustomRefreshFlow === "function"
        ? context.resumePendingCustomRefreshFlow
        : async () => {};
    const resumePendingModuleUpdateFlow = typeof context?.resumePendingModuleUpdateFlow === "function"
        ? context.resumePendingModuleUpdateFlow
        : async () => {};
    const resumePendingComfyInfoRefreshFlow = typeof context?.resumePendingComfyInfoRefreshFlow === "function"
        ? context.resumePendingComfyInfoRefreshFlow
        : async () => {};

    let startupCanceled = false;
    let cancelCatalogStartupLoad = () => {};
    let resolveCatalogSettled = () => {};
    const catalogSettledPromise = new Promise((resolve) => {
        resolveCatalogSettled = resolve;
    });

    const shouldContinueStartup = () => !startupCanceled && shouldContinue();

    const runCatalogStartupLoad = () => {
        if (!shouldContinueStartup()) {
            resolveCatalogSettled();
            return;
        }
        cancelCatalogStartupLoad = startCatalogStartupLoad({
            onSettled: () => {
                if (!shouldContinueStartup()) {
                    resolveCatalogSettled();
                    return;
                }
                resolveCatalogSettled();
            },
        }) || (() => {});
    };

    const run = async () => {
        setStartupBusy(true);
        try {
            runCatalogStartupLoad();
            const hasPendingWork = hasPendingCustomRefresh()
                || hasPendingUpdate()
                || hasPendingComfyInfoRefresh();
            if (hasPendingWork) {
                await resumePendingCustomRefreshFlow();
                if (!shouldContinueStartup()) {
                    return;
                }
                await resumePendingModuleUpdateFlow();
                if (!shouldContinueStartup()) {
                    return;
                }
                await resumePendingComfyInfoRefreshFlow();
                if (!shouldContinueStartup()) {
                    return;
                }
            }
            await catalogSettledPromise;
        } finally {
            if (shouldContinueStartup()) {
                setStartupBusy(false);
            }
        }
    };

    void run();
    return () => {
        startupCanceled = true;
        try {
            cancelCatalogStartupLoad?.();
        } catch (_err) {
            // Ignore stale startup-load cleanup errors.
        }
        resolveCatalogSettled();
    };
}

