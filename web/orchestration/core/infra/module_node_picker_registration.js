/**
 * Module: web/orchestration/core/infra/module_node_picker_registration.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Extension registration helper for Module Node Picker.
 *
 * Purpose:
 *   Encapsulate guarded sidebar-tab registration and fallback-button wiring
 *   so the main picker module remains a composition entry point.
 */

import { installCustomSidebarHostIsolation } from "./module_node_picker_sidebar_hosts.js";

/**
 * Register Module Node Picker extension once and wire sidebar/fallback mount.
 */
export function registerModuleNodePickerExtension(config = {}) {
    const {
        windowObj,
        app,
        guardKey,
        extensionName,
        sidebarTabId,
        fallbackButtonId,
        injectStyles,
        cleanupFallbackButtons,
        attachFallbackButton,
        renderPicker,
    } = config;

    if (!windowObj || !app || !guardKey || !extensionName || !sidebarTabId || !fallbackButtonId) {
        return false;
    }
    if (windowObj[guardKey]) {
        return false;
    }
    windowObj[guardKey] = true;

    app.registerExtension({
        name: extensionName,
        setup() {
            injectStyles?.();
            cleanupFallbackButtons?.(fallbackButtonId);

            if (app.extensionManager && typeof app.extensionManager.registerSidebarTab === "function") {
                const syncSidebarHosts = installCustomSidebarHostIsolation(app);
                app.extensionManager.registerSidebarTab({
                    id: sidebarTabId,
                    icon: "pi pi-th-large",
                    title: "Module Nodes",
                    tooltip: "Выбор и вставка нод по группам Core/Custom",
                    type: "custom",
                    render: (container) => {
                        renderPicker?.(container);
                    },
                });
                syncSidebarHosts();
                return;
            }

            attachFallbackButton?.({
                app,
                sidebarTabId,
                fallbackButtonId,
            });
        },
    });
    return true;
}
