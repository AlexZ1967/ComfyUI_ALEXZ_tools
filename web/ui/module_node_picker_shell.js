/**
 * Module: web/ui/module_node_picker_shell.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shell helpers for Module Node Picker style injection and fallback mount.
 *
 * Purpose:
 *   Keeps static CSS and sidebar-fallback button DOM utilities out of the
 *   picker orchestration module to reduce file size and coupling.
 */

import { getModuleNodePickerStyleText } from "./styles/module_node_picker_styles.js";

const STYLE_ID = "alexz-module-picker-style";

/**
 * Inject Module Node Picker stylesheet once.
 */
export function injectModuleNodePickerStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = getModuleNodePickerStyleText();
    document.head.appendChild(style);
}

/**
 * Remove all fallback buttons created by this extension.
 */
export function cleanupModuleNodePickerFallbackButtons(fallbackButtonId) {
    const byId = document.getElementById(String(fallbackButtonId || ""));
    if (byId && byId.parentNode) {
        byId.parentNode.removeChild(byId);
    }
    for (const el of document.querySelectorAll(".alexz-mod-picker-floating-btn")) {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    }
}

/**
 * Attach fallback button when Sidebar API is unavailable.
 */
export function attachModuleNodePickerFallbackButton(context = {}) {
    const app = context?.app;
    const sidebarTabId = String(context?.sidebarTabId || "");
    const fallbackButtonId = String(context?.fallbackButtonId || "");
    cleanupModuleNodePickerFallbackButtons(fallbackButtonId);

    const button = document.createElement("button");
    button.id = fallbackButtonId;
    button.type = "button";
    button.textContent = "Module Nodes";
    button.title = "Открыть подбор нод";
    button.onclick = () => {
        const manager = app?.extensionManager;
        const sidebar = manager?.sidebarTab || manager;
        const openFn = sidebar && typeof sidebar.activateSidebarTab === "function"
            ? sidebar.activateSidebarTab.bind(sidebar)
            : null;
        if (!openFn) {
            button.textContent = "Sidebar API недоступен";
            return;
        }
        try {
            openFn(sidebarTabId);
        } catch (_err) {
            button.textContent = "Sidebar API недоступен";
        }
    };

    const menuContainer = app?.ui?.menuContainer;
    if (menuContainer) {
        button.style.width = "100%";
        button.style.order = 95;
        menuContainer.append(button);
        return;
    }

    button.className = "alexz-mod-picker-floating-btn";
    document.body.appendChild(button);
}
