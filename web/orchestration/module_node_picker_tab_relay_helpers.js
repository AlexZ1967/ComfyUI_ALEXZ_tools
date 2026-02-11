/**
 * Module: web/orchestration/module_node_picker_tab_relay_helpers.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shared helper utilities for Module Node Picker tab relay.
 *
 * Purpose:
 *   Keeps tab-id inference, sidebar API lookup, and diagnostic DOM snapshot
 *   helpers separated from the relay state machine.
 */

/**
 * Return Sidebar API object for current ComfyUI build shape.
 */
export function getSidebarApi(app) {
    const manager = app?.extensionManager;
    return manager?.sidebarTab || manager || null;
}

/**
 * Read currently active sidebar tab id in a version-tolerant way.
 */
export function getActiveSidebarTabId(app) {
    const sidebar = getSidebarApi(app);
    if (!sidebar) {
        return "";
    }
    const active = sidebar.activeSidebarTabId ?? sidebar.activeSidebarTab ?? "";
    return String(active || "");
}

/**
 * Extract logical tab id from sidebar button attributes/classes.
 */
export function extractTabIdFromButton(buttonEl) {
    if (!(buttonEl instanceof Element)) {
        return "";
    }
    const idAttr = String(buttonEl.getAttribute("id") || "").trim();
    if (idAttr.endsWith("-tab-button")) {
        return idAttr.slice(0, -"-tab-button".length);
    }
    const dataTabId = String(
        buttonEl.getAttribute("data-tab-id")
        || buttonEl.getAttribute("data-sidebar-tab")
        || buttonEl.getAttribute("data-tab")
        || ""
    ).trim();
    if (dataTabId) {
        return dataTabId;
    }
    for (const cls of Array.from(buttonEl.classList || [])) {
        if (cls.endsWith("-tab-button")) {
            return cls.slice(0, -"-tab-button".length);
        }
    }
    return "";
}

/**
 * Resolve sidebar tab-like button element from DOM event target/path.
 */
export function resolveSidebarButtonFromEvent(event) {
    const isSidebarContextElement = (el) => {
        if (!(el instanceof Element)) {
            return false;
        }
        if (el.classList?.contains("side-bar-button")) {
            return true;
        }
        if (extractTabIdFromButton(el)) {
            return true;
        }
        return Boolean(
            el.closest(
                ".side-bar, .sidebar, .comfy-sidebar, [class*='sidebar'], [class*='side-bar']"
            )
        );
    };
    const direct = event?.target;
    if (direct instanceof Element) {
        const closest = direct.closest(
            ".side-bar-button, [class*='-tab-button'], [role='tab'], button[aria-label], button[title]"
        );
        if (closest && isSidebarContextElement(closest)) {
            return closest;
        }
    }
    if (typeof event?.composedPath === "function") {
        for (const item of event.composedPath()) {
            if (!(item instanceof Element)) {
                continue;
            }
            const role = String(item.getAttribute("role") || "").toLowerCase();
            if (role === "tab" && isSidebarContextElement(item)) {
                return item;
            }
            if (isSidebarContextElement(item)) {
                return item;
            }
        }
    }
    return null;
}

/**
 * Return selected state for this extension sidebar button.
 */
export function isOwnButtonSelected(sidebarTabId) {
    const ownBtn = document.querySelector(`.${sidebarTabId}-tab-button`);
    if (!ownBtn) {
        return null;
    }
    return ownBtn.classList.contains("side-bar-button-selected");
}

/**
 * Collect known sidebar tab descriptors from multiple ComfyUI API shapes.
 */
export function collectSidebarTabDescriptors(app) {
    const sidebar = getSidebarApi(app);
    const pools = [
        sidebar?.tabs,
        sidebar?.sidebarTabs,
        sidebar?.tabConfigs,
        app?.extensionManager?.tabs,
        app?.extensionManager?.sidebarTabs,
    ];
    const out = [];
    for (const pool of pools) {
        if (!Array.isArray(pool)) {
            continue;
        }
        for (const item of pool) {
            if (!item || typeof item !== "object") {
                continue;
            }
            const id = String(item.id || item.tabId || item.key || "").trim();
            const title = String(item.title || item.label || item.tooltip || "").trim();
            if (id) {
                out.push({ id, title });
            }
        }
    }
    return out;
}

/**
 * Check whether sidebar exposes tab with the requested id.
 */
export function hasSidebarTabId(app, tabId) {
    if (!tabId) {
        return false;
    }
    const descriptors = collectSidebarTabDescriptors(app);
    return descriptors.some((x) => String(x.id || "") === String(tabId));
}

/**
 * Infer tab id when clicked button has no stable id/class markers.
 */
export function inferFallbackTabIdFromContext(app, event, sidebarTabId) {
    const knownNodesMapId = "easyuse_nodes_map";
    const descriptors = collectSidebarTabDescriptors(app);
    const composed = typeof event?.composedPath === "function" ? event.composedPath() : [];
    const chunks = [];
    for (const item of composed) {
        if (!(item instanceof Element)) {
            continue;
        }
        const cls = String(item.className || "").toLowerCase();
        const text = String(item.textContent || "").toLowerCase().trim();
        const title = String(item.getAttribute("title") || item.getAttribute("aria-label") || "").toLowerCase().trim();
        if (cls) {
            chunks.push(cls);
        }
        if (text) {
            chunks.push(text);
        }
        if (title) {
            chunks.push(title);
        }
    }
    const hay = chunks.join(" | ");
    if (hay.includes("nodesmap") || hay.includes("nodes map")) {
        if (hasSidebarTabId(app, knownNodesMapId)) {
            return knownNodesMapId;
        }
    }
    if (hay.includes("pi-sitemap") && hasSidebarTabId(app, knownNodesMapId)) {
        return knownNodesMapId;
    }
    // Generic fallback by title match from descriptors.
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (!id || id === sidebarTabId) {
            continue;
        }
        if (title.includes("nodesmap") || title.includes("nodes map")) {
            return id;
        }
    }
    return "";
}

/**
 * Infer target tab id from button metadata, title, and known variants.
 */
export function inferTabIdFromButton(app, button) {
    const explicit = extractTabIdFromButton(button);
    if (explicit) {
        return explicit;
    }
    const label = String(
        button.getAttribute("title")
        || button.getAttribute("aria-label")
        || button.textContent
        || ""
    ).trim().toLowerCase();
    if (!label) {
        return "";
    }
    const descriptors = collectSidebarTabDescriptors(app);
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (title && (title === label || title.includes(label) || label.includes(title))) {
            return id;
        }
    }
    if (label.includes("nodesmap") || label.includes("nodes map")) {
        // Known EasyUse tab id variants seen in different builds.
        return "easyuse_nodes_map";
    }
    return "";
}

/**
 * Return compact DOM snapshot of picker container children for diagnostics.
 */
export function getContainerState(root) {
    const container = root?.parentElement;
    if (!(container instanceof Element)) {
        return { childCount: 0, childShort: "n/a" };
    }
    const out = [];
    let childCount = 0;
    for (const node of container.childNodes) {
        childCount += 1;
        if (node === root) {
            out.push("ROOT");
            continue;
        }
        if (node.nodeType === Node.TEXT_NODE) {
            const txt = String(node.textContent || "").trim();
            out.push(txt ? `TXT:${txt.slice(0, 20)}` : "TXT:blank");
            continue;
        }
        if (node instanceof Element) {
            const cls = String(node.className || "").trim();
            out.push(cls ? `${node.nodeName}.${cls.split(/\s+/).slice(0, 2).join(".")}` : node.nodeName);
        } else {
            out.push(String(node.nodeName || "NODE"));
        }
    }
    return { childCount, childShort: out.slice(0, 10).join(" | ") || "n/a" };
}
