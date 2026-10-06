/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_dom_ownership.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   DOM ownership helpers for Module Node Picker tab relay runtime.
 *
 * Purpose:
 *   Encapsulates root attach/detach and mount-host recovery logic so runtime
 *   visibility decisions stay focused on tab intent/state.
 */

/**
 * Create root ownership controller for relay runtime.
 */
export function createRelayDomOwnershipController({ root, mountHost }) {
    const initialHomeContainer = root.parentElement instanceof Element ? root.parentElement : null;
    const explicitMountHost = mountHost instanceof Element ? mountHost : null;
    let homeContainer = initialHomeContainer || explicitMountHost || null;
    let visibilityHost = null;
    const hiddenSiblings = new Map();

    /**
     * Restore the original display rules of other extensions' elements.
     */
    const restoreSiblings = () => {
        for (const [element, display] of hiddenSiblings) {
            if (display.value) {
                element.style.setProperty("display", display.value, display.priority);
            } else {
                element.style.removeProperty("display");
            }
        }
        hiddenSiblings.clear();
        visibilityHost = null;
    };

    /**
     * Hide foreign content temporarily without invalidating its renderer.
     */
    const hideSiblings = (host) => {
        if (visibilityHost !== host) {
            restoreSiblings();
            visibilityHost = host;
        }
        for (const child of Array.from(host.children)) {
            if (child === root || !child.style || hiddenSiblings.has(child)) {
                continue;
            }
            hiddenSiblings.set(child, {
                value: child.style.getPropertyValue("display"),
                priority: child.style.getPropertyPriority("display"),
            });
            // Сохраняем DOM и Vue-состояние соседней панели в общем host.
            child.style.setProperty("display", "none", "important");
        }
    };

    /**
     * Re-attach picker root into active host when needed.
     */
    const ensureAttached = () => {
        const currentParent = root.parentElement instanceof Element ? root.parentElement : null;
        const preferredHost = (explicitMountHost && explicitMountHost.isConnected)
            ? explicitMountHost
            : ((homeContainer && homeContainer.isConnected) ? homeContainer : currentParent);
        if (root.isConnected) {
            if (currentParent instanceof Element) {
                homeContainer = currentParent;
            }
            // Keep root under current sidebar render host when it changes.
            if (preferredHost && currentParent !== preferredHost) {
                preferredHost.appendChild(root);
                homeContainer = preferredHost;
            }
            hideSiblings(homeContainer);
            return true;
        }
        if (preferredHost) {
            homeContainer = preferredHost;
            preferredHost.appendChild(root);
            hideSiblings(preferredHost);
            return true;
        }
        restoreSiblings();
        return false;
    };

    /**
     * Detach picker root from DOM.
     */
    const ensureDetached = () => {
        // Host может быть уже отключён ComfyUI при переходе к Vue-панели.
        // Восстанавливаем стили и в этом случае, а не только для живого root.
        restoreSiblings();
        if (!root.isConnected) {
            return true;
        }
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        return true;
    };

    return {
        ensureAttached,
        ensureDetached,
    };
}
