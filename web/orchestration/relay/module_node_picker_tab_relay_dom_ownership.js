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
            return true;
        }
        if (preferredHost) {
            homeContainer = preferredHost;
            preferredHost.appendChild(root);
            return true;
        }
        return false;
    };

    /**
     * Detach picker root from DOM.
     */
    const ensureDetached = () => {
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
