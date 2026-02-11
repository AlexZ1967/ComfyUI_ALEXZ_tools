/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_dom.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   DOM candidate/context helpers for Module Node Picker tab relay.
 *
 * Purpose:
 *   Centralizes tab-like element detection and sidebar-context checks used by
 *   relay event resolver, avoiding per-event redefinition of helper closures.
 */

const SIDEBAR_CONTEXT_SELECTOR =
    ".side-bar, .sidebar, .comfy-sidebar, [class*='sidebar'], [class*='side-bar']";
const TAB_CANDIDATE_SELECTOR =
    ".side-bar-button, [class*='-tab-button'], [role='tab'], [aria-selected], [aria-controls*='tab']";

/**
 * Return selector used to find tab-like controls.
 */
export function getRelayTabCandidateSelector() {
    return TAB_CANDIDATE_SELECTOR;
}

/**
 * Return true when element belongs to sidebar context.
 */
export function isSidebarContextElement(el) {
    if (!(el instanceof Element)) {
        return false;
    }
    return Boolean(el.closest(SIDEBAR_CONTEXT_SELECTOR));
}

/**
 * Return true when element looks like a tab-like sidebar control.
 */
export function isTabButtonCandidateElement(el, extractTabIdFromButton) {
    if (!(el instanceof Element)) {
        return false;
    }
    if (el.classList?.contains("side-bar-button")) {
        return true;
    }
    if (typeof extractTabIdFromButton === "function" && extractTabIdFromButton(el)) {
        return true;
    }
    const role = String(el.getAttribute("role") || "").toLowerCase();
    if (role === "tab") {
        return true;
    }
    const cls = String(el.className || "").toLowerCase();
    if (cls.includes("tab")) {
        return true;
    }
    if (el.hasAttribute("aria-selected")) {
        return true;
    }
    const controls = String(el.getAttribute("aria-controls") || "").toLowerCase();
    if (controls.includes("tab")) {
        return true;
    }
    return false;
}

