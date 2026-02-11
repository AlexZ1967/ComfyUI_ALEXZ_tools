/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_events.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   DOM event wiring helpers for Module Node Picker tab relay.
 *
 * Purpose:
 *   Encapsulates event-handler creation and bind/unbind operations so relay
 *   facade can stay focused on state/runtime orchestration.
 */

/**
 * Build relay DOM-event handlers from intent controller.
 */
export function createRelayDomEventHandlers(relayIntent) {
    const supportsPointer = typeof window !== "undefined" && "PointerEvent" in window;
    const onPointerDown = supportsPointer ? ((event) => relayIntent.handleEvent(event)) : null;
    const onMouseDown = supportsPointer ? null : ((event) => relayIntent.handleEvent(event));
    const onKeyUp = (event) => relayIntent.onKeyUp(event);
    const onVisibilityChange = () => relayIntent.onVisibilityChange();
    const onPageShow = () => relayIntent.onPageShow();
    return {
        onPointerDown,
        onMouseDown,
        onKeyUp,
        onVisibilityChange,
        onPageShow,
    };
}

/**
 * Register relay DOM listeners.
 */
export function bindRelayDomEvents(handlers = {}) {
    if (typeof document === "undefined" || typeof window === "undefined") {
        return;
    }
    if (handlers.onPointerDown) {
        document.addEventListener("pointerdown", handlers.onPointerDown, true);
    }
    if (handlers.onMouseDown) {
        document.addEventListener("mousedown", handlers.onMouseDown, true);
    }
    if (handlers.onKeyUp) {
        document.addEventListener("keyup", handlers.onKeyUp, true);
    }
    if (handlers.onVisibilityChange) {
        document.addEventListener("visibilitychange", handlers.onVisibilityChange, true);
    }
    if (handlers.onPageShow) {
        window.addEventListener("pageshow", handlers.onPageShow, true);
    }
}

/**
 * Remove relay DOM listeners using state/handler object shape.
 */
export function unbindRelayDomEvents(state = {}) {
    if (typeof document === "undefined" || typeof window === "undefined") {
        return;
    }
    if (state.onPointerDown) {
        document.removeEventListener("pointerdown", state.onPointerDown, true);
    }
    if (state.onMouseDown) {
        document.removeEventListener("mousedown", state.onMouseDown, true);
    }
    if (state.onKeyUp) {
        document.removeEventListener("keyup", state.onKeyUp, true);
    }
    if (state.onVisibilityChange) {
        document.removeEventListener("visibilitychange", state.onVisibilityChange, true);
    }
    if (state.onPageShow) {
        window.removeEventListener("pageshow", state.onPageShow, true);
    }
}

