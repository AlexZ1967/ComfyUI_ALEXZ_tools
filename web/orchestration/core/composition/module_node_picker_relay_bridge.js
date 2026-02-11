/**
 * Module: web/orchestration/core/composition/module_node_picker_relay_bridge.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Relay binding bridge for Module Node Picker composition flow.
 *
 * Purpose:
 *   Encapsulates relay bind wiring from composer to keep composition body
 *   smaller while preserving the same runtime behavior and diagnostics flow.
 */

import { SIDEBAR_TAB_ID } from "../../../constants/module_node_picker_constants.js";
import { bindModuleNodesTabRelay } from "../../../module_node_picker_tab_relay.js";

/**
 * Bind Module Node Picker relay using composer/runtime dependencies.
 */
export function bindModuleNodePickerRelayBridge({
    appInstance,
    root,
    container,
    pickerStore,
    debugUi,
}) {
    bindModuleNodesTabRelay({
        app: appInstance,
        root,
        mountHost: container,
        sidebarTabId: SIDEBAR_TAB_ID,
        onDiag: (diag) => {
            if (pickerStore?.get?.("debugEnabled") !== true) {
                return;
            }
            debugUi?.setDiagnosticText?.(diag);
        },
    });
}
