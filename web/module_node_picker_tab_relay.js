/**
 * Module: web/module_node_picker_tab_relay.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module Node Picker tab relay public entrypoint.
 *
 * Purpose:
 *   Keeps stable import path for picker runtime while delegating relay
 *   implementation to orchestration/relay facade module.
 */

export {
    bindModuleNodesTabRelayFacade as bindModuleNodesTabRelay,
    unbindModuleNodesTabRelayFacade as unbindModuleNodesTabRelay,
} from "./orchestration/relay/module_node_picker_tab_relay_facade.js";
