/**
 * Module: web/constants/module_node_picker_constants.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shared constants for Module Node Picker frontend.
 *
 * Purpose:
 *   Centralize extension IDs, storage keys, default selections, and UI marks
 *   so picker composition modules can stay focused on orchestration logic.
 */

export const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
export const SIDEBAR_TAB_ID = "alexz-module-nodes";
export const MODULE_PICKER_GUARD_KEY = "__alexz_module_node_picker_registered__";
export const FALLBACK_BUTTON_ID = "alexz-module-nodes-fallback-btn";
export const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
export const NODE_PICKER_DEBUG_KEY = "__alexz_module_picker_debug__";
export const NODE_PICKER_DEBUG_STORAGE_KEY = "alexz_module_picker_debug";
export const NODE_PICKER_SELECTED_GROUP_STORAGE_KEY = "alexz_module_picker_selected_group";
export const NODE_PICKER_SELECTED_MODULE_STORAGE_KEY = "alexz_module_picker_selected_module";
export const COMFYUI_CHECK_MODE_STORAGE_KEY = "alexz_comfyui_check_mode";
export const MODULE_PICKER_RUNTIME_STATE_KEY = "__alexz_module_picker_runtime_state__";
export const LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY = "alexz_module_picker_custom_status_checked";
export const LEGACY_PENDING_CUSTOM_REFRESH_STORAGE_KEY = "alexz_module_picker_pending_custom_refresh";
export const LEGACY_PENDING_UPDATE_STORAGE_KEY = "alexz_module_picker_pending_update";
export const PICKER_CLEANUP_KEY = "__alexz_module_node_picker_cleanup__";

export const GROUP_LABELS = {
    core: "Core_Nodes",
    core_extras: "Core_Extras_Nodes",
    api: "API_Nodes",
    custom: "Custom_Nodes",
};

export const COMFY_GROUP_ORDER = ["core", "core_extras", "api"];
export const MODULE_MARK_UPDATED = "✅";
export const MODULE_MARK_REMOTE_UPDATE = "🟥";
