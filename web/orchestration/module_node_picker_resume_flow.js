/**
 * Module: web/orchestration/module_node_picker_resume_flow.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Public resume-flow facade for Module Node Picker.
 *
 * Purpose:
 *   Preserves stable exports for pending refresh/update restoration while
 *   delegating concrete implementations to focused internal modules.
 */

import { resumePendingCustomRefreshFlowImpl } from "./module_node_picker_resume_custom_refresh.js";
import { resumePendingModuleUpdateFlowImpl } from "./module_node_picker_resume_module_update.js";
import { resumePendingComfyInfoRefreshFlowImpl } from "./module_node_picker_resume_comfy_refresh.js";

/**
 * Resume in-flight Custom Nodes refresh after picker re-open/re-render.
 */
export async function resumePendingCustomRefreshFlow(context) {
    return resumePendingCustomRefreshFlowImpl(context);
}

/**
 * Resume in-flight module-update job after picker re-open/re-render.
 */
export async function resumePendingModuleUpdateFlow(context) {
    return resumePendingModuleUpdateFlowImpl(context);
}

/**
 * Resume interrupted ComfyUI info refresh after picker re-open/re-render.
 */
export async function resumePendingComfyInfoRefreshFlow(context) {
    return resumePendingComfyInfoRefreshFlowImpl(context);
}
