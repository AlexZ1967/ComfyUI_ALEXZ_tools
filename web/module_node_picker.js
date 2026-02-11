/**
 * Module: web/module_node_picker.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module Node Picker frontend entrypoint.
 *
 * Purpose:
 *   Registers sidebar tab/fallback button and delegates picker composition to
 *   orchestration module.
 */

import { app } from "../../../scripts/app.js";
import {
    EXT_NAME,
    SIDEBAR_TAB_ID,
    MODULE_PICKER_GUARD_KEY,
    FALLBACK_BUTTON_ID,
} from "./constants/module_node_picker_constants.js";
import {
    injectModuleNodePickerStyles,
    cleanupModuleNodePickerFallbackButtons,
    attachModuleNodePickerFallbackButton,
} from "./ui/module_node_picker_shell.js";
import { registerModuleNodePickerExtension } from "./orchestration/core/infra/module_node_picker_registration.js";
import { renderModuleNodePicker } from "./orchestration/core/composition/module_node_picker_composer.js";

registerModuleNodePickerExtension({
    windowObj: window,
    app,
    guardKey: MODULE_PICKER_GUARD_KEY,
    extensionName: EXT_NAME,
    sidebarTabId: SIDEBAR_TAB_ID,
    fallbackButtonId: FALLBACK_BUTTON_ID,
    injectStyles: injectModuleNodePickerStyles,
    cleanupFallbackButtons: cleanupModuleNodePickerFallbackButtons,
    attachFallbackButton: attachModuleNodePickerFallbackButton,
    renderPicker: (container) => renderModuleNodePicker(container, { appInstance: app }),
});
