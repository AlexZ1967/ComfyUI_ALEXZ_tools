/**
 * Module: web/orchestration/flow/module_node_picker_polling_controller.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Polling controller for Module Node Picker refresh/update progress loops.
 *
 * Purpose:
 *   Encapsulates token-based poll lifecycle guards so the main picker module
 *   does not manage refresh/update token state directly.
 */

import {
    pollRefreshProgressLoop,
    pollUpdateProgressLoop,
} from "./module_node_picker_update_flow.js";

/**
 * Create polling controller for refresh and update progress endpoints.
 */
export function createModuleNodePickerPollingController(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const fetchModuleRefreshStatus = context?.fetchModuleRefreshStatus;
    const fetchModuleUpdateStatus = context?.fetchModuleUpdateStatus;
    const formatRefreshLine = context?.formatRefreshLine;
    const formatUpdateLine = context?.formatUpdateLine;
    const setRefreshLine = context?.setRefreshLine;
    const getProcessTarget = typeof context?.getProcessTarget === "function"
        ? context.getProcessTarget
        : () => "custom";
    const customAlert = context?.customAlert;
    const customAlertText = context?.customAlertText;
    const refreshSleepMs = Math.max(100, Number(context?.refreshSleepMs || 400));
    const updateSleepMs = Math.max(100, Number(context?.updateSleepMs || 450));

    let refreshPollToken = 0;
    let updatePollToken = 0;

    const invalidate = () => {
        refreshPollToken += 1;
        updatePollToken += 1;
    };

    const pollRefreshProgress = async () => {
        const token = ++refreshPollToken;
        return pollRefreshProgressLoop({
            shouldContinue,
            isTokenActive: () => token === refreshPollToken,
            fetchModuleRefreshStatus,
            formatRefreshLine,
            setRefreshLine,
            getProcessTarget,
            customAlert,
            customAlertText,
            sleepMs: refreshSleepMs,
        });
    };

    const pollUpdateProgress = async () => {
        const token = ++updatePollToken;
        return pollUpdateProgressLoop({
            shouldContinue,
            isTokenActive: () => token === updatePollToken,
            fetchModuleUpdateStatus,
            formatUpdateLine,
            setRefreshLine,
            sleepMs: updateSleepMs,
        });
    };

    return {
        invalidate,
        pollRefreshProgress,
        pollUpdateProgress,
    };
}
