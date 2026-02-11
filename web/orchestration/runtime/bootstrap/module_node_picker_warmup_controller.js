/**
 * Module: web/orchestration/runtime/bootstrap/module_node_picker_warmup_controller.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Warmup polling controller for Module Node Picker catalog loads.
 *
 * Purpose:
 *   Tracks backend runtime warmup state, updates compact header indicator, and
 *   schedules silent catalog re-polls until warmup is finished.
 */

/**
 * Create warmup controller that manages background catalog warmup polling.
 */
export function createModuleNodePickerWarmupController(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const setWarmupIndicator = typeof context?.setWarmupIndicator === "function"
        ? context.setWarmupIndicator
        : () => {};
    const maxAttempts = Math.max(1, Number(context?.maxAttempts ?? 30));
    const delayMs = Math.max(150, Number(context?.delayMs ?? 1000));

    let warmupPollTimer = 0;
    let warmupPollAttempts = 0;
    let poller = async () => {};

    const clearWarmupPoll = () => {
        if (warmupPollTimer) {
            window.clearTimeout(warmupPollTimer);
            warmupPollTimer = 0;
        }
    };

    const setPoller = (handler) => {
        poller = typeof handler === "function" ? handler : async () => {};
    };

    const onManualLoadStart = () => {
        warmupPollAttempts = 0;
        clearWarmupPoll();
    };

    const scheduleWarmupPoll = (nextOptions = {}) => {
        if (warmupPollAttempts >= maxAttempts) {
            // Do not keep stale warmup indicator visible after retry budget ends.
            clearWarmupPoll();
            warmupPollAttempts = 0;
            setWarmupIndicator(false);
            return;
        }
        clearWarmupPoll();
        warmupPollTimer = window.setTimeout(() => {
            warmupPollTimer = 0;
            if (!shouldContinue()) {
                return;
            }
            void Promise.resolve(poller(nextOptions)).catch(() => {
                // Fail-safe: never leave warmup hint hanging on rejected poll calls.
                clearWarmupPoll();
                warmupPollAttempts = 0;
                setWarmupIndicator(false);
            });
        }, delayMs);
    };

    const handleCatalogResult = (result, nextOptionsFactory) => {
        const warmup = result?.runtimeWarmup || null;
        if (result?.ok && warmup && !Boolean(warmup.done)) {
            setWarmupIndicator(true);
            warmupPollAttempts += 1;
            const nextOptions = typeof nextOptionsFactory === "function"
                ? (nextOptionsFactory() || {})
                : {};
            scheduleWarmupPoll(nextOptions);
            return;
        }
        clearWarmupPoll();
        warmupPollAttempts = 0;
        setWarmupIndicator(false);
    };

    const dispose = () => {
        clearWarmupPoll();
        warmupPollAttempts = 0;
        setWarmupIndicator(false);
    };

    return {
        setPoller,
        onManualLoadStart,
        handleCatalogResult,
        dispose,
    };
}
