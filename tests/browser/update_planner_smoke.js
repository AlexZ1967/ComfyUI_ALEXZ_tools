/** Check compact batch updates and widget locking against mocked APIs. */
async (page) => {
    const report = { status: "FAIL", checks: [], failures: [], mockedRequests: [] };
    const require = (condition, message) => { if (!condition) throw new Error(message); };
    const plan = {
        id: "browser-fixture", expires_at: Date.now() / 1000 + 3600,
        counts: { safe: 2, risk: 1, unknown: 0, blocked: 0, up_to_date: 1 },
        batch_count: 2, batch_error: "", batch: { additions: [] }, baseline: ["Private diagnostic"],
        results: [
            { module: "ComfyUI_ALEXZ_tools", status: "safe", reasons: [], requirements_changed: false,
                additions: [], diff: "", before: "aaa", target: "bbb" },
            { module: "comfyui-manager", status: "safe", reasons: [], requirements_changed: false,
                additions: [], diff: "", before: "aaa", target: "bbb" },
            { module: "Fixture_Risk", status: "risk", reasons: ["Private risk"], requirements_changed: true,
                additions: [], diff: "-torch>=2\n+torch>=99", before: "aaa", target: "bbb" },
            { module: "AUN-ComfyUI-Nodes", status: "up_to_date", reasons: [], requirements_changed: null,
                additions: [], diff: "" },
        ],
    };
    let checked = false;
    let checkingPolls = 0;
    let execution = null;
    const restartRequired = [];
    const errors = [];
    const onError = error => { if (/ALEXZ_tools|module_updates|module_node_picker/i.test(error.stack || error.message)) errors.push(error.message); };
    const pattern = "**/alexz_tools/**";
    const guard = async route => {
        const request = route.request();
        const path = new URL(request.url()).pathname;
        const fulfill = payload => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(payload) });
        if (path.endsWith("/module_acknowledge_all")) return fulfill({ status: "ok" });
        if (/\/update_plan_(status|check|apply|cancel)$/.test(path)) {
            report.mockedRequests.push({ path, method: request.method() });
            if (path.endsWith("_check")) {
                checked = true; checkingPolls = 2;
                return fulfill({ status: "started" });
            }
            if (path.endsWith("_apply")) {
                const input = request.postDataJSON();
                require(input.plan_id === plan.id && input.confirmed === true && input.mode === "checked"
                    && (!input.module || ["ComfyUI_ALEXZ_tools", "comfyui-manager"].includes(input.module)), "Invalid checked update contract");
                report.mockedRequests.at(-1).module = input.module;
                execution = { phase: "updating", message: "Private git output", directory: "fixture-only" };
                return fulfill({ status: "queued", directory: "fixture-only" });
            }
            if (path.endsWith("_cancel")) {
                execution = { ...execution, phase: "cancelled", message: "Private cancelled" };
                return fulfill({ status: "cancelling" });
            }
            if (checkingPolls > 0) { checkingPolls -= 1; return fulfill({ status: "ok", phase: "checking", message: "Fixture checking", plan: null }); }
            return fulfill({ status: "ok", phase: checked ? "ready" : "idle", message: "Private status", plan: checked ? plan : null, execution,
                updated_modules: restartRequired, restart_required_modules: restartRequired });
        }
        if (request.method() === "GET" && /\/(node_catalog|module_info|module_refresh_status|module_update_status|comfyui_info)$/.test(path)) return route.continue();
        report.failures.push(`Blocked real operation: ${request.method()} ${path}`);
        return route.abort("blockedbyclient");
    };
    const check = async (name, action) => { await action(); report.checks.push({ name, status: "PASS" }); };
    await page.route(pattern, guard);
    page.on("pageerror", onError);
    await page.evaluate(() => {
        window.__alexzSmokeConfirm = { original: window.confirm, messages: [], accept: false };
        window.confirm = message => { window.__alexzSmokeConfirm.messages.push(message); return window.__alexzSmokeConfirm.accept; };
    });
    const picker = page.locator(".alexz-mod-picker");
    const refresh = picker.getByRole("button", { name: "Refresh Custom Nodes Info", exact: true });
    const update = picker.getByRole("button", { name: /^Обновить модули/, includeHidden: true });
    const remount = async () => {
        await page.getByTestId("apps-tab-button").click();
        await page.getByTestId("alexz-module-nodes-tab-button").click();
        await picker.waitFor({ state: "visible" });
        await page.waitForTimeout(300);
    };
    try {
        await remount();
        await check("One refresh and one compact update action", async () => {
            require(await update.count() === 1 && await update.isHidden(), "Initial update action invalid");
            await refresh.click();
            require(await update.isHidden(), "Update shown during check");
            require(await picker.locator(".alexz-update-summary:visible").count() === 1, "Duplicate progress plaque");
            await page.waitForFunction(() => document.querySelector(".alexz-update-planner-summary")?.textContent === "Найдено обновлений: 3 модулей.");
            require(await update.isEnabled(), "Checked batch unavailable");
        });
        await check("No per-module update details or diagnostic panels", async () => {
            const selects = picker.locator(".alexz-mod-picker-selection-block select");
            await selects.nth(0).selectOption("custom");
            for (const module of ["ComfyUI_ALEXZ_tools", "AUN-ComfyUI-Nodes"]) {
                await selects.nth(2).selectOption(module);
                require(await picker.locator(".alexz-module-update").count() === 0, "New module information remains");
                const individual = picker.getByRole("button", { name: "Обновить модуль", exact: true });
                if (module === "ComfyUI_ALEXZ_tools") {
                    await individual.waitFor({ state: "visible" });
                    require(await individual.count() === 1 && await individual.isEnabled(), "Individual update action unavailable");
                } else {
                    await individual.waitFor({ state: "detached" });
                    require(await individual.count() === 0, "Current module shows update action");
                }
            }
            require(await picker.locator(".alexz-update-planner-diagnostics, .alexz-update-disabled-reason").count() === 0, "Diagnostic clutter remains");
            require(!(await picker.innerText()).includes("Private diagnostic"), "Backend diagnostics leaked into widget");
        });
        await check("Dismissed batch confirmation does not update", async () => {
            await update.click();
            require(!report.mockedRequests.some(item => item.path.endsWith("_apply")), "Dismissed batch executed");
        });
        await check("Entire picker freezes during update", async () => {
            await page.evaluate(() => { window.__alexzSmokeConfirm.accept = true; });
            await update.click();
            require(await picker.evaluate(root => root.inert && [...root.querySelectorAll("button, input, select, textarea")].every(e => e.disabled)), "Picker controls are not all frozen");
            require(!(await picker.innerText()).includes("fixture-only"), "Local diagnostic path shown");
            execution = { ...execution, phase: "done", message: "Private done" };
            await page.waitForFunction(() => !document.querySelector(".alexz-mod-picker")?.inert);
            require(await picker.evaluate(root => !root.inert), "Picker still frozen after completion");
            require(await refresh.isEnabled(), "Refresh not restored");
        });
        await check("Errors stay compact and unlock the picker", async () => {
            execution = { ...execution, phase: "error", message: "Private traceback" };
            await remount();
            const text = await picker.innerText();
            require(text.includes("Ошибка обновления. Подробности в консоли ComfyUI."), "Missing compact failure");
            require(!text.includes("Private traceback") && !text.includes("fixture-only"), "Error details leaked into widget");
            require(await picker.evaluate(root => !root.inert), "Picker frozen after error");
        });
        await check("Individual update selects one module and locks the widget", async () => {
            execution = null;
            await remount();
            const selects = picker.locator(".alexz-mod-picker-selection-block select");
            await selects.nth(0).selectOption("custom");
            await selects.nth(2).selectOption("ComfyUI_ALEXZ_tools");
            await picker.getByRole("button", { name: "Обновить модуль", exact: true }).click();
            await page.waitForFunction(() => document.querySelector(".alexz-update-planner-summary")?.textContent.includes("Ход работы"));
            require(report.mockedRequests.some(item => item.path.endsWith("_apply") && item.module === "ComfyUI_ALEXZ_tools"), "Wrong individual update selection");
            require(await picker.evaluate(root => root.inert), "Individual update did not lock picker");
            execution = { ...execution, phase: "done", message: "Private done" };
            const item = plan.results.find(item => item.module === "ComfyUI_ALEXZ_tools");
            item.status = "up_to_date"; item.before = item.target;
            plan.id = "remaining-first"; plan.batch_count = 1;
            restartRequired.push("ComfyUI_ALEXZ_tools");
            await page.waitForFunction(() => !document.querySelector(".alexz-mod-picker")?.inert);
            require(await update.isEnabled() && (await update.textContent()).includes("(1)"), "Remaining batch was lost");
            const status = picker.locator(".alexz-module-runtime-status");
            require((await status.innerText()).includes("Требуется перезагрузка"), "Missing module restart status");
            require(await status.evaluate(node => getComputedStyle(node).color === "rgb(255, 107, 107)"), "Restart status is not red");
            require(await picker.locator(".alexz-module-update-button").count() === 0, "Completed module still has update action");
        });
        await check("Successful update marker survives panel remount and selection", async () => {
            await remount();
            const selects = picker.locator(".alexz-mod-picker-selection-block select");
            await selects.nth(0).selectOption("custom");
            const option = selects.nth(2).locator("option[value='ComfyUI_ALEXZ_tools']");
            require((await option.textContent()).includes("✅"), "Successful update marker missing after remount");
            await selects.nth(2).selectOption("AUN-ComfyUI-Nodes");
            await selects.nth(2).selectOption("ComfyUI_ALEXZ_tools");
            require((await option.textContent()).includes("✅"), "Module info erased successful update marker");
        });
        await check("Next candidate updates without Refresh and batch count decreases", async () => {
            const selects = picker.locator(".alexz-mod-picker-selection-block select");
            await selects.nth(2).selectOption("comfyui-manager");
            const action = picker.getByRole("button", { name: "Обновить модуль", exact: true });
            await action.waitFor({ state: "visible" });
            require(await action.isEnabled(), "Next candidate unavailable");
            await action.click();
            await page.waitForFunction(() => document.querySelector(".alexz-mod-picker")?.inert);
            require(report.mockedRequests.some(item => item.path.endsWith("_apply") && item.module === "comfyui-manager"), "Wrong next module selection");
            execution = { ...execution, phase: "done", message: "Private done" };
            const item = plan.results.find(item => item.module === "comfyui-manager");
            item.status = "up_to_date"; item.before = item.target;
            plan.id = "remaining-second"; plan.batch_count = 0;
            restartRequired.push("comfyui-manager");
            await page.waitForFunction(() => !document.querySelector(".alexz-mod-picker")?.inert);
            require(await update.isVisible() && await update.isDisabled() && (await update.textContent()).includes("(0)"), "Empty batch count invalid");
            require((await picker.locator(".alexz-module-runtime-status").innerText()).includes("Требуется перезагрузка"), "Second restart status missing");
            await selects.nth(2).selectOption("ComfyUI_ALEXZ_tools");
            await page.waitForFunction(() => document.querySelector(".alexz-module-runtime-status")?.textContent.includes("Требуется перезагрузка"));
            require(report.mockedRequests.filter(item => item.path.endsWith("_check")).length === 1, "Sequential updates required another Refresh");
            restartRequired.length = 0;
            await remount();
            const currentSelects = picker.locator(".alexz-mod-picker-selection-block select");
            await currentSelects.nth(0).selectOption("custom");
            await currentSelects.nth(2).selectOption("ComfyUI_ALEXZ_tools");
            require(!(await picker.locator(".alexz-module-runtime-status").innerText()).includes("Требуется перезагрузка"), "Restart status survived a new server session");
        });
        await check("Legacy cancellation and zero-batch disabled action", async () => {
            checked = false;
            execution = { phase: "waiting_for_shutdown", message: "Private legacy", directory: "fixture-only" };
            await remount();
            await picker.getByRole("button", { name: "Отменить ожидающее обновление", exact: true }).click();
            await page.waitForFunction(() => document.querySelector(".alexz-update-planner-summary")?.textContent === "Обновление отменено.");
            require((await picker.innerText()).includes("Обновление отменено"), "Legacy cancellation failed");
            execution = null; checked = true; plan.batch_count = 0;
            await remount();
            require(await update.isVisible() && await update.isDisabled(), "Zero-batch button invalid");
        });
        await check("No real updates and no module errors", async () => {
            require(report.failures.length === 0 && errors.length === 0, [...report.failures, ...errors].join("\n"));
            require(report.mockedRequests.filter(item => item.path.endsWith("_apply")).length === 3, "Wrong apply count");
            require(report.mockedRequests.filter(item => item.path.endsWith("_check")).length === 1, "Duplicate check");
            require(!report.mockedRequests.some(item => item.path.endsWith("/module_refresh")), "Duplicate backend refresh");
        });
        report.status = "PASS";
    } catch (error) { report.failures.push(error.message); }
    finally {
        page.off("pageerror", onError);
        await page.unroute(pattern, guard);
        await page.evaluate(() => { window.confirm = window.__alexzSmokeConfirm.original; delete window.__alexzSmokeConfirm; });
    }
    return report;
}
