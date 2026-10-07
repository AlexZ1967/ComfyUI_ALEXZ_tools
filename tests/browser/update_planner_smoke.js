/** Exercise update UI against mocked API responses without changing custom nodes. */
async (page) => {
    const report = { status: "FAIL", checks: [], failures: [], mockedRequests: [] };
    const require = (condition, message) => { if (!condition) throw new Error(message); };
    const plan = {
        id: "browser-fixture", expires_at: Date.now() / 1000 + 3600,
        counts: { safe: 1, risk: 1, unknown: 1, blocked: 1, up_to_date: 1 },
        batch_count: 1, batch_error: "", batch: { additions: [] }, baseline: ["Исходная проблема fixture"],
        results: [
            { module: "Fixture_Checked", status: "safe", reasons: ["Конфликтов не обнаружено"],
                requirements_changed: false, additions: [], diff: "", before: "aaa", target: "bbb" },
            { module: "Fixture_Risk", status: "risk", reasons: ["Нужен torch>=99; защищена установленная версия"],
                requirements_changed: true, additions: [], diff: "-torch>=2\n+torch>=99", before: "aaa", target: "bbb" },
            { module: "Fixture_Unknown", status: "unknown", reasons: ["Ошибка сети"],
                requirements_changed: null, additions: [], diff: "" },
            { module: "Fixture_Blocked", status: "blocked", reasons: ["Локальные изменения"],
                requirements_changed: null, additions: [], diff: "" },
            { module: "Fixture_Current", status: "up_to_date", reasons: [],
                requirements_changed: null, additions: [], diff: "" },
        ],
    };
    let checked = false;
    let execution = null;
    const pattern = "**/alexz_tools/**";
    const errors = [];
    const onError = (error) => { if (/ALEXZ_tools|module_updates|module_node_picker/i.test(error.stack || error.message)) errors.push(error.message); };
    const guard = async (route) => {
        const request = route.request();
        const path = new URL(request.url()).pathname;
        if (/\/(module_refresh|module_refresh_status|module_acknowledge_all)$/.test(path)) {
            report.mockedRequests.push({ path, method: request.method() });
            if (path.endsWith("/module_refresh")) {
                require(request.postDataJSON().sync_upstreams === false, "Duplicate upstream fetch requested");
            }
            await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok",
                refresh: { running: false, phase: "done", modules_need_update: 2, modules_unknown_update: 1 } }) });
            return;
        }
        if (/\/alexz_tools\/update_plan_(status|check|apply|cancel)$/.test(path)) {
            report.mockedRequests.push({ path, method: request.method() });
            let payload;
            if (path.endsWith("_check")) {
                checked = true;
                payload = { status: "started" };
            } else if (path.endsWith("_apply")) {
                const input = request.postDataJSON();
                require(input.plan_id === plan.id && input.confirmed === true, "Missing explicit plan acknowledgment");
                require(input.mode === "code_only" && input.acknowledge_risk === true && input.module === "Fixture_Risk",
                    "Risk button did not preserve code-only contract");
                execution = { phase: "waiting_for_shutdown", message: "Fixture waiting", directory: "fixture-only" };
                payload = { status: "queued", message: "Fixture waiting", directory: "fixture-only" };
            } else if (path.endsWith("_cancel")) {
                execution = { ...execution, phase: "cancelled", message: "Fixture cancelled" };
                payload = { status: "cancelling" };
            } else {
                payload = { status: "ok", phase: checked ? "ready" : "idle", message: "Fixture analysis",
                    plan: checked ? plan : null, execution };
            }
            await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(payload) });
            return;
        }
        if (request.method() === "GET" && /\/(node_catalog|module_info|module_refresh_status|module_update_status|comfyui_info)$/.test(path)) {
            await route.continue();
            return;
        }
        report.failures.push(`Blocked real backend operation: ${request.method()} ${path}`);
        await route.abort("blockedbyclient");
    };
    const check = async (name, action) => {
        await action();
        report.checks.push({ name, status: "PASS" });
    };
    await page.route(pattern, guard);
    page.on("pageerror", onError);
    // MCP обрабатывает native dialogs отдельно и может завершить tool раньше
    // async-сценария. В собственной тестовой вкладке проверяем тот же confirm
    // через управляемую подмену, сохраняя и восстанавливая оригинал.
    await page.evaluate(() => {
        window.__alexzSmokeConfirm = { original: window.confirm, messages: [], accept: false };
        window.confirm = (message) => {
            window.__alexzSmokeConfirm.messages.push(message);
            return window.__alexzSmokeConfirm.accept;
        };
    });
    try {
        const host = page.locator(".alexz-update-planner");
        const picker = page.locator(".alexz-mod-picker");
        const refresh = picker.getByRole("button", { name: "Refresh Custom Nodes Info", exact: true });
        const details = picker.getByRole("button", { name: "Подробнее", exact: true });
        await host.waitFor({ state: "attached", timeout: 10000 });
        await check("No update on panel load", async () => {
            require(!report.mockedRequests.some(item => item.method === "POST"), "Panel performed automatic work");
            require(await refresh.count() === 1, "Unified refresh button missing");
            require(await picker.getByRole("button", { name: "Проверить обновления и зависимости", exact: true }).count() === 0,
                "Duplicate check button remains");
            require(await host.isHidden(), "Details expanded on load");
        });
        await check("Counts and separate risk statuses", async () => {
            await refresh.click();
            await page.waitForFunction(() => document.querySelector(".alexz-update-planner-summary")?.textContent === "Найдено обновлений: 2 модулей.");
            require(await host.isHidden(), "Details automatically expanded after refresh");
            require((await picker.locator(".alexz-update-planner-summary").innerText()).split("\n").length === 1,
                "Summary must remain short");
            await details.click();
            await host.waitFor({ state: "visible" });
            await host.locator("details[data-module='Fixture_Risk']").waitFor({ state: "attached" });
            require((await host.locator(".alexz-update-planner-diagnostics").innerText()).includes("Проверенных для общего апдейта: 1"), "Batch count incorrect");
            for (const status of ["safe", "risk", "unknown", "blocked", "up_to_date"]) {
                require(await host.locator(`details[data-status='${status}']`).count() === 1, `Missing ${status}`);
            }
        });
        const risky = host.locator("details[data-module='Fixture_Risk']");
        await check("Dependency diff and blocked actions", async () => {
            await risky.locator("summary").click();
            require((await risky.locator("pre").innerText()).includes("torch>=99"), "Requirements diff missing");
            require(await host.locator("details[data-status='blocked'] button").count() === 0, "Dirty module can be updated");
            require(await host.locator("details[data-status='unknown'] button").count() === 0, "Unknown Git target can be updated");
        });
        await check("Dismissed confirmation does not apply", async () => {
            await risky.getByRole("button", { name: "Апдейт только кода…", exact: true }).click();
            await page.waitForTimeout(150);
            const observed = await page.evaluate(() => window.__alexzSmokeConfirm.messages.at(-1));
            require(observed?.includes("модуль может перестать загружаться"), "Risk confirmation missing");
            require(!report.mockedRequests.some(item => item.path.endsWith("_apply")), "Dismissed action reached API");
        });
        await check("Acknowledged code-only queue and cancellation", async () => {
            await page.evaluate(() => { window.__alexzSmokeConfirm.accept = true; });
            await risky.getByRole("button", { name: "Апдейт только кода…", exact: true }).click();
            const cancel = host.getByRole("button", { name: "Отменить ожидающее обновление", exact: true });
            await cancel.waitFor({ state: "visible" });
            require(await host.getByRole("button", { name: "Обновить проверенные (1)", exact: true }).isDisabled(), "Queue allows duplicate update");
            await cancel.click();
            await page.waitForFunction(() => document.querySelector(".alexz-update-planner-summary")?.textContent.includes("cancelled"));
        });
        await check("No real mutation or module error", async () => {
            require(report.failures.length === 0 && errors.length === 0, [...report.failures, ...errors].join("\n"));
            require(report.mockedRequests.filter(item => item.path.endsWith("_apply")).length === 1, "Unexpected apply count");
            require(report.mockedRequests.filter(item => item.path.endsWith("/update_plan_check")).length === 1, "Duplicate check");
            require(report.mockedRequests.filter(item => item.path.endsWith("/module_refresh")).length === 1, "Duplicate catalog refresh");
            await details.click();
            require(await host.isHidden(), "Details do not collapse");
            await details.click();
            require(await host.isVisible(), "Details cannot reopen");
        });
        report.status = "PASS";
    } catch (error) {
        report.failures.push(error.message);
    } finally {
        page.off("pageerror", onError);
        await page.unroute(pattern, guard);
        await page.evaluate(() => {
            window.confirm = window.__alexzSmokeConfirm.original;
            delete window.__alexzSmokeConfirm;
        });
    }
    return report;
}
