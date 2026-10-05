/** Run read-only frontend smoke checks using the Page supplied by Playwright MCP. */
async (page) => {
    const baseURL = "http://127.0.0.1:8188";
    const extensionPath = "/extensions/ComfyUI_ALEXZ_tools/";
    const moduleName = "ComfyUI_ALEXZ_tools";
    const deadline = Date.now() + 60000;
    const remaining = () => Math.max(1, deadline - Date.now());
    const report = { status: "FAIL", url: baseURL, checks: [], failures: [], backgroundErrors: [], warnings: 0 };
    const responses = new Map();
    const consoleTasks = [];
    let helloLoaded = false;
    // generated-файлы определяются по URL/stack с именем пакета ALEXZ_tools.
    const belongsToAlexz = (text) => /ALEXZ_tools|ALEXZ[._](?:Tools|ModulePicker)|module_node_picker/i.test(text);
    const recordError = (text) => {
        (belongsToAlexz(text) ? report.failures : report.backgroundErrors).push(text);
    };
    const isExtension = (url) => new URL(url).pathname.startsWith(extensionPath);
    const onConsole = (message) => {
        if (message.type() === "warning") report.warnings += 1;
        if (message.text().includes("[ALEXZ_tools] TypeScript frontend extension loaded.")
            && message.location().url.includes(`${extensionPath}generated/hello.js`)) helloLoaded = true;
        if (message.type() !== "error") return;
        // Ошибка может находиться во вложенном объекте console.error, а не в location.
        consoleTasks.push((async () => {
            const details = await Promise.all(message.args().map((arg) => arg.evaluate((value) => {
                try {
                    return JSON.stringify(value, (_key, item) => item instanceof Error
                        ? { message: item.message, stack: item.stack } : item);
                } catch {
                    return String(value);
                }
            }).catch(() => "")));
            recordError(`${message.text()} ${message.location().url} ${details.join(" ")}`);
        })());
    };
    const onPageError = (error) => recordError(error.stack || error.message);
    const onResponse = (response) => {
        if (!isExtension(response.url())) return;
        responses.set(new URL(response.url()).pathname, response.status());
        if (response.status() >= 400) report.failures.push(`HTTP ${response.status()}: ${response.url()}`);
    };
    const onRequestFailed = (request) => {
        if (isExtension(request.url())) {
            report.failures.push(`Request failed: ${request.url()} ${request.failure()?.errorText}`);
        }
    };
    // Разрешены только чтение каталога/информации и polling текущего статуса jobs.
    const guardPattern = /\/alexz_tools\//;
    const guard = async (route) => {
        const request = route.request();
        const url = new URL(request.url());
        const endpoint = url.pathname.replace(/^\/api/, "");
        const query = url.searchParams;
        const allowed = request.method() === "GET" && (
            endpoint === "/alexz_tools/node_catalog" && query.get("cache_only") === "1"
            || endpoint === "/alexz_tools/module_info" && query.get("cache_only") === "1"
                && !["refresh", "sync_upstream"].some((key) => query.has(key) && query.get(key) !== "0")
            || ["/alexz_tools/module_refresh_status", "/alexz_tools/module_update_status"].includes(endpoint)
        );
        if (allowed) return route.continue();
        report.failures.push(`Blocked backend operation: ${request.method()} ${request.url()}`);
        await route.abort("blockedbyclient");
    };
    const require = (condition, message) => { if (!condition) throw new Error(message); };
    const check = async (name, action) => {
        try {
            await action();
            report.checks.push({ name, status: "PASS" });
        } catch (error) {
            report.checks.push({ name, status: "FAIL" });
            throw error;
        }
    };

    page.on("console", onConsole);
    page.on("pageerror", onPageError);
    page.on("response", onResponse);
    page.on("requestfailed", onRequestFailed);
    await page.route(guardPattern, guard);
    try {
        await check("ComfyUI loaded", async () => {
            const response = await page.goto(baseURL, { waitUntil: "load", timeout: remaining() });
            require(response?.ok(), "ComfyUI document did not return HTTP 2xx");
            await page.waitForFunction(() => {
                const app = window.comfyAPI?.app?.app;
                return document.readyState === "complete" && app?.vueAppReady && app?.graph && app?.canvas;
            }, null, { timeout: remaining() });
        });
        await check("ALEXZ.Tools.Hello setup", async () => {
            await page.waitForFunction(() => window.comfyAPI.app.app.extensions
                .some((extension) => extension.name === "ALEXZ.Tools.Hello"), null, { timeout: remaining() });
            // Регистрация extensions завершается раньше вызова setup.
            while (!helloLoaded && Date.now() < deadline) await page.waitForTimeout(100);
            require(helloLoaded, "Hello was registered but its setup log was not observed");
        });
        await check("Module Node Picker opens", async () => {
            const button = page.getByTestId("alexz-module-nodes-tab-button");
            await button.waitFor({ state: "visible", timeout: remaining() });
            const active = await page.evaluate(() => window.comfyAPI.app.app.extensionManager.sidebarTab.activeSidebarTabId);
            if (active !== "alexz-module-nodes") await button.click({ timeout: remaining() });
            await page.locator(".alexz-mod-picker").waitFor({ state: "visible", timeout: remaining() });
        });
        await check("Picker DOM and node list", async () => {
            const panel = page.locator(".alexz-mod-picker");
            await panel.locator(".alexz-mod-picker-title").waitFor({ timeout: remaining() });
            require(await panel.locator(".alexz-mod-picker-title").innerText() === "Node Picker", "Picker title missing");
            require(await panel.getByRole("button", { name: "Debug", exact: true }).count() === 1, "Debug control missing");
            require(await panel.locator("select[title='ComfyUI update-check mode']").count() === 1, "Mode control missing");
            require(await panel.locator(".alexz-mod-picker-selection-block select").count() === 3, "Selection controls missing");
            require(await panel.locator(".alexz-mod-picker-selection-block input[type='text']").count() === 1, "Module filter missing");
            const selects = panel.locator(".alexz-mod-picker-selection-block select");
            await panel.locator(".alexz-mod-picker-selection-block input[type='text']").fill("", { timeout: remaining() });
            await selects.nth(0).selectOption("custom", { timeout: remaining() });
            // Второй select выбирает группу ComfyUI; третий — custom module.
            await selects.nth(2).locator(`option[value='${moduleName}']`).waitFor({ state: "attached", timeout: remaining() });
            await selects.nth(2).selectOption(moduleName, { timeout: remaining() });
            const card = panel.locator(".alexz-mod-picker-module-card");
            await card.locator(".alexz-mod-picker-module-title").filter({ hasText: moduleName })
                .waitFor({ timeout: remaining() });
            if (await panel.locator(".alexz-mod-picker-node").count() === 0) {
                await card.locator(".alexz-mod-picker-module-title").click({ timeout: remaining() });
            }
            await panel.locator(".alexz-mod-picker-node").first().waitFor({ state: "visible", timeout: remaining() });
        });
        await check("Catalog and module-info API", async () => {
            const result = await page.evaluate(async (module) => {
                const api = window.comfyAPI.api.api;
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), 10000);
                try {
                    const read = async (path) => {
                        const response = await api.fetchApi(path, { cache: "no-store", signal: controller.signal });
                        return { status: response.status, payload: await response.json() };
                    };
                    const catalog = await read("/alexz_tools/node_catalog?cache_only=1&comfyui_mode=releases");
                    const info = await read(`/alexz_tools/module_info?group=custom&module=${module}&refresh=0&sync_upstream=0&cache_only=1`);
                    const extensions = await read("/extensions");
                    return { catalog, info, extensions };
                } finally {
                    clearTimeout(timer);
                }
            }, moduleName);
            for (const [name, value] of Object.entries(result)) {
                require(value.status >= 200 && value.status < 300 && !value.payload.error, `${name}: HTTP/JSON error`);
            }
            require(result.catalog.payload.groups?.some((group) => group.id === "custom"
                && group.nodes?.some((node) => node.module === moduleName)), "ALEXZ nodes missing from catalog");
            require(result.info.payload.module === moduleName && result.info.payload.info?.module === moduleName,
                "Module-info payload does not describe ALEXZ_tools");
            report.extensionFiles = result.extensions.payload.filter((url) => new URL(url, baseURL).pathname.startsWith(extensionPath));
        });
        await check("Extension resources loaded", async () => {
            const expected = report.extensionFiles.map((url) => new URL(url, baseURL).pathname);
            require(expected.includes(`${extensionPath}generated/hello.js`), "Hello bundle missing from /extensions");
            require(expected.includes(`${extensionPath}module_node_picker.js`), "Picker missing from /extensions");
            const missing = expected.filter((path) => !responses.has(path));
            require(missing.length === 0, `Advertised extensions were not loaded: ${missing.join(", ")}`);
            require(expected.every((path) => responses.get(path) < 400), "Extension HTTP failures observed");
        });
        // Небольшое окно для отложенных ошибок после render; networkidle здесь не подходит.
        await page.waitForTimeout(1000);
        await Promise.all(consoleTasks);
        await check("No ALEXZ errors or unsafe operations", async () => {
            require(report.failures.length === 0, report.failures.join("\n"));
        });
        report.status = "PASS";
    } catch (error) {
        report.failures.push(error.message);
    } finally {
        await Promise.all(consoleTasks);
        page.off("console", onConsole);
        page.off("pageerror", onPageError);
        page.off("response", onResponse);
        page.off("requestfailed", onRequestFailed);
        await page.unroute(guardPattern, guard);
        if (report.failures.length) report.status = "FAIL";
    }
    report.backgroundErrorCount = report.backgroundErrors.length;
    report.backgroundErrors = report.backgroundErrors.slice(0, 5);
    report.extensionFiles = report.extensionFiles?.length || 0;
    return report;
}
