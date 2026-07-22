# Frontend Checks

Все команды запускаются из корня `ComfyUI_ALEXZ_tools` и используют Conda
environment `p313`, заданный в `Makefile`.

## Быстрые команды

- `make js-check-all` рекурсивно проверяет синтаксис всех `.js` и `.mjs` файлов
  в `web/` через текущий Node.js runtime.
- `make js-test` запускает lightweight behavioral test для `Module Node Picker`.
- `make test` последовательно запускает docs-check, полный Python pytest,
  frontend syntax check и behavioral JS test.

Старый target `make js-check` сохранён как совместимый alias для
`make js-check-all`.

## Диагностика

При синтаксической ошибке `js-check-all` печатает стандартную диагностику Node.js
и относительный путь проблемного файла, после чего возвращает ненулевой exit
code. Если behavioral test падает, исправлять следует первое assertion/error в
выводе `make js-test`.

Для запуска в другом Conda environment передайте его явно:

```bash
make test CONDA_ENV=my_env
```
