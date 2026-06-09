# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Contexto del proyecto (Equipo 52 · TC5035.10 · MNA Tec)

Pregunta de investigación: **¿los métodos de image enhancement mejoran la estimación de profundidad monocular en endoscopía (SCARED)?**
- **Avance 4**: pesos originales de cada paper. **Avance 5**: pesos re-entrenados en SCARED (Dr. Ricardo Espinosa Loera).
- **New Version** de ambos: usan el **split oficial de AF-SfMLearner** (`data/splits/endovis/test_files.txt`, 550 frames, datasets 1-7) y añaden **Endo-STTN** como 5º enhancement.

## Cómo se ejecuta (CRÍTICO)
- **El kernel corre en Google Colab; el repo vive en el disco local del usuario.** Yo NO veo Drive ni el disco externo `E:` (TOSHIBA EXT) desde mi sesión — debo pedir al usuario que corra snippets en Colab y me pegue la salida.
- En Colab el repo se clona en `/content/repo_52`. Si ya existe de una sesión previa, hace `git pull --ff-only` (no se re-clona solo). **Editar/recargar un archivo NO cambia lo que ya está corriendo en el kernel de Colab**: una celda en ejecución conserva en memoria la versión vieja de las funciones hasta que se re-ejecuta esa celda.
- **Editar notebooks vía Python + json.dump**: SIEMPRE normalizar `source` a lista (`splitlines(keepends=True)`) y quitar `outputs`/`execution_count` de celdas markdown, o GitHub deja de renderizar (nbconvert error). Validar con `nbconvert --to html` antes de commitear. Ver [[feedback_notebook_json_editing]].
- HTML de notebooks se generan a `docs/reports/` (CON código, sin `--no-input`) y se parchean con `patch_mermaid.py`. Enlazados en `reader.html` (tabs) e `index.html`.
- Entorno local de validación: `C:\Users\User\anaconda3\envs\computer_vision\python.exe`. Ver [[reference_python_env]].

## Rendimiento del experimento
- **Chamfer es el cuello de botella** (KD-Trees en CPU, ~16500×, lleva la corrida a ~7h) y **NO está en la tabla que pide el profe**. Está tras el flag `COMPUTE_CHAMFER = False` (default). Reactivable solo si se necesita métrica 3D (p.ej. para un paper MICAI). Hay una celda opcional "5b" que calcula Chamfer rápido solo para `none` vs `endosttn`.
- El A100 NO acelera la extracción (es CPU + I/O de Drive), solo la inferencia.
- **Caché persistente**: la extracción de los 550 frames se guarda en `BASE/split_frames.npz` (Drive). Primera vez extrae; sesiones siguientes cargan en segundos. Flag `FORCE_REEXTRACT`.
- El loop tiene **checkpointing** (CSV cada 50 evals): parar y reanudar NO pierde avance.
- Usar **`tqdm` clásico, NO `tqdm.notebook`** (los ipywidgets rompen el render en VS Code/GitHub).

## Git
- Commit/push con usuario `jmtoral` / `jmtoralcruz@gmail.com`. Mensajes simples SIN tildes ni caracteres que rompan PowerShell here-strings.
- `git add` con rutas explícitas (no `-A`: dispara bloqueos del sandbox). Evitar `Select-String`/`find`/`grep` por shell.
