# Reporte de Exploración: Dataset C3VDv2

**Notebook:** `notebooks/04_exploracion_c3vdv2.ipynb`
**Fecha de ejecución:** 2026-05-11
**Autor:** Proyecto Integrador — Comparativa de Métodos de Reconstrucción 3D en Colonoscopía

---

## 1. Objetivo

Caracterizar el dataset C3VDv2 (*Colonoscopy 3D Video Dataset versión 2*, Golhar et al., Johns Hopkins University, 2025) como insumo para la fase de modelado del proyecto, donde se compararán tres métodos de reconstrucción 3D (Endo-Depth-and-Motion, EndoGaussian y Neural Exposure Fields). El análisis responde tres preguntas:

1. ¿Cuál es la estructura del dataset y qué afirmaciones del paper conviene validar empíricamente?
2. ¿Qué patrones de missingness, sesgo y correlación presentan los metadatos?
3. ¿Qué escenarios clínicos son más problemáticos para el algoritmo de registro 2D/3D del paper?

---

## 2. Volumetría del dataset

### 2.1 Composición por tipo de video

| Tipo | n videos | Descripción | GT por píxel |
|---|---|---|---|
| v1 | 63 | Colon limpio, trayectoria baseline | Sí |
| v2 | 57 | Colon limpio, trayectoria alternativa | Sí |
| v3 | 57 | Colon con debris, misma trayectoria que v2 | Heredado de v2 |
| v4 | 15 | Deformación activa del phantom | No |
| full | 8 | Screening simulado completo (gastroenterólogo) | No |
| **TOTAL** | **192 videos** | **169 371 cuadros** | — |

**Adquisición:** Olympus CF-HQ190L montado sobre brazo robótico UR-3e de 6-DoF, sobre phantoms de silicona de alta fidelidad. Lente fisheye con modelo de cámara omnidireccional de Scaramuzza (no pinhole). Licencia CC BY 4.0.

### 2.2 Modalidades de ground truth por frame (videos v1/v2/v3)

| Modalidad | Archivo | Codificación |
|---|---|---|
| RGB | `rgb/NNNN.png` | 1350×1080, uint8 |
| Profundidad | `depth/NNNN_depth.tiff` | uint16, escala lineal 0–100 mm |
| Normales | `normals/NNNN_normals.tiff` | 16-bit RGB, ±1 → 0–65535 |
| Optical flow | `optical_flow/NNNN_flow.tiff` | 16-bit RG, ±20 px |
| Oclusión | `occlusions/NNNN_occlusion.tiff` | 8-bit binario |
| Diffuse | `diffuse/NNNN_diffuse.png` | 8-bit grayscale |
| Pose | `pose.txt` | 4×4 homogénea row-major por línea |
| Malla GT | `coverage_mesh.obj` | OBJ con vértices observados / no observados |

---

## 3. Etapa 1 — Integridad y missingness

### 3.1 Clasificación de valores faltantes

Todo el missingness observado es **estructural**, explicable por el diseño del dataset. No se detectaron anomalías que requieran imputación.

| Columna | Nulos | Justificación |
|---|---|---|
| Quantitative Score | 23 (v3 + v4 + full) | v3 hereda registro de v2; v4/full no tienen GT por píxel |
| Qualitative Score | 23 (v4 + full) | Sin GT por píxel para calificar |
| Camera Speed | 8 (full) | Captura manual sin configuración UR3e |
| Brightness | 8 (full) | Idem |

### 3.2 Cardinalidad de variables categóricas

| Variable | Cardinalidad | Valores |
|---|---|---|
| Colon | 2 | c1, c2 |
| Segment | 10 | ascending, cecum, descending, full, rectum, sigmoid, sigmoid1, sigmoid2, transverse1, transverse2 |
| Segment Full (desambiguado por colon) | 17 | c1_ascending, c1_cecum, …, c2_sigmoid |
| Phantom Number (textura) | 4 | t1, t2, t3, t4 |
| Video Number | 4 | v1, v2, v3, v4 |
| Qualitative Score | 3 | 1.0, 2.0, 3.0 |

**Hallazgo:** El segmento `sigmoid` de Colon 2 es anatómicamente distinto a `sigmoid1` y `sigmoid2` de Colon 1. No deben agregarse. La desagregación por colon es obligatoria.

### 3.3 Integridad referencial del pareo v2/v3

El paper afirma que cada video v3 reutiliza la trayectoria del robot del video v2 correspondiente. La verificación encontró **56 pares completos** (no 57) con dos asimetrías no documentadas en el paper:

- `c1_ascending_t1` tiene v2 pero no v3
- `c2_transverse2_t4` tiene v3 pero no v2

---

## 4. Etapa 2 — Análisis univariado

### 4.1 Sesgo de variables numéricas

| Variable | n | Media | σ | Skewness | Transformación |
|---|---|---|---|---|---|
| Camera Speed | 169 | 17.99 | 10.18 | +2.07 | Requerida (log) |
| Camera Speed (log) | 169 | 2.84 | 0.44 | +0.67 | Opcional |
| Brightness | 169 | 0.80 | 3.33 | +0.22 | Ninguna |
| Edge Enhancement | 177 | 2.55 | 0.61 | −0.41 | Ninguna |
| Quantitative Score | 112 | −0.02 | 0.98 | −0.55 | Opcional |
| Total Frames | 177 | 921.95 | 2 431.88 | +4.53 | Requerida (log) |

**Hallazgo:** Camera Speed (+2.07) y Total Frames (+4.53) presentan sesgo fuerte. La transformación log reduce Camera Speed a +0.67. Brightness, Edge Enhancement y Quantitative Score son razonablemente simétricos.

### 4.2 Análisis multi-etiqueta de Tags

Los videos están anotados con 0 a 5 tags por video (15 tags únicos en total). Tras higiene de cadenas (eliminación de dobles espacios, corrección `zizag` → `zigzag`):

- Tags más frecuentes: `polyp` (n=48), `textureless enface` (n=34), `fast` (n=26), `zigzag` (n=25), `saturation` (n=22), `loop` (n=21), `water on lens` (n=21).
- Co-ocurrencias relevantes: `polyp` aparece con casi todos los demás tags por ser el más frecuente; `fast` y `zigzag` co-ocurren en 6 videos (ambos describen movimiento no-suave).

---

## 5. Etapa 3 — Análisis bivariado

### 5.1 Matriz de correlación (Spearman)

Únicamente una correlación resulta estadísticamente significativa:

- **Camera Speed ↔ Total Frames:** ρ = −0.60 (p = 1.4 × 10⁻¹⁷). Velocidades mayores producen videos más cortos.

El resto de correlaciones permanecen en el rango [−0.15, +0.15] y se interpretan como efectivamente nulas.

### 5.2 Validación del sistema de scoring del paper

Las medianas del score cuantitativo por categoría cualitativa siguen un patrón monótono que valida el doble sistema de scoring del paper:

| Qualitative Score | Mediana de Quantitative Score |
|---|---|
| 1 (mejor) | +0.62 |
| 2 | −0.29 |
| 3 (peor) | −1.01 |

Esta correspondencia autoriza el uso del Qualitative Score como filtro fiable de calidad.

### 5.3 Tasa de fallo de registro por escenario clínico

Tras normalización por fila (qué fracción de cada tag cae en cada score):

| Tag | n | Score 1 | Score 2 | Score 3 | Interpretación |
|---|---|---|---|---|---|
| debris on lens | 5 | 80% | 20% | 0% | Bien tolerado |
| straight line | 9 | 78% | 0% | 22% | Trayectoria simple |
| mirrored path | 8 | 62% | 0% | 38% | Re-visitación estresa registro |
| fast | 26 | 58% | 23% | 19% | Robusto |
| zigzag | 25 | 56% | 28% | 16% | Robusto |
| **loop** | **21** | **29%** | **24%** | **48%** | **MAYOR tasa de fallo** |
| **water on lens** | **21** | **33%** | **24%** | **43%** | **Segundo más problemático** |
| polyp | 48 | 35% | 25% | 40% | Frecuente pero no el más problemático |

**Hallazgo central:** El conteo absoluto sugería que `polyp` era el escenario más problemático por aparecer en muchos videos score 3. La normalización por fila reorienta la interpretación: las trayectorias circulares (`loop`, `mirrored path`) y los obstáculos sobre la lente (`water on lens`) son las verdaderas dificultades para el algoritmo de registro 2D/3D.

---

## 6. Etapas 4-6 — Análisis con muestra limitada

Las etapas siguientes se ejecutaron sobre muestras estratificadas por restricciones de acceso al repositorio JHU Dataverse (HTTP 403 en el endpoint de listado sin autenticación). Los resultados son válidos para inspección comparativa pero no estadísticamente representativos.

### 6.1 Etapa 4 — Análisis cinemático (n=9 videos)

| Métrica | Rango observado |
|---|---|
| Frames por secuencia | 134 – 1037 |
| Magnitud de desplazamiento inter-frame (mediana) | 0.2 – 0.5 unidades |
| Outliers extremos detectados | 1 secuencia con picos hasta 6.5 |

**Hallazgo:** Las ráfagas extremas representan un riesgo para los métodos de tracking fotométrico (Endo-Depth-and-Motion) por violar el supuesto de movimiento pequeño entre frames. Los métodos que reciben poses como input (EndoGaussian, NExF) no son afectados por esta variabilidad.

### 6.2 Etapa 5 — Análisis multimodal (1 video, 3 frames)

Sobre el segmento `c1_sigmoid1_t2_v2`:

- **RGB:** brillo medio 161.1 sobre 255 (exposición razonable, no saturada ni sub-expuesta).
- **Profundidad:** rango efectivo 0–88.1 mm, consistente con la codificación documentada (uint16, escala 0–100 mm).
- **Normales:** 3 canales 16-bit codificando direcciones (X, Y, Z), escalados de ±1 a 0–65535.

### 6.3 Etapa 6 — Balance espacial de profundidad (10 frames)

Sobre 13 559 510 píxeles útiles tras enmascarar viñeteo fisheye:

| Banda de profundidad | Proporción |
|---|---|
| Cerca (0–20 mm) | 32.7% |
| Medio (20–50 mm) | 51.6% |
| Lejos (50–100 mm) | 15.7% |

**Hallazgo:** El campo medio domina por geometría del lumen colónico (paredes laterales próximas ocupan banda delgada; cuerpo del lumen ocupa la mayoría del área visual). Una función de pérdida uniforme en píxeles asignará peso ~3× mayor al campo medio que al lejano. Una pérdida espacialmente balanceada es la mitigación estándar.

---

## 7. Resumen ejecutivo

| Dimensión | Resultado |
|---|---|
| Videos totales | 192 |
| Cuadros totales | 169 371 |
| Videos con GT por píxel (v1+v2+v3) | 177 |
| Pares completos v2/v3 | 56 (con 2 asimetrías documentadas) |
| Resolución RGB | 1350×1080 px |
| Codificación profundidad | uint16, escala 0–100 mm |
| Modelo de cámara | Scaramuzza omnidireccional (no pinhole) |
| Variables con sesgo fuerte | Camera Speed (+2.07), Total Frames (+4.53) |
| Correlaciones significativas | Sólo Camera Speed ↔ Total Frames (ρ = −0.60) |
| Tags más problemáticos | loop (48%), water on lens (43%) |
| Validación del scoring del paper | Monótona y consistente |

### Hallazgos relevantes para la fase de modelado

1. **Fuga de información por pares v2/v3.** Los 56 pares comparten trayectoria idéntica. Mantener cada par en el mismo split (entrenamiento o prueba) es obligatorio para evitar fuga.

2. **Score 3 contamina entrenamiento.** Los videos con Qualitative Score 3 tienen artefactos de registro confirmados por la validación cruzada con el Quantitative Score. Excluirlos del conjunto de entrenamiento; reservarlos como stress-test.

3. **Modelo de cámara fisheye no estándar.** Asumir proyección pinhole introducirá errores sistemáticos. Los tres métodos a comparar deben configurarse con soporte fisheye/omnidireccional desde la implementación.

4. **Función de pérdida espacialmente balanceada.** El balance preliminar 33/52/16 (cerca/medio/lejos) sugiere sub-representación del campo lejano. Una pérdida ponderada por banda de profundidad es la mitigación estándar para regresión directa (Endo-Depth-and-Motion).

5. **Tags problemáticos.** `loop`, `water on lens` y `mirrored path` son los escenarios con mayor tasa de fallo de registro. Reportar métricas estratificadas por escenario clínico.

---

## 8. Limitaciones del análisis

- **Etapas 1-3 (cobertura completa):** ejecutadas sobre los 192 videos del Excel de metadatos. Resultados estadísticamente sólidos.
- **Etapa 4 (cobertura parcial):** ejecutada sobre 9 archivos `pose.txt` descargados manualmente vía navegador. La trazabilidad al nombre canónico se perdió (los archivos quedaron como `pose(1).txt`, `pose(2).txt`).
- **Etapa 5 (cobertura parcial):** 3 frames de 1 único segmento.
- **Etapa 6 (cobertura parcial):** 10 frames del mismo segmento que la Etapa 5.

Las limitaciones de las Etapas 4-6 se deben a que el repositorio Johns Hopkins Dataverse devuelve HTTP 403 al endpoint de metadatos API sin autenticación, y el proceso de creación de cuenta no pudo completarse en el plazo del avance.

---

## 9. Siguientes pasos

- **`01_eda_estratificado.ipynb`** — Cerrar Etapas 5 y 6 con muestreo estratificado sobre ~30 videos para informar las decisiones de normalización RGB y función de pérdida espacial.
- **`02_validacion_pareo_v2v3.ipynb`** — Cuantificar la diferencia frame-a-frame entre trayectorias v2 y v3 pareadas. Debería ser sub-milimétrica dada la repetibilidad ±0.03 mm del UR3e.
- **`03_split_estratificado.ipynb`** — Construir la partición train/val/test respetando: pares v2/v3 juntos, estratificación por (colon, segment, texture), score 3 sólo en test, v4 como conjunto cualitativo separado.
