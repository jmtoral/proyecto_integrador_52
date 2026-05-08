# La luz que opaca

Corrección de iluminación como preprocesamiento para estimación de profundidad monocular en endoscopía.

## Descripción

Este repositorio contiene el proyecto integrador de maestría, cuyo objetivo es evaluar si la corrección del gradiente de iluminación del endoscopio mejora el desempeño de modelos de estimación de profundidad monocular en video endoscópico.

Los datos empleados en esta investigación responden a una naturaleza semiestructurada y no tabular. A diferencia de los datasets tabulares convencionales donde cada observación es una fila y cada variable una columna, los datos endoscópicos se organizan como secuencias de video estéreo, mapas de profundidad tridimensionales, nubes de puntos y archivos de calibración de cámara que deben interpretarse conjuntamente para tener significado clínico.

El pipeline completo es:

```
video endoscópico (RGB crudo)
        ↓
  corrección de iluminación   ← contribución del proyecto
        ↓
  modelo de profundidad monocular  (MonoIIT / NExF / EndoGaussian)
        ↓
  depth map estimado
        ↓
  evaluación vs. GT de structured light  (AbsRel, RMSE, Chamfer Distance)
```

La hipótesis es que eliminar el gradiente radial de iluminación — más brillante en el centro, oscuro en los bordes — reduce el error de profundidad, especialmente en la periferia de la imagen donde la validez del GT cae más rápido.

## Datasets

Se trabajan con tres bases de datos clínicas endoscópicas:

### SCARED

**SCARED** (Structured light endoscopY And Reconstructed Depth) ofrece mapas de profundidad reales capturados con luz estructurada sobre tejido porcino (Allan et al., 2021). Es el dataset principal del proyecto.

El dataset completo (versión extendida en HuggingFace) contiene **9 datasets** de diferentes cerdos o sesiones, cada uno con **5 keyframes** — aproximadamente **45 keyframes** en total, ~250 GB en disco.

| Archivo por keyframe | Descripción |
|---|---|
| `Left_Image.png` | Frame de referencia RGB (1280×1024 px) — entrada del modelo |
| `Right_Image.png` | Cámara derecha (par estéreo) |
| `left_depth_map.tiff` | GT de profundidad (X,Y,Z) float32 en mm por structured light |
| `point_cloud.obj` | Nube de puntos 3D en bruto (GT para Chamfer Distance) |
| `endoscope_calibration.yaml` | Matriz intrínseca K de la cámara (aproximada) |
| `data/rgb.mp4` | Video del keyframe (frames L+R apilados verticalmente) |
| `data/frame_data.tar.gz` | JSONs con pose de cámara por cuadro del video |
| `data/scene_points.tar.gz` | TIFFs de depth warpeado por cuadro (~2 GB, no es GT perfecto) |

**Notas técnicas:**
- Los TIFF se leen con `tifffile` (float32), no con visores de imagen normales.
- Los depth maps warpeados en `scene_points.tar.gz` tienen pequeños errores por cinemática o sincronización — usar solo para entrenamiento auxiliar.
- La calibración es aproximada; los modelos deben ser robustos a errores en K.
- `dataset_8` y `dataset_9` indexan keyframes desde 0 (`keyframe_0…keyframe_4`); los demás desde 1.
- `keyframe_4` de `dataset_1` no tiene carpeta `data/` por error de logging en la captura.

Los datos crudos no están en el repositorio por su tamaño.

### Hamlyn

**Hamlyn** provee secuencias endoscópicas estéreo in-vivo (Recasens et al., 2021). Complementa SCARED con tejido real y condiciones clínicas más variadas.

### C3VD

**C3VD** (Colonoscopy 3D Video Dataset) aporta realismo clínico mediante maniquíes con mapas de referencia a nivel de píxel (Golhar et al., 2025).

## Estructura de archivos y formatos

Entender la organización física de los archivos en carpetas es fundamental para extraer y emparejar sus contenidos correctamente. Cada tipo de archivo tiene un rol específico:

| Tipo | Extensión | Contenido | Cómo se lee |
|---|---|---|---|
| Imagen RGB | `.png` | Foto del tejido (uint8, 0–255) | `cv2.imdecode` |
| Mapa de profundidad | `.tiff` | Coordenadas XYZ por píxel (float32, mm) | `tifffile.imread` |
| Nube de puntos GT | `.obj` | Vértices 3D en texto plano | Parseo manual de líneas `v ` |
| Calibración | `.yaml` | Matriz intrínseca K | `cv2.FileStorage` |
| Video de movimiento | `.mp4` | Frames L+R apilados verticalmente | `cv2.VideoCapture` / ffmpeg |
| Poses por cuadro | `frame_data.tar.gz` | JSONs con transformaciones de cámara | `tarfile` + `json.load` |
| Depths warpeados | `scene_points.tar.gz` | TIFFs de profundidad por cuadro de video | `tarfile` + `tifffile` |

Todo el dataset puede leerse **directamente del ZIP en memoria** (`io.BytesIO`) sin extraer nada al disco.

## EDA

El notebook [`notebooks/00_exploracion_scared.ipynb`](notebooks/00_exploracion_scared.ipynb) contiene la exploración inicial del dataset SCARED. Responde tres preguntas básicas:

1. **¿Qué hay en cada ZIP?** — Cantidad de keyframes, archivos y tamaños por dataset.
2. **¿Cómo se ve cada tipo de archivo?** — Visualización de PNG, TIFF, OBJ, YAML, MP4, TAR.GZ.
3. **¿Qué dimensiones y rangos tienen los datos?** — Resoluciones, rangos de profundidad, cobertura de GT.

**Hallazgos principales:**

- Resolución de imágenes: **1280×1024 px** en todos los keyframes.
- El endoscopio opera a **40–130 mm** del tejido según el keyframe.
- La cobertura de GT varía entre **~70% y ~87%** de los píxeles; el resto son reflejos especulares y bordes de curvatura extrema donde el structured light no puede triangular.
- El gradiente de iluminación es visible y medible: la luminancia cae del centro hacia los bordes de forma consistente.
- Los archivos `.obj` tienen vértices parseables directamente de líneas `v x y z`.
- `dataset_8` y `dataset_9` indexan keyframes desde 0 — no hardcodear nombres de keyframe.

Los outputs del EDA (PNGs y CSVs) están en [`outcomes/eda_outputs/`](outcomes/eda_outputs/).

## Variables del experimento

**Variable independiente:** imagen RGB con o sin corrección de iluminación  
**Variable dependiente:** error de estimación de profundidad (AbsRel, RMSE, Chamfer Distance)

La corrección de iluminación transforma `X_crudo → X_corrected` antes de pasarlo al modelo de profundidad. Todo lo demás (modelo, hiperparámetros, GT) permanece constante para aislar el efecto.

## Estructura del repositorio

```text
.
├── artifacts/              # Salidas temporales de experimentos
├── configs/                # Configuraciones versionables
│   ├── data/
│   ├── model/
│   └── train/
├── data/
│   ├── annotations/        # Máscaras, etiquetas, metadatos
│   ├── external/
│   ├── interim/            # Datos parcialmente transformados
│   ├── processed/          # Datos listos para modelado
│   └── raw/                # Datos originales (no tracked en git)
├── docs/                   # Notas metodológicas
├── notebooks/
│   └── 00_exploracion_scared.ipynb  # EDA inicial de archivos y formatos
├── outcomes/
│   └── eda_outputs/        # PNGs y CSVs del EDA
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── laluzqueopaca/
│       ├── data/
│       ├── evaluation/     # Métricas: AbsRel, RMSE, Chamfer
│       ├── models/
│       ├── training/
│       └── utils/
└── tests/
```

## Siguientes pasos

1. `01_eda_iluminacion.ipynb` — Análisis cuantitativo del gradiente de iluminación en los keyframes de SCARED.
2. Implementar el método de corrección de iluminación sobre los keyframes.
3. Correr MonoIIT sobre las imágenes crudas y corregidas; comparar AbsRel/RMSE.
4. Extender el análisis a los frames de `rgb.mp4` + `scene_points.tar.gz` para evaluación temporal.
5. Integrar Hamlyn y C3VD para validar la generalización del método.

## Referencias

- Allan, M., et al. (2021). Stereo Correspondence and Reconstruction of Endoscopic Data Challenge. *arXiv:2101.01133*.
- Recasens, D., et al. (2021). Endo-Depth-and-Motion: Reconstruction and Tracking in a Surgical Setting Using Depth Prediction. *IEEE RA-L*.
- Golhar, M., et al. (2025). C3VD: Colonoscopy 3D Video Dataset Enabling Robust Benchmarking. *arXiv:2206.01023*.
