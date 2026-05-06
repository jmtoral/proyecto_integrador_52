# La luz que opaca

Corrección de iluminación como preprocesamiento para estimación de profundidad monocular en endoscopía.

## Descripción

Este repositorio contiene el proyecto integrador de maestría, cuyo objetivo es evaluar si la corrección del gradiente de iluminación del endoscopio mejora el desempeño de modelos de estimación de profundidad monocular en video endoscópico.

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

## Dataset

Se trabaja con el **SCARED dataset** (Structured light endoscopY And Reconstructed Depth), específicamente `dataset_1`, disponible en HuggingFace.

| Archivo | Descripción |
|---|---|
| `Left_Image.png` | Frame de referencia RGB (1280×1024 px) |
| `left_depth_map.tiff` | Ground truth de profundidad por structured light |
| `endoscope_calibration.yaml` | Matriz intrínseca K de la cámara izquierda |
| `point_cloud.obj` | Nube de puntos 3D en bruto del sistema de captura |
| `data/rgb.mp4` | Video completo del keyframe |
| `data/scene_points.tar.gz` | Depth GT para todos los frames del video |

**5 keyframes** con coberturas de GT entre 71.9% (KF5) y 86.8% (KF3). Los píxeles sin GT corresponden a reflejos especulares y bordes de curvatura extrema donde el structured light no puede triangular.

Los datos crudos (`data/scared_raw/`) no están en el repositorio por su tamaño (13.4 GB).

## EDA

El notebook [`notebooks/scared_real_eda.ipynb`](notebooks/scared_real_eda.ipynb) contiene el análisis exploratorio completo del dataset:

1. Imágenes RGB — resolución, iluminación, balance de canales
2. Depth maps — cobertura por keyframe, rango near/far
3. Estadísticas de profundidad — percentiles P2/P98 para `poses_bounds.npy`
4. Validez espacial — patrón de viñeteado circular
5. Nube de puntos 3D interactiva (reproyección desde TIFF + K)
6. Superficies 3D de depth
7. Análisis de nube de puntos desde `.obj`
8. Estadísticas RGB y ranking de dificultad por keyframe

**Hallazgos principales del EDA:**

- El rango de profundidad varía por keyframe: P2/P98 entre [28.9, 58.6] mm (KF4) y [58.6, 109.6] mm (KF3).
- KF5 es el keyframe más difícil: mayor porcentaje de especulares (3.79%) y menor textura (varianza Laplaciana = 63).
- KF4 es el más fácil (score = 0.358). **Se recomienda iniciar experimentos con KF4.**
- El gradiente de iluminación es visible y medible: la luminancia cae del centro a los bordes de forma consistente en los 5 keyframes.
- Los archivos `.obj` tienen coordenadas NaN con el parser actual — se requiere inspección del formato antes de usarlos para Chamfer Distance.

Los outputs del EDA (PNGs y CSVs) están en [`outcomes/eda_outputs/`](outcomes/eda_outputs/).

## Variables del experimento

**Variable independiente:** imagen RGB con o sin corrección de iluminación  
**Variable dependiente:** error de estimación de profundidad (AbsRel, RMSE, Chamfer Distance)

La corrección de iluminación transforma `X_crudo → X_corrected` antes de pasarlo al modelo de profundidad. Todo lo demás (modelo, hiperparámetros, GT) permanece constante para aislar el efecto.

## Estructura

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
├── notebooks/              # EDA y experimentos
│   └── scared_real_eda.ipynb
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

1. Implementar el método de corrección de iluminación sobre los keyframes de SCARED.
2. Correr MonoIIT sobre las imágenes crudas y corregidas, comparar AbsRel/RMSE.
3. Inspeccionar el formato de los `.obj` para habilitar evaluación con Chamfer Distance.
4. Extender el análisis a los frames de `rgb.mp4` + `scene_points.tar.gz` para evaluación temporal.
