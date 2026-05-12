# <center> <font color="#0036a3">Maestría en Inteligencia Artificial Aplicada (MNA)</font> </center>

<center>

[![Institución](https://img.shields.io/badge/INSTITUCIÓN-TECNOLÓGICO_DE_MONTERREY-0036a3?style=for-the-badge)](https://tec.mx)
[![Programa](https://img.shields.io/badge/PROGRAMA-MNA-0036a3?style=for-the-badge)](https://tec.mx)
[![Materia](https://img.shields.io/badge/MATERIA-PROYECTO_INTEGRADOR-E0A800?style=for-the-badge&logoColor=white)](https://tec.mx)

</center>

<center>

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![GitHub](https://img.shields.io/badge/Repo-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/jmtoral/proyecto_integrador_52)

</center>

---

# La luz que opaca
### Detección y supresión de reflejos especulares en imágenes endoscópicas

**Materia:** Proyecto Integrador — TC5035.10

---

### Equipo Docente
* **Dra. Grettel Barceló Alonso** | Profesora Titular
* **Dr. Luis Eduardo Falcón Morales** | Profesor Titular
* **Mtra. Verónica Sandra Guzmán de Valle** | Profesora Asistente

**Patrocinador:** Dr. Gerardo Camacho

---

## Equipo 52

<table style="width:100%; border:none; border-collapse:collapse;">
  <tr>
    <td align="center" style="border:none; width:33%; padding:10px;">
      <img src="https://ui-avatars.com/api/?name=Elda+Morales&size=140&background=0036a3&color=fff&rounded=true" width="140px" style="border-radius:10px;">
      <br>
      <h4>Elda Cristina Morales Sánchez de la Barquera</h4>
      <code>A00449074</code><br>
      <small>MNA Student</small>
    </td>
    <td align="center" style="border:none; width:33%; padding:10px;">
      <img src="https://ui-avatars.com/api/?name=Maria+Gutierrez&size=140&background=0036a3&color=fff&rounded=true" width="140px" style="border-radius:10px;">
      <br>
      <h4>María Paula Gutiérrez Cervantes</h4>
      <code>A01747706</code><br>
      <small>MNA Student</small>
    </td>
    <td align="center" style="border:none; width:33%; padding:10px;">
      <img src="https://ui-avatars.com/api/?name=Jose+Toral&size=140&background=0036a3&color=fff&rounded=true" width="140px" style="border-radius:10px;">
      <br>
      <h4>José Manuel Toral Cruz</h4>
      <code>A01122243</code><br>
      <small>MNA Student</small>
    </td>
  </tr>
</table>

---

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

| Notebook | Dataset | Contenido |
|---|---|---|
| [`00_exploracion_scared.ipynb`](notebooks/00_exploracion_scared.ipynb) | SCARED | Estructura, formatos, dimensiones, rangos de profundidad |
| [`01_eda_iluminacion.ipynb`](notebooks/01_eda_iluminacion.ipynb) | SCARED | Gradiente radial, CV de luminancia, especulares, correlación luminancia-GT |
| [`02_hamlyn_eda.ipynb`](notebooks/02_hamlyn_eda.ipynb) | Hamlyn | Exploración de secuencias estéreo in-vivo |
| [`03_eda_calidad_input.ipynb`](notebooks/03_eda_calidad_input.ipynb) | SCARED | Calidad de video, flujo óptico, poses, cobertura GT en frames |

**Hallazgos principales del EDA de iluminación (`01_eda_iluminacion`):**
- El gradiente radial afecta todos los keyframes: el centro es entre 1.3x y 3.3x más brillante que el borde.
- El gradiente no es simétrico: el pico de luminancia coincide con el centro óptico (cx=597, cy=520), no con el centro geométrico (640, 512).
- Existe correlación negativa fuerte entre luminancia y cobertura de GT (r hasta -0.86): las zonas más brillantes son las que menos datos de entrenamiento tienen.
- El keyframe más problemático es KF5 (CV de luminancia=0.665, GT válido=71.9%).

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
│   ├── 00_exploracion_scared.ipynb
│   ├── 01_eda_iluminacion.ipynb
│   ├── 02_hamlyn_eda.ipynb
│   └── 03_eda_calidad_input.ipynb
├── outcomes/
│   └── eda_outputs/        # PNGs y CSVs del EDA
├── reports/
│   ├── figures/            # Imágenes para los reportes
│   ├── 00_reporte_exploracion_scared.md
│   └── 01_reporte_eda_iluminacion.md
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

1. Implementar los métodos de corrección de iluminación: CLAHE, Retinex, MonoIIF.
2. Correr NeXF, EndoGaussian y EndoDepthAndMotion sobre imágenes crudas y corregidas.
3. Comparar AbsRel, RMSE, SSIM, PSNR y Chamfer Distance entre condiciones.
4. Extender el análisis a los frames de `rgb.mp4` + `scene_points.tar.gz` para evaluación temporal.
5. Integrar Hamlyn y C3VD para validar la generalización del método.

## Referencias

- Allan, M., et al. (2021). Stereo Correspondence and Reconstruction of Endoscopic Data Challenge. *arXiv:2101.01133*.
- Recasens, D., et al. (2021). Endo-Depth-and-Motion: Reconstruction and Tracking in a Surgical Setting Using Depth Prediction. *IEEE RA-L*.
- Golhar, M., et al. (2025). C3VD: Colonoscopy 3D Video Dataset Enabling Robust Benchmarking. *arXiv:2206.01023*.
