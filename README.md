# <center> <font color="#0036a3">Maestría en Inteligencia Artificial Aplicada (MNA)</font> </center>

<center>

[![Institución](https://img.shields.io/badge/INSTITUCIÓN-TECNOLÓGICO_DE_MONTERREY-0036a3?style=for-the-badge)](https://tec.mx)
[![Programa](https://img.shields.io/badge/PROGRAMA-MNA-0036a3?style=for-the-badge)](https://tec.mx)
[![Materia](https://img.shields.io/badge/MATERIA-PROYECTO_INTEGRADOR-E0A800?style=for-the-badge&logoColor=white)](https://tec.mx)

</center>

<center>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/)
[![tifffile](https://img.shields.io/badge/tifffile-TIFF_float32-6a6a6a?style=flat-square)](https://github.com/cgohlke/tifffile)
[![GitHub](https://img.shields.io/badge/Repo-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/jmtoral/proyecto_integrador_52)
[![Sitio Web](https://img.shields.io/badge/Sitio_Web-GitHub_Pages-0036a3?style=flat-square&logo=github&logoColor=white)](https://jmtoral.github.io/proyecto_integrador_52/)

</center>

---

# La luz que opaca
### Detección y supresión de reflejos especulares en imágenes endoscópicas

**Materia:** Proyecto Integrador — TC5035.10  
**Sitio web:** [https://jmtoral.github.io/proyecto_integrador_52/](https://jmtoral.github.io/proyecto_integrador_52/)

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
      <img src="https://raw.githubusercontent.com/jmtoral/proyecto_integrador_52/main/reports/elda.jpg" width="140px" height="140px" style="border-radius:50%; object-fit:cover;">
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
      <img src="https://raw.githubusercontent.com/jmtoral/proyecto_integrador_52/main/reports/jmtc.jpg" width="160px" height="160px" style="border-radius:50%; object-fit:cover;">
      <br>
      <h4>José Manuel Toral Cruz</h4>
      <code>A01122243</code><br>
      <small>MNA Student</small>
    </td>
  </tr>
</table>

---

## Entregas

### Avance 1 — Análisis Exploratorio de Datos

El reporte ejecutivo del Avance 1 está disponible en [`reports/Avance1.52.pdf`](reports/Avance1.52.pdf). Consolida los hallazgos de los análisis exploratorios sobre los tres datasets y constituye la entrega formal para evaluación docente.

### Avance 2 — Ingeniería de Características

El notebook [`Avance2_52_Feature_Engineering.ipynb`](notebooks/Avance2_52_Feature_Engineering.ipynb) implementa el pipeline completo de ingeniería de características: extracción de features por keyframe (imagen + profundidad + video), transformaciones (cap a 150 mm, log(Z+1), normalización min-max), selección por correlación y varianza, y PCA sobre la nube de puntos 3D. Produce un dataframe de 20 filas × N features (4 datasets × 5 keyframes) como insumo para la fase de modelado.


## Descripción

Este repositorio contiene el Proyecto Integrador para la Maestría de Inteligencia Artifical Aplicada. EL objetivo del proyecto es evaluar tres modelos de reconstrucción de imágenes endoscópicas en 3D y evaluar su desempeño con tres datasets. 

Los datos empleados en esta investigación responden a una naturaleza semiestructurada y no tabular. A diferencia de los datasets tabulares convencionales donde cada observación es una fila y cada variable una columna, los datos endoscópicos se organizan como secuencias de video estéreo, mapas de profundidad tridimensionales, nubes de puntos y archivos de calibración de cámara que deben interpretarse conjuntamente..

El experimento es un diseño factorial: **3 modelos × 3 métodos de corrección + línea base sin corrección**, todo lo demás constante.

```
SCARED (keyframes RGB + GT TIFF)
        ↓
  corrección de iluminación   ← variable independiente
  sin corrección / CLAHE / Retinex / MonoIIT
        ↓
  modelo de profundidad   ← tres candidatos en paralelo
  NeXF · EndoGaussian · EndoDepthAndMotion
        ↓
  depth map estimado
        ↓
  evaluación vs. GT de luz estructurada
  AbsRel · RMSE · Chamfer Distance
```

La hipótesis es que eliminar el gradiente radial de iluminación (más brillante en el centro, oscuro en los bordes) reduce el error de profundidad, especialmente en la periferia. El criterio de ganador es el menor AbsRel en el split de test (datasets 8–9).

## Datasets

Se trabajan con tres bases de datos clínicas endoscópicas:

### SCARED

**SCARED** (Structured light endoscopY And Reconstructed Depth) ofrece mapas de profundidad reales capturados con luz estructurada sobre tejido porcino ex-vivo (Allan et al., 2021). Es el dataset principal del proyecto.

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




## Notebooks

### Avance 1 — Análisis Exploratorio de Datos

| Notebook | Dataset | Contenido |
|---|---|---|
| [`Avance1_Equipo52_00_exploracion_scared.ipynb`](notebooks/Avance1_Equipo52_00_exploracion_scared.ipynb) | SCARED | Estructura, formatos, dimensiones, rangos de profundidad |
| [`Avance1_Equipo52_01_eda_iluminacion.ipynb`](notebooks/Avance1_Equipo52_01_eda_iluminacion.ipynb) | SCARED | Gradiente radial, CV de luminancia, especulares, correlación luminancia-GT |
| [`Avance1_Equipo52_02_hamlyn_eda.ipynb`](notebooks/Avance1_Equipo52_02_hamlyn_eda.ipynb) | Hamlyn | Exploración de secuencias estéreo in-vivo |
| [`Avance2_Equipo_52_03_eda_calidad_input.ipynb`](notebooks/Avance2_Equipo_52_03_eda_calidad_input.ipynb) | SCARED | Calidad de video, flujo óptico, poses, cobertura GT en frames |
| [`Avance1_Equipo52_04_exploracion_C3VDv2.ipynb`](notebooks/Avance1_Equipo52_04_exploracion_C3VDv2.ipynb) | C3VD | Exploración del dataset de colonoscopía 3D |

### Avance 2 — Ingeniería de Características

| Notebook | Dataset | Contenido |
|---|---|---|
| [`Avance2_52_Feature_Engineering.ipynb`](notebooks/Avance2_52_Feature_Engineering.ipynb) | SCARED | Construcción de dataframe por keyframe, transformaciones (cap, log, min-max), selección y PCA geométrico |


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
│   ├── Avance1_Equipo52_00_exploracion_scared.ipynb
│   ├── Avance1_Equipo52_01_eda_iluminacion.ipynb
│   ├── Avance1_Equipo52_02_hamlyn_eda.ipynb
│   ├── Avance2_Equipo_52_03_eda_calidad_input.ipynb
│   └── Avance1_Equipo52_04_exploracion_C3VDv2.ipynb
├── outcomes/
│   └── eda_outputs/        # PNGs y CSVs del EDA
├── reports/
│   ├── figures/            # Imágenes para los reportes
│   ├── Avance1.52.pdf      # Reporte ejecutivo Avance 1 (entrega formal)
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

1. Implementar los métodos de corrección de iluminación: CLAHE, Retinex, MonoIIT.
2. Correr NeXF, EndoGaussian y EndoDepthAndMotion sobre imágenes crudas y corregidas.
3. Comparar AbsRel, RMSE, SSIM, PSNR y Chamfer Distance entre condiciones.
4. Integrar Hamlyn y C3VD para validar la generalización del método.

## Agradecimientos

El Equipo 52 agradece la orientación y asesoría técnica del **Dr. Gilberto Ochoa Ruiz** y del **Dr. Ricardo Espinosa Loera** (Tecnológico de Monterrey), cuyas contribuciones al campo de la visión por computadora y el procesamiento de imágenes médicas enriquecieron el diseño metodológico de este proyecto.

## Nota sobre uso de inteligencia artificial

El análisis de datos, el código, la interpretación de resultados y las decisiones metodológicas son trabajo del Equipo 52. Para la revisión de estilo en la redacción, la incorporación de buenas prácticas de código y la gestión del repositorio se utilizó [Claude](https://claude.ai) (Anthropic, modelo `claude-sonnet-4-6`) como asistente de productividad.

*Todo contenido generado con asistencia de IA fue revisado, validado y aprobado por lxs integrantes del equipo antes de incorporarse al repositorio. Lxs autorxs son responsables de la exactitud técnica y académica de este trabajo.*

## Referencias

- Allan, M., et al. (2021). Stereo Correspondence and Reconstruction of Endoscopic Data Challenge. *arXiv:2101.01133*.
- Recasens, D., et al. (2021). Endo-Depth-and-Motion: Reconstruction and Tracking in a Surgical Setting Using Depth Prediction. *IEEE RA-L*.
- Golhar, M., et al. (2025). C3VD: Colonoscopy 3D Video Dataset Enabling Robust Benchmarking. *arXiv:2206.01023*.
