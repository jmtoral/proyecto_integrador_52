# Reporte de Exploración: Dataset SCARED

**Notebook:** `notebooks/00_exploracion_scared.ipynb`  
**Fecha de ejecución:** 2026-08-05  
**Autor:** Proyecto Integrador — Corrección de Iluminación en Endoscopía

---

## 1. Objetivo

Caracterizar la estructura física y el contenido del dataset SCARED (*Stereo Correspondence and Reconstruction of Endoscopic Data*, MICCAI 2019) sin descomprimir los archivos al disco. El análisis responde tres preguntas básicas:

1. ¿Qué hay en cada ZIP? ¿Cuántos keyframes y qué archivos contiene cada uno?
2. ¿Cómo se ve cada tipo de archivo? (PNG, TIFF, OBJ, YAML, MP4, TAR.GZ)
3. ¿Qué dimensiones y rangos de valores tienen los datos más importantes?

---

## 2. Volumetría del dataset

### 2.1 ZIPs encontrados

| Dataset | Keyframes | ZIP (GB) |
|---|---|---|
| dataset_1 | keyframe_1…keyframe_5 | 13.4 |
| dataset_2 | keyframe_1…keyframe_5 | 24.0 |
| dataset_3 | keyframe_1…keyframe_5 | 29.0 |
| dataset_4 | keyframe_1…keyframe_5 | 20.1 |
| dataset_5 | keyframe_1…keyframe_5 | 30.8 |
| dataset_6 | keyframe_1…keyframe_5 | 39.5 |
| dataset_7 | keyframe_1…keyframe_5 | 21.4 |
| dataset_8 | keyframe_0…keyframe_4 | 34.0 |
| dataset_9 | keyframe_0…keyframe_4 | 17.3 |
| **TOTAL** | **45 keyframes** | **229 GB** |

**Hallazgo:** `dataset_8` y `dataset_9` indexan sus keyframes desde 0 (`keyframe_0…keyframe_4`) en lugar de desde 1. Esto es inconsistente con los datasets 1–7 y debe manejarse programáticamente — no se deben hardcodear los nombres de keyframe.

### 2.2 Archivos por keyframe

Inventario de `dataset_1/keyframe_1`, representativo de todos los keyframes:

| Archivo | Tamaño |
|---|---|
| `Left_Image.png` | 2.5 MB |
| `Right_Image.png` | 2.5 MB |
| `left_depth_map.tiff` | 15.7 MB |
| `right_depth_map.tiff` | 15.7 MB |
| `point_cloud.obj` | 63.6 MB |
| `endoscope_calibration.yaml` | ~0 MB |
| `data/rgb.mp4` | 33.0 MB |
| `data/frame_data.tar.gz` | ~0 MB |
| `data/scene_points.tar.gz` | 2 611.9 MB |

**Excepción documentada:** `keyframe_4` de `dataset_1` no tiene carpeta `data/` por un error de logging durante la captura original. Los cinco archivos estáticos (imágenes, TIFFs, OBJ, YAML) sí están presentes.

---

## 3. Imágenes RGB (`Left_Image.png` / `Right_Image.png`)

- **Forma del array:** `(1024, 1280, 3)` → (alto, ancho, canales)
- **Tipo de dato:** `uint8` — valores enteros en el rango [0, 255]
- **Resolución:** 1280×1024 px
- **Rango de valores observado:** [0, 255] (rango completo utilizado)

Las imágenes muestran tejido digestivo porcino iluminado por el endoscopio. El par estéreo (izquierda/derecha) tiene un desplazamiento lateral visible entre las dos cámaras, que es el principio que permite calcular profundidad por disparidad. El gradiente de iluminación no uniforme — más brillante en el centro, más oscuro en los bordes — es claramente visible, y constituye el problema central que este proyecto busca corregir.

---

## 4. Mapa de profundidad GT (`left_depth_map.tiff`)

### Estructura del TIFF

El archivo no es una imagen convencional: cada píxel almacena una tripleta `(X, Y, Z)` de coordenadas 3D en milímetros. Para el proyecto se utiliza únicamente el canal Z (profundidad).

- **Forma del array:** `(1024, 1280, 3)` → (alto, ancho, canales XYZ)
- **Tipo de dato:** `float32`
- **Unidades:** milímetros

### Rangos de coordenadas (`dataset_1/keyframe_1`)

| Coordenada | Mínimo | Máximo |
|---|---|---|
| X | −72.5 mm | 80.9 mm |
| Y | −54.6 mm | 61.6 mm |
| Z (profundidad) | 35.1 mm | 176.3 mm |

### Cobertura de Ground Truth

| Métrica | Valor |
|---|---|
| Píxeles válidos (Z > 0) | 1 027 266 de 1 310 720 |
| Cobertura GT | **78.4%** |
| Profundidad mínima | 35.1 mm |
| Profundidad mediana | 58.6 mm |
| Profundidad máxima | 176.3 mm |

El 21.6% restante de píxeles carece de Ground Truth. Estos píxeles tienen valor `(0, 0, 0)` en el TIFF y corresponden principalmente a **reflejos especulares** (zonas donde el structured light se refleja sin poder triangular) y **bordes de curvatura extrema** (zonas donde el ángulo de incidencia es demasiado oblicuo). Son la fuente principal de datos faltantes en el dataset.

**Distribución de profundidades:** Unimodal con moda aproximada en 55–60 mm. La cola derecha (profundidades > 100 mm) corresponde a zonas periféricas de la imagen donde el endoscopio está más alejado del tejido.

---

## 5. Calibración de cámara (`endoscope_calibration.yaml`)

El archivo usa el formato de matrices OpenCV (`!!opencv-matrix`) y contiene los parámetros intrínsecos de ambas cámaras. Los valores recuperados para la cámara izquierda de `dataset_1/keyframe_1` son:

**Matriz intrínseca K (cámara izquierda — M1):**

```
[[1035.3,    0.0,  597.0],
 [   0.0, 1035.1,  520.4],
 [   0.0,    0.0,    1.0]]
```

| Parámetro | Valor | Interpretación |
|---|---|---|
| `fx` | 1035.3 px | Longitud focal horizontal |
| `fy` | 1035.1 px | Longitud focal vertical |
| `cx` | 597.0 px | Centro óptico, eje X |
| `cy` | 520.4 px | Centro óptico, eje Y |

El centro óptico real (597, 520) está desplazado ~43 px respecto al centro geométrico del sensor (640, 512), lo que es esperable en una óptica endoscópica. El README del dataset advierte que esta calibración es **aproximada** y que los modelos deben ser robustos a pequeños errores en K.

El YAML también contiene la matriz M2 (cámara derecha), la rotación R y traslación T entre cámaras, y los vectores de distorsión D1/D2.

---

## 6. Nube de puntos 3D (`point_cloud.obj`)

El archivo Wavefront OBJ contiene únicamente vértices (sin caras ni mallas):

- **Total de vértices:** 1 310 720
- **Caras:** 0 (solo nube de puntos, no malla triangulada)
- **Formato:** líneas `v x y z` con coordenadas float64 de alta precisión

**Rangos de la nube (muestra de 8 000 puntos de `dataset_1/keyframe_1`):**

| Coordenada | Mínimo | Máximo |
|---|---|---|
| X | −35.3 mm | 62.6 mm |
| Y | −49.7 mm | 31.7 mm |
| Z | 49.7 mm | 115.4 mm |

Las proyecciones 2D confirman la forma convexa del tejido: vista lateral (XZ) muestra una distribución en bandas horizontales coherente con la curvatura del órgano; vista frontal (XY) muestra la forma ovalada del campo de visión del endoscopio.

**Observación técnica:** Los rangos Z del OBJ (49.7–115.4 mm) son más estrechos que los del TIFF (35.1–176.3 mm). Esto es esperable: el TIFF incluye todos los píxeles válidos (incluyendo los más periféricos y alejados), mientras que la nube OBJ corresponde únicamente a los puntos que el proyector de luz estructurada pudo triangular con mayor confianza.

---

## 7. Poses de cámara (`data/frame_data.tar.gz`)

El TAR contiene **197 archivos JSON** nombrados `frame_data000000.json` … `frame_data000196.json`, uno por cuadro del video `rgb.mp4`.

**Estructura de cada JSON:**

```json
{
  "camera-calibration": {
    "DL": [...],   // distorsión cámara izquierda (5 coeficientes)
    "DR": [...],   // distorsión cámara derecha
    "KL": [[fx,0,cx],[0,fy,cy],[0,0,1]],  // matriz K izquierda
    "KR": [[fx,0,cx],[0,fy,cy],[0,0,1]],  // matriz K derecha
    "R":  [...],   // rotación entre cámaras
    ...
  },
  ...  // transformación de pose (no mostrada en el extracto)
}
```

El JSON del frame 0 confirma los mismos parámetros de calibración que el YAML del keyframe (`fx ≈ 1035`, `cx ≈ 597`, `cy ≈ 520`), con ligeras variaciones entre frames que reflejan la calibración dinámica del sistema.

---

## 8. Depths warpeados (`data/scene_points.tar.gz`)

El TAR contiene **197 archivos TIFF** nombrados `scene_points000000.tiff` … `scene_points000196.tiff`.

| Métrica | Valor |
|---|---|
| Frames en el TAR | 197 |
| Tamaño típico por TIFF | 31.5 MB |
| Forma del TIFF | `(2048, 1280, 3)` |
| Tipo | `float32` |
| Cobertura válida (frame 0) | **59.5%** |

**Observaciones importantes:**
- La resolución vertical del TIFF warpeado es **2048 px** (no 1024), porque contiene los frames de ambas cámaras apilados verticalmente — consistente con el formato del video `rgb.mp4`.
- El 40.5% de píxeles sin cobertura en el frame 0 corresponde a zonas donde la cámara se desplazó fuera del área que el keyframe cubre.
- El README advierte que el warping introduce pequeños errores por cinemática o desincronización. **No usar como GT para métricas finales.**

---

## 9. Visualización RGB + Depth (overlay)

El overlay de `Left_Image.png` con `left_depth_map.tiff` (dataset_1/keyframe_1) revela el patrón espacial de los datos faltantes:

- Las zonas sin GT (transparentes en el overlay) se concentran en **manchas brillantes** en el centro-derecha de la imagen — reflejos especulares donde la superficie del tejido actúa como espejo para el proyector de luz estructurada.
- El mapa de profundidad muestra una transición suave de valores bajos (cercano, ~50 mm, zona central) a valores altos (lejano, ~80+ mm, periferia), coherente con la curvatura convexa del tejido.
- La zona de profundidad máxima coincide con la zona de menor luminancia en el RGB, consistente con el gradiente de iluminación radial del endoscopio.

---

## 10. Resumen ejecutivo

| Dimensión | Resultado |
|---|---|
| Datasets disponibles | 9 ZIPs (229 GB en disco) |
| Keyframes totales | 45 |
| Resolución de imágenes | 1280×1024 px |
| Tipo de dato RGB | uint8, rango [0, 255] |
| Tipo de dato depth | float32, coordenadas XYZ en mm |
| Rango de profundidad | 35–176 mm (varía por keyframe) |
| Cobertura GT (depth map) | ~78% de píxeles (keyframe_1/dataset_1) |
| Datos faltantes | ~22% — reflejos especulares y bordes |
| Cobertura GT (scene_points warpeados) | ~60% (disminuye por desplazamiento de cámara) |
| Frames de video por keyframe | 197 (aprox.) |
| Lectura en memoria | Posible directamente desde ZIP con `io.BytesIO` |

### Hallazgos relevantes para el proyecto

1. **El gradiente de iluminación es medible y consistente.** El overlay RGB+Depth confirma que las zonas más brillantes coinciden con las más cercanas, y las zonas más oscuras (bordes del campo de visión) con las más lejanas. Esto implica que el modelo de profundidad puede estar confundiendo señal de iluminación con señal de profundidad.

2. **El 22% de datos faltantes tiene un patrón espacial no aleatorio.** Los reflejos especulares se concentran en zonas de alta luminancia y curvatura favorable para la especularidad, no en posiciones aleatorias. Cualquier método de evaluación debe excluir explícitamente estas zonas o tratarlas por separado.

3. **La calibración tiene desviación del centro geométrico.** El centro óptico real (~597, 520) se desplaza ~43 px del centro geométrico (640, 512). Este desplazamiento puede amplificar el gradiente radial de iluminación hacia el lado izquierdo de la imagen.

4. **Los datos de video (`scene_points.tar.gz`) tienen resolución 2048×1280** — el doble en altura — porque contienen el par estéreo apilado. Al usarlos hay que partir el TIFF por la mitad para obtener la vista izquierda.

5. **`dataset_8` y `dataset_9` rompen la convención de nombres.** Usar siempre `catalog[ds]['keyframes']` en lugar de listas hardcodeadas.

---

## 11. Siguientes pasos

- **`01_eda_iluminacion.ipynb`** — Cuantificar el gradiente de iluminación radial: perfiles de luminancia, ajuste de modelo gaussiano, mapa de corrección estimado.
- **`02_modelos_baseline.ipynb`** — Primeros experimentos de corrección: CLAHE, homomorphic filtering, retinex. Evaluar impacto en AbsRel y RMSE contra el GT de `left_depth_map.tiff`.
