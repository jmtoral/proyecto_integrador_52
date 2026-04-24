# La luz que opaca

Detección y supresión de reflejos especulares en imágenes endoscópicas.

## Descripción

Este repositorio está preparado como base de trabajo para un proyecto de visión por computadora enfocado en uno de los artefactos más críticos en endoscopía: los reflejos especulares. Estos reflejos saturan píxeles, ocultan textura anatómica relevante y degradan tanto la interpretación visual del especialista como el desempeño de modelos de IA para tareas CAD.

El objetivo general del proyecto es construir y evaluar un pipeline reproducible para:

1. Detectar regiones con reflejos especulares en imágenes endoscópicas.
2. Explorar técnicas para su supresión o reconstrucción.
3. Medir el impacto de esa mitigación sobre tareas posteriores de análisis.

## Situación actual del repositorio

Actualmente el repositorio se encuentra en etapa de arranque y exploración. Ya cuenta con:

- Estructura base de proyecto para investigación en visión por computadora.
- Organización de datos, configuraciones, código fuente, notebooks, pruebas y reportes.
- Un baseline inicial muy simple para detección de reflejos por brillo y baja saturación.
- Un primer notebook de exploración para cargar hasta 10 imágenes y revisar posibles reflejos.
- Ambiente de trabajo pensado para evolucionar después hacia tareas de detección, segmentación o supresión.

Por ahora, el método definitivo todavía no está cerrado. La prioridad inmediata es entender las entradas, explorar imágenes reales, revisar artefactos y definir con más claridad si el problema se abordará primero como detección, segmentación o restauración.

## Dataset inicial de trabajo

Mientras se prueba la estructura del repositorio y los primeros notebooks, las imágenes iniciales de referencia serán tomadas del dataset oficial Kvasir de Simula:

- Kvasir: [https://datasets.simula.no/kvasir/](https://datasets.simula.no/kvasir/)

Kvasir es un dataset público de imágenes del tracto gastrointestinal para investigación en detección asistida por computadora. En su sitio oficial se describe como un conjunto de imágenes endoscópicas clasificadas y anotadas por médicos, útil para tareas de aprendizaje automático, visión por computadora y evaluación reproducible. También se indica que su uso está restringido a fines de investigación y educación.

En este proyecto se usará inicialmente como fuente de imágenes endoscópicas para:

- Aprender a consumir y organizar imágenes reales.
- Explorar brillo, saturación y regiones con reflejos intensos.
- Construir baselines tempranos.
- Preparar un flujo reproducible antes de definir el método final.

Nota: Kvasir no es un dataset específico de reflejos especulares anotados, por lo que al inicio funcionará principalmente como fuente de imágenes endoscópicas reales para exploración, prototipos y posibles anotaciones propias.

## Problema

La literatura reciente reporta cuatro retos principales:

- Escasez de datos médicos etiquetados y restricciones de privacidad.
- Presencia de artefactos visuales severos, especialmente reflejos especulares.
- Brecha de dominio entre hospitales, equipos y modalidades como WLI y NBI.
- Restricciones de tiempo real para apoyo intraoperatorio.

Este repositorio está organizado para permitir investigación iterativa aunque el método final aún no esté definido.

## Posibles líneas de trabajo

- Detección clásica por umbralado en espacios RGB, HSV o Lab.
- Segmentación supervisada de reflejos.
- Inpainting clásico o guiado por máscaras.
- Restauración con modelos encoder-decoder.
- Adaptación de dominio entre modalidades o fuentes de datos.
- Evaluación de impacto en tareas downstream.

## Estructura

```text
.
|-- artifacts/              # Salidas temporales de experimentos
|-- configs/                # Configuraciones versionables
|   |-- data/
|   |-- model/
|   `-- train/
|-- data/
|   |-- annotations/        # Máscaras, etiquetas, metadatos
|   |-- external/           # Datos de terceros
|   |-- interim/            # Datos limpios o transformados parcialmente
|   |-- processed/          # Datasets listos para modelado
|   `-- raw/                # Datos originales
|-- docs/                   # Notas metodológicas y documentación
|-- notebooks/              # Exploración y prototipos
|-- reports/
|   `-- figures/            # Figuras para entregables
|-- scripts/                # Scripts de apoyo y pipelines
|-- src/
|   `-- laluzqueopaca/
|       |-- data/           # Carga, validación y preprocesamiento
|       |-- evaluation/     # Métricas y evaluación
|       |-- models/         # Baselines y modelos
|       |-- training/       # Entrenamiento e inferencia
|       `-- utils/          # Utilidades compartidas
|-- tests/                  # Pruebas unitarias
|-- .gitignore
`-- pyproject.toml
```

## Flujo sugerido

1. Conseguir y documentar datasets candidatos.
2. Definir una tarea base de detección de reflejos.
3. Implementar un baseline sencillo y reproducible.
4. Agregar métricas de segmentación y de restauración.
5. Comparar enfoques clásicos y de aprendizaje profundo.

## Primeros siguientes pasos

- Documentar datasets potenciales en `docs/`.
- Crear un script de exploración de brillo y saturación.
- Definir un formato canónico para máscaras de reflejo.
- Elegir un baseline inicial: detección, supresión o ambos.
