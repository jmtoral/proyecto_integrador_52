# La luz que opaca

Deteccion y supresion de reflejos especulares en imagenes endoscopicas.

## Descripcion

Este repositorio esta preparado como base de trabajo para un proyecto de vision por computadora enfocado en uno de los artefactos mas criticos en endoscopia: los reflejos especulares. Estos reflejos saturan pixeles, ocultan textura anatomica relevante y degradan tanto la interpretacion visual del especialista como el desempeno de modelos de IA para tareas CAD.

El objetivo general del proyecto es construir y evaluar un pipeline reproducible para:

1. Detectar regiones con reflejos especulares en imagenes endoscopicas.
2. Explorar tecnicas para su supresion o reconstruccion.
3. Medir el impacto de esa mitigacion sobre tareas posteriores de analisis.

## Situacion actual del repositorio

Actualmente el repositorio se encuentra en etapa de arranque y exploracion. Ya cuenta con:

- estructura base de proyecto para investigacion en vision por computadora
- organizacion de datos, configuraciones, codigo fuente, notebooks, pruebas y reportes
- un baseline inicial muy simple para deteccion de reflejos por brillo y baja saturacion
- un primer notebook de exploracion para cargar hasta 10 imagenes y revisar posibles reflejos
- ambiente de trabajo pensado para evolucionar despues hacia tareas de deteccion, segmentacion o supresion

Por ahora, el metodo definitivo todavia no esta cerrado. La prioridad inmediata es entender las entradas, explorar imagenes reales, revisar artefactos y definir con mas claridad si el problema se abordara primero como deteccion, segmentacion o restauracion.

## Dataset inicial de trabajo

Mientras se prueba la estructura del repo y los primeros notebooks, las imagenes iniciales de referencia seran tomadas del dataset oficial Kvasir de Simula:

- Kvasir: [https://datasets.simula.no/kvasir/](https://datasets.simula.no/kvasir/)

Kvasir es un dataset publico de imagenes del tracto gastrointestinal para investigacion en deteccion asistida por computadora. En su sitio oficial se describe como un conjunto de imagenes endoscopicas clasificadas y anotadas por medicos, util para tareas de aprendizaje automatico, vision por computadora y evaluacion reproducible. Tambien se indica que su uso esta restringido a fines de investigacion y educacion.

En este proyecto se usara inicialmente como fuente de imagenes endoscopicas para:

- aprender a consumir y organizar imagenes reales
- explorar brillo, saturacion y regiones con reflejos intensos
- construir baselines tempranos
- preparar un flujo reproducible antes de definir el metodo final

Nota: Kvasir no es un dataset especifico de reflejos especulares anotados, por lo que al inicio funcionara principalmente como fuente de imagenes endoscopicas reales para exploracion, prototipos y posibles anotaciones propias.

## Problema

La literatura reciente reporta cuatro retos principales:

- Escasez de datos medicos etiquetados y restricciones de privacidad.
- Presencia de artefactos visuales severos, especialmente reflejos especulares.
- Brecha de dominio entre hospitales, equipos y modalidades como WLI y NBI.
- Restricciones de tiempo real para apoyo intraoperatorio.

Este repositorio esta organizado para permitir investigacion iterativa aunque el metodo final aun no este definido.

## Posibles lineas de trabajo

- Deteccion clasica por umbralado en espacios RGB, HSV o Lab.
- Segmentacion supervisada de reflejos.
- Inpainting clasico o guiado por mascaras.
- Restauracion con modelos encoder-decoder.
- Adaptacion de dominio entre modalidades o fuentes de datos.
- Evaluacion de impacto en tareas downstream.

## Estructura

```text
.
|-- artifacts/              # Salidas temporales de experimentos
|-- configs/                # Configuraciones versionables
|   |-- data/
|   |-- model/
|   `-- train/
|-- data/
|   |-- annotations/        # Mascaras, etiquetas, metadatos
|   |-- external/           # Datos de terceros
|   |-- interim/            # Datos limpios o transformados parcialmente
|   |-- processed/          # Datasets listos para modelado
|   `-- raw/                # Datos originales
|-- docs/                   # Notas metodologicas y documentacion
|-- notebooks/              # Exploracion y prototipos
|-- reports/
|   `-- figures/            # Figuras para entregables
|-- scripts/                # Scripts de apoyo y pipelines
|-- src/
|   `-- laluzqueopaca/
|       |-- data/           # Carga, validacion y preprocesamiento
|       |-- evaluation/     # Metricas y evaluacion
|       |-- models/         # Baselines y modelos
|       |-- training/       # Entrenamiento e inferencia
|       `-- utils/          # Utilidades compartidas
|-- tests/                  # Pruebas unitarias
|-- .gitignore
`-- pyproject.toml
```

## Flujo sugerido

1. Conseguir y documentar datasets candidatos.
2. Definir una tarea base de deteccion de reflejos.
3. Implementar un baseline sencillo y reproducible.
4. Agregar metricas de segmentacion y de restauracion.
5. Comparar enfoques clasicos y de aprendizaje profundo.

## Primeros siguientes pasos

- Documentar datasets potenciales en `docs/`.
- Crear un script de exploracion de brillo y saturacion.
- Definir un formato canonico para mascaras de reflejo.
- Elegir un baseline inicial: deteccion, supresion o ambos.
