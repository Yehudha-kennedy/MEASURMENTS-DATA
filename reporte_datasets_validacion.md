# Reporte: Datasets de Validación Externa para Proyecto de Clasificación S11

**Fecha:** 24 de junio de 2026
**Elaborado por:** Equipo de Ciencia de Datos
**Dirigido a:** Socio de proyecto
**Asunto:** Fuentes de datos Open Access identificadas para validación cruzada del modelo de clasificación de sensor de microondas (dry / wet / ice)

---

## 1. Contexto del Proyecto

Estamos desarrollando un modelo de Machine Learning que clasifica el estado de un sensor de microondas en tres categorías — **seco (dry)**, **mojado (wet)** y **hielo (ice)** — a partir de mediciones del parámetro de dispersión S11 (coeficiente de reflexión) en la banda de 8–12 GHz.

Nuestro dataset actual consta de **1,603 trazas** en formato Touchstone `.s1p`, cada una con 4,002 puntos de frecuencia que registran magnitud (dB) y fase (°) del parámetro S11.

| Clase | Muestras | Proporción |
|:---|:---:|:---:|
| Dry sensor | 601 | 37.5% |
| Ice on sensor | 501 | 31.2% |
| Wet sensor | 501 | 31.2% |

Para fortalecer la robustez y credibilidad científica del modelo, necesitamos **validar externamente** que nuestro pipeline de procesamiento y nuestras conclusiones son generalizables. Se identificaron tres fuentes de datos Open Access que aportan valor en tres ejes complementarios de validación.

---

## 2. Dataset 1 — EMI Shielding S-Parameter Data

| Campo | Detalle |
|:---|:---|
| **Plataforma** | Mendeley Data |
| **DOI** | `10.17632/whcnz7cpbx.1` |
| **Enlace** | https://data.mendeley.com/datasets/whcnz7cpbx/1 |
| **Formato** | Archivos Touchstone `.s2p` (parámetros S medidos con VNA) |
| **Licencia** | CC BY 4.0 (uso libre con atribución) |

### ¿Qué es?

Contiene mediciones reales de parámetros S (S11, S21, S12, S22) obtenidas mediante un Analizador de Redes Vectorial (VNA) sobre materiales nanocompuestos utilizados para blindaje electromagnético. Los archivos están en formato Touchstone, el mismo estándar industrial que utilizamos en nuestro proyecto.

### ¿Por qué nos sirve?

Este dataset valida la **integridad de nuestro pipeline de procesamiento de datos**. Al procesar archivos Touchstone generados por un VNA distinto al nuestro, con materiales y condiciones diferentes, podemos confirmar que:

- Nuestro parser de archivos `.s1p` / `.s2p` funciona correctamente con datos de otras fuentes
- Nuestro proceso de extracción de features (magnitud, fase, picos de resonancia) es robusto y no está sobreajustado a las particularidades de nuestro equipo
- Los datos de S11 extraídos de archivos `.s2p` externos mantienen la coherencia esperada

### ¿Qué podemos hacer con él?

1. **Test de robustez del parser**: Alimentar nuestro código de lectura de Touchstone con estos archivos `.s2p` y verificar que no hay errores de formato, encoding o columnas
2. **Benchmarking de features**: Extraer las mismas features que usamos en nuestro modelo (magnitud promedio, desviación estándar, pendiente de fase, etc.) y verificar que los rangos numéricos son coherentes
3. **Prueba de estrés**: Si nuestro pipeline procesa sin errores datos de un equipo y materiales completamente distintos, tenemos evidencia de que es generalizable

> **Valor para el proyecto:** Garantiza que nuestro código de procesamiento no tiene dependencias ocultas del equipo de medición. Esta es una verificación de ingeniería fundamental antes de publicar resultados.

---

## 3. Dataset 2 — VNA S-Parameter Brain Phantom Data

| Campo | Detalle |
|:---|:---|
| **Plataforma** | Mendeley Data |
| **DOI** | `10.17632/y8syxphvnr.1` |
| **Enlace** | https://data.mendeley.com/datasets/y8syxphvnr/1 |
| **Formato** | Datos de parámetros S (S11, S21, S12, S22) medidos con VNA |
| **Licencia** | CC BY 4.0 |

### ¿Qué es?

Datos experimentales de mediciones de microondas sobre phantoms de cerebro de cordero que simulan diferentes grados de atrofia cerebral. Cada condición experimental representa un nivel distinto de contenido de agua/material dentro del phantom, lo que genera respuestas electromagnéticas distintas.

### ¿Por qué nos sirve?

Este es el dataset más relevante de los tres porque aborda un **problema de clasificación fundamentalmente análogo al nuestro**: distinguir estados de un material basándose en cambios de constante dieléctrica medidos con S11 y un VNA.

Las similitudes clave son:

| Aspecto | Nuestro proyecto | Este dataset |
|:---|:---|:---|
| **Instrumento** | VNA | VNA |
| **Parámetro clave** | S11 (reflexión) | S11 (reflexión) |
| **Variable que cambia** | Contenido de agua/hielo en sensor | Contenido de agua en phantom |
| **Tarea** | Clasificación multi-clase | Clasificación multi-clase |
| **Principio físico** | Cambio en ε_r altera S11 | Cambio en ε_r altera S11 |

### ¿Qué podemos hacer con él?

1. **Validación cruzada del enfoque**: Aplicar nuestro mismo modelo (arquitectura + features) a estos datos y verificar si logra discriminar entre las distintas condiciones experimentales. Si sí, es evidencia de que nuestro enfoque metodológico es sólido y transferible
2. **Comparación de métricas**: Reportar accuracy/F1-score en nuestros datos vs en este dataset externo. Una performance razonable en ambos demuestra generalización
3. **Soporte para publicación**: Poder citar que nuestro modelo fue validado con datos experimentales de un tercer grupo de investigación independiente fortalece significativamente cualquier paper o reporte técnico

> **Valor para el proyecto:** Proporciona la evidencia más fuerte de que nuestro enfoque de clasificación basado en S11 no es un artefacto de nuestro setup experimental particular, sino que captura un fenómeno físico real y generalizable.

---

## 4. Dataset 3 — Soil Dielectric Permittivity Sensor

| Campo | Detalle |
|:---|:---|
| **Plataforma** | Mendeley Data |
| **DOI** | `10.17632/mt24hdxsr2.1` |
| **Enlace** | https://data.mendeley.com/datasets/mt24hdxsr2/1 |
| **Formato** | Datos tabulares de mediciones de sensor microondas TDT |
| **Licencia** | CC BY 4.0 |

### ¿Qué es?

Datos de un sensor microondas de dominio temporal (TDT) diseñado para medir propiedades dieléctricas de suelos superficiales. El dataset incluye mediciones de permitividad bajo diferentes niveles de humedad del suelo, cubriendo un rango desde suelo seco hasta suelo saturado de agua.

### ¿Por qué nos sirve?

Este dataset valida el **fundamento físico** de nuestro proyecto. Nuestro sensor clasifica tres estados que tienen constantes dieléctricas radicalmente distintas:

| Material | Constante dieléctrica (ε_r) |
|:---|:---:|
| Aire (dry sensor) | ~1.0 |
| Hielo (ice on sensor) | ~3.15 |
| Agua (wet sensor) | ~80.0 |

El dataset de suelo captura exactamente esta transición — un material que pasa de seco (baja ε_r) a saturado de agua (alta ε_r) — y registra cómo el sensor de microondas responde a esos cambios. Es una validación independiente de que el principio físico en el que se basa nuestro proyecto es correcto y medible.

### ¿Qué podemos hacer con él?

1. **Correlación física**: Graficar la respuesta del sensor de suelo vs contenido de humedad y superponer las regiones correspondientes a nuestras tres clases (dry ≈ ε_r=1, ice ≈ ε_r=3.15, wet ≈ ε_r=80). Esto muestra que la separación entre clases tiene una base física sólida
2. **Argumentación teórica**: Incluir en reportes y presentaciones una figura que demuestre que sensores independientes confirman que la constante dieléctrica cambia de forma medible entre aire, hielo y agua
3. **Definición de umbrales**: Usar los datos de permitividad de este dataset para definir rangos teóricos esperados de S11 para cada clase y comparar con nuestros valores reales

> **Valor para el proyecto:** Ancla nuestro modelo en la física electromagnética. Si alguien cuestiona "¿por qué un sensor de microondas puede distinguir hielo de agua?", este dataset es la respuesta cuantitativa.

---

## 5. Resumen de Estrategia de Validación

Los tres datasets cubren **ejes complementarios** de validación:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Dataset 1 (EMI Shielding)                             │
│   → Valida el PIPELINE de procesamiento                 │
│   → "¿Nuestro código funciona con datos externos?"      │
│                                                         │
│   Dataset 2 (Brain Phantom)                             │
│   → Valida el MODELO de clasificación                   │
│   → "¿Nuestro enfoque ML funciona en otro contexto?"    │
│                                                         │
│   Dataset 3 (Soil Permittivity)                         │
│   → Valida la FÍSICA del proyecto                       │
│   → "¿El principio de detección es correcto?"           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Eje de validación | Dataset | Pregunta que responde | Resultado esperado |
|:---|:---|:---|:---|
| **Ingeniería** | EMI Shielding | ¿El pipeline es robusto? | Parser procesa .s2p externos sin errores |
| **Metodológico** | Brain Phantom | ¿El modelo generaliza? | Accuracy >70% en clasificación externa |
| **Científico** | Soil Permittivity | ¿La física es correcta? | Tendencias de ε_r coinciden con nuestras clases |

---

## 6. Próximos Pasos

- [ ] Descargar los 3 datasets desde Mendeley Data (acceso libre, solo requiere cuenta gratuita)
- [ ] Ejecutar el pipeline de lectura Touchstone sobre Dataset 1 como prueba de integración
- [ ] Aplicar el modelo entrenado al Dataset 2 y reportar métricas comparativas
- [ ] Generar figura comparativa de ε_r vs respuesta del sensor con Dataset 3
- [ ] Documentar resultados en sección de "Validación Externa" del reporte final

---

*Todos los datasets listados son de acceso abierto (Open Access) bajo licencia CC BY 4.0 y pueden utilizarse libremente con la debida atribución en publicaciones y reportes técnicos.*
