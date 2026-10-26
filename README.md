# Repositorio de trabajo de título RS

Aquí se encuentran los archivos referenciados en el trabajo de título RS.

---

## Índice

| Directorio | Contenido |
|---|---|
| `fastsim/` | Generación de rutas y datasets sintéticos para el modelo predictivo de consumo energético (paquete `trailer_sim`, sobre FASTSim 3.1.0) |
| `util/` | Herramientas de utilidad general (previsualización de CSV, etc.) |

Cada subdirectorio tiene su propio `README.md` con el detalle de uso, configuración
y decisiones de diseño. El de `fastsim/` es la referencia principal: documenta la
física de los trailers, la mezcla de clases de vehículos, los niveles de fidelidad,
el ruido del dataset y las limitaciones conocidas del modelo.

---

## Requisitos

- Python ≥ 3.11
- `fastsim` 3.1.0 (se instala como dependencia; no se requiere GPU ni conexión para simular)
- `pandas` solo para `util/csv_preview.py`

---

## Puesta en marcha

```bash
cd fastsim
python -m venv .venv
.venv/bin/pip install -e .[dev]      # instalación editable + pytest/hypothesis
```

Verificación:

```bash
.venv/bin/python -m pytest -q        # 78 tests
.venv/bin/trailer-sim --help
```

---

## Uso

Generar un dataset (los resultados quedan en `../datasets/<run_name>/`):

```bash
# prueba básica de funcionamiento (200 rutas)
.venv/bin/python -m trailer_sim generate --config configs/smoke.yaml

# simulación de desarrollo (1.500 rutas)
.venv/bin/python -m trailer_sim generate --config configs/dev.yaml
```

Otros subcomandos útiles:

| Comando | Efecto |
|---|---|
| `trailer-sim show-config` | imprime la configuración efectiva de una corrida |
| `trailer-sim pin-vehicles` | regenera la plantilla BEV de FASTSim |
| `python3 util/csv_preview.py <archivo.csv>` | previsualiza un CSV como tabla en la terminal |

---

## Datos generados

El directorio `datasets/` no se versiona (ver `.gitignore`): se reconstruye con
`trailer-sim generate` a partir de una semilla, de forma reproducible. Cada corrida
produce:

| Archivo | Contenido |
|---|---|
| `train.csv`, `val.csv` | formato largo, una fila por segmento de ruta; split por ruta |
| `manifest.json` | configuración, semillas, versiones, conteos, rangos y simplificaciones asumidas |
| `routes_geometry.jsonl` | geometría cruda por ruta (elevación, rumbo, velocidad) para la fase de validación en Unity |
| `README.md` | resumen de la corrida generada |

---

## Licencia

Código bajo licencia **MIT** — ver [`LICENSE`](LICENSE).

Las dependencias de terceros y sus licencias están listadas en [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
