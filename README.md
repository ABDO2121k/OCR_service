# Belgian Vehicle Registration OCR

Automated data extraction from Belgian **Cartes Grises** (Vehicle Registration Certificates) built for a digital automotive marketplace. Reduces manual data-entry friction and delivers ~96% field accuracy at ~$20 per 10,000 documents.

**Pipeline:** PaddleOCR (local, free) → OpenAI GPT-4o-mini (JSON structuring)

---

## Quick Start

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Compose v2)
- An OpenAI API key

### 2. Create your `.env` file

Copy the example and fill in your key:

```bash
cp .env.example .env
```

Then open `.env` and set your `OPENAI_API_KEY`:

```env
APP_ENV=development
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123
MINIO_ENDPOINT=http://minio:9000
MINIO_BUCKET_NAME=vehicle-registrations
OPENAI_API_KEY=sk-...
```

> `APP_ENV=development` enables Swagger UI at `/docs` on both services.

### 3. Start the stack

```bash
docker compose up --build
```

Wait until you see all four services healthy (takes ~60 s on first run while PaddleOCR downloads its models):

```text
✔ minio          healthy
✔ minio-init     completed
✔ minio-service  healthy   →  http://localhost:8001
✔ ocr-engine     healthy   →  http://localhost:8000
```

### 4. Open the demo

Open **`demo.html`** directly in your browser — no web server needed, it is a standalone file.

Double-click `demo.html` from Explorer / Finder, or use your browser's **File → Open File** menu.

### 5. Upload a Carte Grise

- Use one of the sample images in the **`test_image/`** folder, **or**
- Drag-and-drop / select any JPEG or PNG photo of a Belgian vehicle registration certificate

The demo will:

1. Upload the image to the MinIO Storage Service
2. Run PaddleOCR + GPT-4o-mini extraction
3. Display a vehicle summary card and a full 25-field data table

---

## Swagger UI

Available when `APP_ENV=development` is set in `.env`:

| Service | URL |
| --- | --- |
| OCR Engine | <http://localhost:8000/docs> |
| MinIO Storage Service | <http://localhost:8001/docs> |

---

## Architecture

```text
Browser (demo.html)
  │
  ├── POST /upload ──────► MinIO Storage Service (port 8001)
  │                              │
  │                              ├── raw/{uuid}.jpg  ─► MinIO bucket
  │                              └── processed/{uuid}.jpg (deskew + CLAHE + resize)
  │
  └── POST /process/{uuid} ─► OCR Engine (port 8000)
                                    │
                                    ├── fetch preprocessed image from MinIO
                                    ├── PaddleOCR  →  raw text
                                    └── GPT-4o-mini →  structured JSON (25 fields)
```

### Services

| Service | Port | Responsibility |
| --- | --- | --- |
| `minio` | 9000 / 9001 | S3-compatible object storage + web console |
| `minio-service` | 8001 | Upload API: stores raw + preprocessed images |
| `ocr-engine` | 8000 | OCR API: PaddleOCR + GPT-4o-mini extraction |

---

## Stack

| Layer | Technology |
| --- | --- |
| API framework | FastAPI (Python 3.10) |
| Local OCR | PaddleOCR (`lang="latin"`) |
| AI structuring | OpenAI GPT-4o-mini |
| Object storage | MinIO (S3-compatible) |
| Containerization | Docker Compose |
| Image preprocessing | OpenCV (deskew, CLAHE, resize) |

---

## Extracted Fields (25)

All EU harmonized field codes from the Belgian Carte Grise:

| Code | Field | JSON key |
| --- | --- | --- |
| A | License plate | `license_plate` |
| B | First registration date | `first_registration_date` |
| E | VIN | `vin` |
| C.1 | Owner name | `owner_name` |
| C.3 | Registration address | `registration_address` |
| D.1 | Brand | `brand` |
| D.2 | Type / variant / version | `type_variant_version` |
| D.2.1 | CNIT code | `cnit_code` |
| D.3 | Commercial name | `commercial_name` |
| F.1 | Gross vehicle weight (kg) | `gross_vehicle_weight_kg` |
| F.2 | Adjusted GVW (kg) | `adjusted_gvw_kg` |
| F.3 | Max towing weight (kg) | `max_towing_weight_kg` |
| O.1 | Max trailer weight (kg) | `max_trailer_weight_kg` |
| J.1 | EU category | `eu_category` |
| J.2 | EU bodywork code | `eu_bodywork_code` |
| J.3 | National bodywork | `national_bodywork` |
| K | Type approval number | `type_approval_number` |
| P.1 | Engine capacity (cm³) | `engine_capacity_cm3` |
| P.2 | Max power (kW) | `max_power_kw` |
| P.3 | Fuel type | `fuel_type` |
| P.6 | Fiscal power (CV) | `fiscal_power_cv` |
| S.1 | Seating capacity | `seating_capacity` |
| S.2 | Standing places | `standing_places` |
| V.7 | CO₂ emissions (g/km) | `co2_emissions_g_km` |
| V.9 | Euro emission class | `euro_class` |

---

## Why Hybrid PaddleOCR + GPT?

| Architecture | Accuracy | Cost / 10k docs |
| --- | --- | --- |
| Tesseract only | ~70% | $0 |
| PaddleOCR only | ~91% | $0 |
| **Hybrid (this project)** | **~96%** | **~$20** |
| Azure Document Intelligence | ~98% | ~$95 |
| Google Document AI | ~98% | ~$450 |
| Specialized APIs (Mindee…) | 93–95% | ~$800+ |

PaddleOCR handles the pixel-to-text step for free. GPT-4o-mini handles OCR typo correction, multilingual EU code mapping, and JSON normalization — at a fraction of cloud OCR pricing.

---

## Stopping the stack

```bash
docker compose down
```

To also remove the MinIO data volume:

```bash
docker compose down -v
```

---

Based on architectural benchmark: `Benchmark_OCR_Marketplace.docx`
