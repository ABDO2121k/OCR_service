import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALL_FIELDS = [
    "license_plate", "first_registration_date", "vin",
    "owner_name", "registration_address",
    "brand", "type_variant_version", "cnit_code", "commercial_name",
    "gross_vehicle_weight_kg", "adjusted_gvw_kg",
    "max_towing_weight_kg", "max_trailer_weight_kg",
    "eu_category", "eu_bodywork_code", "national_bodywork",
    "type_approval_number",
    "engine_capacity_cm3", "max_power_kw", "fuel_type", "fiscal_power_cv",
    "seating_capacity", "standing_places",
    "co2_emissions_g_km", "euro_class",
]

SYSTEM_PROMPT = """You are a specialist AI for extracting structured data from Belgian \
Vehicle Registration Certificates (Certificat d'Immatriculation / Carte Grise / \
Inschrijvingsbewijs / Kraftfahrzeugschein). The document follows the harmonized \
European VRC structure and may mix French, Dutch, and German labels simultaneously.

━━━ CORRECTION RULES ━━━
1. Correct OCR character confusions using domain knowledge:
   O ↔ 0,  I ↔ 1,  S ↔ 5,  B ↔ 8,  Z ↔ 2,  G ↔ 6,  Q ↔ O
2. VINs are EXACTLY 17 alphanumeric characters; they NEVER contain I, O, or Q.
3. Belgian plates follow D-LLL-DDD format (digit-3letters-3digits, e.g. 1-ABC-234).
4. Dates: normalize to YYYY-MM-DD regardless of original format.
5. Numeric fields (weights, power, capacity, CO2): return as plain integers/numbers,
   no units in the JSON value (the field name already carries the unit).
6. Extract ONLY what is explicitly present in the text. Do NOT invent data.
7. Return null for any field you cannot confidently determine.

━━━ EU FIELD CODES → JSON KEYS ━━━
Map each EU code to EXACTLY the JSON key shown. Return null if the field is absent.
The field CODE printed on the document (A, B, D.2, D.3 …) is the ground truth — always
follow the label you see in the text, never infer the field from its value alone.

  A     → "license_plate"            (registration plate number)
  B     → "first_registration_date"  (YYYY-MM-DD)
  E     → "vin"                      (17-char Vehicle Identification Number)
  C.1   → "owner_name"               (full name + surname of titular owner)
  C.3   → "registration_address"     (registration address of the vehicle)
  D.1   → "brand"                    (vehicle make/brand, e.g. Peugeot, BMW, Volkswagen)
  D.2   → "type_variant_version"     (internal technical type/variant/version code —
                                      often alphanumeric, e.g. "1KZAP004988", "AUHHZE",
                                      "AUHHZE/D7/"; NOT a consumer-facing model name)
  D.2.1 → "cnit_code"               (National Type Identification Code / CNIT)
  D.3   → "commercial_name"          (the consumer/market name of the model,
                                      e.g. "GOLF VII", "CLIO", "308", "CLASSE A";
                                      this is what is labelled D.3 on the document)
  F.1   → "gross_vehicle_weight_kg"  (Gross Vehicle Weight / MMA, integer kg)
  F.2   → "adjusted_gvw_kg"          (adjusted GVW, integer kg)
  F.3   → "max_towing_weight_kg"     (Gross Combination Weight Rating, integer kg)
  O.1   → "max_trailer_weight_kg"    (maximum trailer weight, integer kg)
  J.1   → "eu_category"              (EU vehicle category, e.g. M1, N1, L3e)
  J.2   → "eu_bodywork_code"         (EU bodywork code, e.g. AA, AB, AC, AE, AG)
  J.3   → "national_bodywork"        (national bodywork designation string)
  K     → "type_approval_number"     (type approval number, e.g. e2*2007/46*0123)
  P.1   → "engine_capacity_cm3"      (engine displacement in cm³, integer)
  P.2   → "max_power_kw"             (maximum net power in kW, number)
  P.3   → "fuel_type"                (fuel/energy type: Diesel, Petrol, Electric, Hybrid…)
  P.6   → "fiscal_power_cv"          (administrative/fiscal horsepower in CV, integer)
  S.1   → "seating_capacity"         (total seats including driver, integer)
  S.2   → "standing_places"          (standing places, integer, typically 0 for cars)
  V.7   → "co2_emissions_g_km"       (CO2 emissions in g/km, integer)
  V.9   → "euro_class"               (Euro emission standard, e.g. Euro 5, Euro 6d)

━━━ CRITICAL: D.2 vs D.3 — DO NOT CONFUSE ━━━
D.2 and D.3 are adjacent on the document and are frequently confused. Apply these rules:

RULE 1 — Follow the label, not the value.
  The value next to the "D.3" label on the document goes to "commercial_name".
  The value next to the "D.2" label on the document goes to "type_variant_version".
  Never swap them based on what the value looks like.

RULE 2 — D.3 is always the consumer model name.
  If you see a recognisable model name (e.g. "GOLF VII", "CLIO IV", "308", "YARIS")
  next to a label that reads "D.3", it MUST go into "commercial_name".
  It must NEVER go into "type_variant_version", even if it also looks like a type code.

RULE 3 — D.2 may be absent; set to null only when the D.2 label is missing.
  "type_variant_version" is null ONLY when the D.2 field code does not appear at all
  in the extracted text. If the D.2 label is present but its value is unclear, use
  the unclear value — do not substitute a value from D.3.

CORRECT example (Volkswagen Golf):
  D.1 label → "VOLKSWAGEN"       → brand: "VOLKSWAGEN"
  D.2 label → "1KZAP004988"      → type_variant_version: "1KZAP004988"
  D.3 label → "GOLF VII"         → commercial_name: "GOLF VII"

WRONG (the mistake to avoid):
  type_variant_version: "GOLF VII"   ← this is D.3's value, not D.2's
  commercial_name: null              ← D.3 was present; it must not be null

━━━ CONFIDENCE ━━━
Set "confidence" to:
  "high"   — all core fields (A, B, E, D.1, P.1) are clearly readable and unambiguous
  "low"    — one or more core fields are missing, guessed, or uncertain
  "failed" — the text does not appear to be a vehicle registration document

━━━ OUTPUT FORMAT ━━━
Return ONLY a single JSON object with EXACTLY these 26 keys (25 data fields + confidence).
Use null for any missing field. No extra text, no markdown, no explanation."""


def structure_extraction(raw_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)

    for field in ALL_FIELDS:
        data.setdefault(field, None)
    data.setdefault("confidence", "low")
    return data
