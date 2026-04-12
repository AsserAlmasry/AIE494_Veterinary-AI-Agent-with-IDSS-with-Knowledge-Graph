"""Quick smoke test — verifies Neo4j KG context and RAG retrieval are working."""
import json
import requests

import pathlib

# Prefer HEALTHY.jpeg in cwd, fall back to any available test image
img_candidates = list(pathlib.Path(".").glob("*.jpeg")) + list(pathlib.Path(".").glob("*.jpg"))
if not img_candidates:
    # Use a bundled skimage test image
    import skimage.data
    import numpy as np
    from PIL import Image
    arr = skimage.data.chelsea()  # a cat, but any RGB image works to verify the pipeline
    pil = Image.fromarray(arr.astype(np.uint8))
    img_path = pathlib.Path("data/smoke_test_cow.jpg")
    img_path.parent.mkdir(exist_ok=True)
    pil.save(img_path)
    img_candidates = [img_path]

img_path = img_candidates[0]
print(f"Using test image: {img_path}")

with open(img_path, "rb") as f:
    files = {"image": (img_path.name, f, "image/jpeg")}
    data  = {"generate_report": "true"}
    r = requests.post(
        "http://localhost:8000/api/v1/predict",
        files=files,
        data=data,
        timeout=90,
    )


result = r.json()
stages = result.get("stages", {})

print("=" * 60)
print("STATUS")
print(f"  HTTP:    {r.status_code}")
print(f"  cow_id:  {result.get('cow_id')}")
print(f"  success: {result.get('success')}")
print(f"  latency: {result.get('total_latency_ms')} ms")
print(f"  errors:  {result.get('errors', [])}")

print()
print("DISEASE")
preds = stages.get("disease", {}).get("predictions", [])
for p in preds:
    sv = "OK" if p.get("safety_validated", True) else "--"
    print(f"  [{sv}] {p['disease']:<25} {p['confidence']:.1%}")

print()
print("KNOWLEDGE GRAPH")
kg = stages.get("knowledge_graph", {})
print(f"  disease_info:      {list(kg.get('disease_info', {}).keys())}")
print(f"  treatment_protocols: {len(kg.get('treatment_protocols', []))}")
print(f"  progression_risks:   {len(kg.get('progression_risks', []))}")
print(f"  zoonotic_alerts:     {len(kg.get('zoonotic_alerts', []))}")
print(f"  cow_history:         {len(kg.get('cow_history', []))}")

tx = kg.get("treatment_protocols", [])
if tx:
    t = tx[0]
    print(f"  -> 1st treatment: {t.get('treatment')} [evidence={t.get('evidence_level')}]")

print()
print("RAG")
rag = stages.get("rag", {})
print(f"  retrieved:  {rag.get('retrieved')} docs")
docs = rag.get("documents", [])
for doc in docs[:2]:
    print(f"  [{doc.get('similarity'):.3f}] {doc.get('title','?')[:70]}")

print()
print("CLINICAL SUMMARY")
cs = stages.get("clinical_summary", {})
print(json.dumps(cs, indent=2))

print()
print("REPORT SNIPPET (first 300 chars)")
report = stages.get("report", {}).get("report", "")
print(report[:300])
