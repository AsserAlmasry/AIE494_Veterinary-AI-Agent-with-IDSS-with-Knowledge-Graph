"""Quick smoke test for the MMCOWS pipeline."""
import urllib.request, json, io

# Build multipart form data manually (no requests needed)
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
img_path = "mmcows/mmcow/visual_data/images/0725/cam_1/1690271846_02-57-26.jpg"

with open(img_path, "rb") as f:
    img_data = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + img_data + (
    f"\r\n--{boundary}\r\n"
    f'Content-Disposition: form-data; name="generate_report"\r\n\r\n'
    f"false\r\n"
    f"--{boundary}--\r\n"
).encode()

req = urllib.request.Request(
    "http://localhost:8001/api/v1/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

try:
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    print("=== PREDICTION RESULT ===")
    print(f"  Status: {r.status}")
    print(f"  Gate: {d.get('stages', {}).get('gate')}")
    print(f"  Cow ID: {d.get('cow_id')}")
    print(f"  Success: {d.get('success')}")
    print(f"  Latency: {d.get('total_latency_ms')}ms")
    print(f"  Errors: {d.get('errors')}")
    
    # Check model results
    stages = d.get("stages", {})
    if "identity" in stages:
        identity = stages["identity"]
        print(f"\n  [Identity] Decision: {identity.get('decision')}")
        print(f"  [Identity] Total Cows: {identity.get('total_cows_detected')}")
        for det in identity.get("detections", [])[:3]:
            print(f"    Cow #{det.get('cow_id')} conf={det.get('confidence'):.4f} bbox={det.get('bbox')}")
    
    if "health" in stages and isinstance(stages["health"], dict):
        h = stages["health"]
        print(f"\n  [Health] Score: {h.get('health_score')}")
        print(f"  [Health] Risk: {h.get('risk_level')}")
        print(f"  [Health] Anomaly: {h.get('anomaly_detected')}")
    
    if "milk" in stages and isinstance(stages["milk"], dict):
        m = stages["milk"]
        print(f"\n  [Milk] Yield: {m.get('predicted_yield_kg')}kg")
        print(f"  [Milk] Risk: {m.get('risk_level')}")
    
    if "heat_stress" in stages and isinstance(stages["heat_stress"], dict):
        hs = stages["heat_stress"]
        print(f"\n  [Heat] Stress Level: {hs.get('stress_level')}")
    
    if "annotated_image_b64" in stages:
        print(f"\n  [Image] Annotated image: {len(stages['annotated_image_b64'])} chars (base64)")
    
    if "clinical_summary" in stages:
        print(f"\n  [Summary] {stages['clinical_summary']}")

except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
