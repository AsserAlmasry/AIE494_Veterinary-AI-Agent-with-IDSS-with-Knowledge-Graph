import requests
import json
import time

def test_predict():
    url = "http://127.0.0.1:8000/api/v1/predict"
    
    # Using the generated test image
    image_path = 'C:/Users/Dell/.gemini/antigravity/brain/e593d122-031a-49a3-81b1-9d13a40db813/cow_mastitis_test_image_1775975581715.png'
    files = {
        'image': ('Mastitis.png', open(image_path, 'rb'), 'image/png')
    }
    data = {
        'sensor_json': '{}',
        'animal_weight_kg': 600,
        'cow_id_override': 0,
        'generate_report': 'true'
    }
    
    print(f"Sending request to {url}...")
    start_time = time.perf_counter()
    response = requests.post(url, files=files, data=data)
    elapsed = time.perf_counter() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Wall clock time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Pipeline Latency: {result.get('total_latency_ms')}ms")
        print(f"Pipeline Version: {result.get('pipeline_version')}")
        
        summary = result.get('stages', {}).get('clinical_summary', {})
        print(f"Primary Finding: {summary.get('primary_finding')}")
        print(f"Confidence: {summary.get('primary_confidence')}")
        
        report = result.get('stages', {}).get('report', {}).get('report', "")
        print("\n--- REPORT EXCERPT ---")
        # Look for the dosing string
        if "600 kg" in report:
            print("SUCCESS: Weight-based dosing found in report.")
            # Print a few lines around the dose
            lines = report.split('\n')
            for i, line in enumerate(lines):
                if "Flunixin" in line and "600" in line:
                    print(f"Dose Line: {line}")
        else:
            print("WARNING: Weight-based dosing NOT found in report excerpt.")
            
        if "Mastitis" in report or "mastitis" in report:
            print("SUCCESS: Mastitis mentioned in report.")
        else:
            print("WARNING: Mastitis NOT mentioned in report.")

if __name__ == "__main__":
    test_predict()
