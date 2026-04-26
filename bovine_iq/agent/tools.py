import os
import re
from langchain_core.tools import tool
from data.ingestion import DataIngestion
from knowledge.vector_store import VetVectorStore

ingestion = DataIngestion()
vet_db = VetVectorStore()

@tool
def get_cow_status(cow_id: str) -> str:
    """Returns the latest Core Body Temperature (CBT), average milk yield, and health event count for a given cow (e.g., C01)."""
    stats = ingestion.get_latest_stats(cow_id)
    other = stats.get('other_sensors', {})
    other_str = ", ".join([f"{k}: {v}" for k, v in other.items()])
    return (f"Status for {cow_id}:\\n"
            f"- Latest CBT: {stats['latest_cbt']} °C\\n"
            f"- Avg Daily Milk Yield (last 7 days): {stats['avg_milk_7d']} kg\\n"
            f"- Total Health Events Logged: {stats['health_events']}\\n"
            f"- Other Integrated Sensors: {other_str}")

@tool
def search_vet_knowledge(query: str) -> str:
    """Search the veterinary textbooks and research papers (RAG) for medical protocols, pathophysiology, treatments, and clinical signs for bovine diseases."""
    return vet_db.search(query, k=2)

@tool
def get_cow_image(cow_id: str) -> str:
    """Retrieves an actual image of the cow from the MMCOWS visual_data dataset feed, draws a bounding box around it, and displays it on the screen."""
    import os, random
    from PIL import Image, ImageDraw, ImageFont
    
    try:
        class_id = str(int(cow_id.replace("C", "")))
    except ValueError:
        return f"Error: Invalid cow_id {cow_id}. Must be like 'C01'."
        
    cam1_img_dir = "d:/graduation project/Mmcows/mmcows/mmcow/visual_data/images/0725/cam_1"
    cam1_lbl_dir = "d:/graduation project/Mmcows/mmcows/mmcow/visual_data/labels/combined/0725/cam_1"
    
    if not os.path.exists(cam1_lbl_dir):
        return "Error: Visual dataset labels directory not found."
        
    # Quickly find an image containing this cow's YOLO label
    matched_files = []
    # Only scan first 500 files to avoid blocking chat for too long
    all_files = [f for f in os.listdir(cam1_lbl_dir) if f.endswith('.txt')]
    random.shuffle(all_files)
    
    for f in all_files[:500]:
        try:
            with open(os.path.join(cam1_lbl_dir, f), 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    parts = line.strip().split()
                    # Support 1-indexed or 0-indexed YOLO labels
                    if parts and parts[0] in [class_id, str(int(class_id)-1)]:
                        matched_files.append((f, parts))
                        break
        except Exception:
            pass
            
    if not matched_files:
        return f"Error: Could not find any recent frame currently showing {cow_id} in the camera view."
        
    selected_label, bbox_parts = random.choice(matched_files)
    selected_img = selected_label.replace('.txt', '.jpg')
    img_path = os.path.join(cam1_img_dir, selected_img)
    
    if not os.path.exists(img_path):
        return f"Error: Image frame {selected_img} not found."
        
    # Draw Bounding Box Using PIL
    try:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # YOLO format: class x_center y_center width height
        x_c, y_c, bw, bh = map(float, bbox_parts[1:5])
        x1 = (x_c - bw/2) * w
        y1 = (y_c - bh/2) * h
        x2 = (x_c + bw/2) * w
        y2 = (y_c + bh/2) * h
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="#34d399", width=5)
        
        # Draw Label block
        label_text = f"Cow {cow_id}"
        try:
            font = ImageFont.truetype("arial.ttf", 26)
        except (IOError, ImportError):
            font = ImageFont.load_default()
            
        rect_y1 = max(0, y1 - 35)
        draw.rectangle([x1, rect_y1, x1 + 120, y1], fill="#34d399")
        draw.text((x1 + 8, rect_y1 + 5), label_text, fill="black", font=font)
        
        # Save temp image for Streamlit rendering
        out_dir = "d:/graduation project/Mmcows/mmcows/bovine_iq/temp_images"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"highlighted_{cow_id}_{selected_img}")
        
        # Avoid saving directly as JPEG to prevent quality degradation loops if not needed, but JPEG is fine
        img.save(out_path, format="JPEG", quality=90)
        
        return f"Successfully retrieved the visual camera feed with bounding box tracking for {cow_id}. [SHOW_IMAGE: {out_path}]"
    except Exception as e:
        return f"Error displaying visual data boundary boxes: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """
    Executes generated Python code locally and returns the stdout output and errors.
    You have FULL PERMISSION to import and use any Python library inside the script, including image processing (cv2, PIL), file I/O, data science (pandas), or ML pipelines. Do not refuse graphical or file-based requests.
    """
    import sys
    from io import StringIO
    import traceback
    
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Provide common data science libraries in the execution context
        import pandas as pd
        import numpy as np
        
        exec_globals = {
            'pd': pd, 
            'np': np,
            '__builtins__': __builtins__
        }
        
        exec(code, exec_globals)
        output = redirected_output.getvalue()
        if not output:
             output = "Code executed successfully with no stdout output."
        return output
    except Exception as e:
        error_msg = traceback.format_exc()
        return f"Execution Failed. Please debug and retry. Traceback:\n{error_msg}"
    finally:
        sys.stdout = old_stdout

TOOLS = [get_cow_status, search_vet_knowledge, get_cow_image, execute_python_code]
