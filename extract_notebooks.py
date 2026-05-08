import json
import sys
import os

def extract_notebook(path, outfile):
    if not os.path.exists(path):
        outfile.write(f"File not found: {path}\n")
        return
    outfile.write(f"\n{'='*80}\nNotebook: {os.path.basename(path)}\n{'='*80}\n")
    with open(path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            outfile.write(f"Error reading {path}: {e}\n")
            return
            
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        if not source.strip():
            continue
        if cell_type == 'markdown':
            outfile.write(f"\n--- Markdown ---\n{source[:500]}...\n") 
        elif cell_type == 'code':
            outfile.write(f"\n--- Code ---\n{source[:1000]}\n")
            
if __name__ == '__main__':
    notebooks = [
        r"C:\Users\Dell\.gemini\antigravity\scratch\veterinary_ai_system\graduation project models\Task 1-20260426T224107Z-3-001\Task 1\mmcow-task-1.ipynb",
        r"C:\Users\Dell\.gemini\antigravity\scratch\veterinary_ai_system\graduation project models\Task 2-20260426T224124Z-3-001\Task 2\heat-stress-monitoring-system-task-2.ipynb",
        r"C:\Users\Dell\.gemini\antigravity\scratch\veterinary_ai_system\graduation project models\Task 3-Disease Classification\aie494-grad-cow-disease-classification (2).ipynb",
        r"C:\Users\Dell\.gemini\antigravity\scratch\veterinary_ai_system\graduation project models\Task 4-20260426T224204Z-3-001\Task 4\mmcows-task-4-organized.ipynb",
        r"C:\Users\Dell\.gemini\antigravity\scratch\veterinary_ai_system\graduation project models\Task 5-20260426T224142Z-3-001\Task 5\health-disease-early-warning-system-task-5.ipynb"
    ]
    with open('notebook_summary.txt', 'w', encoding='utf-8') as out:
        for nb in notebooks:
            extract_notebook(nb, out)
