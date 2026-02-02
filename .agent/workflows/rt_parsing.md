---
description: Parse and Verify Inputs (Phase 0)
---

1. Ask the user for the `pdf_path` (Required).

2. Run command:
   ```bash
   if [ -f "$pdf_path" ]; then
     echo "PDF found: $pdf_path"
   else
     echo "Error: PDF file not found at $pdf_path"
     exit 1
   fi
   
   # Ensure dependencies are installed
   if [ -f "requirements.txt" ]; then
       # Check if packages are installed (simple check)
       pip freeze | grep -f requirements.txt > /dev/null || echo "Warning: Some requirements might be missing. Run 'pip install -r requirements.txt'"
   fi
   ```
