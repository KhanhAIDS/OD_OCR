import gradio as gr
import cv2
import json
import os
import base64
from ultralytics import RTDETR
from openai import OpenAI

MODEL_PATH = "best.pt" 
model = RTDETR(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Bounding box color table
COLORS = [(255, 56, 56), (44, 153, 168), (255, 157, 151), (255, 112, 31), (255, 178, 29), (115, 218, 189)]

def encode_image(img_numpy):
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def apply_clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

def extract_content(image_crop, class_name):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key: return "Missing OPENAI_API_KEY"
    
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_crop)
    
    if class_name.lower() == "table":
        sys_msg = "You are a table reconstruction engine specialized in engineering drawings."
        prompt = """Task:
Given a cropped image containing exactly one table region, extract the table and reconstruct it as a single strict HTML <table> element that matches the visible table as faithfully as possible.

Primary objective:
Preserve the table structure exactly as shown in the image:
- keep the same row order
- keep the same column order
- preserve merged cells using rowspan and colspan only when they are visually supported
- preserve cell boundaries, alignment, and hierarchy
- preserve all visible text inside the table

Critical rules:
1. Output ONLY the raw HTML string for one <table> element.
2. Do NOT wrap the result in markdown.
3. Do NOT add any explanation, notes, comments, labels, or code fences.
4. Do NOT output JSON.
5. Do NOT invent content that is not visible.
6. Do NOT omit any visible cell text.
7. Do NOT collapse or merge cells unless the image clearly indicates a merge.
8. Do NOT change the semantic meaning of any text.
9. Do NOT clean up spelling, abbreviations, symbols, or units unless the image is obviously ambiguous.
10. Do NOT reorder rows or columns.
11. Do NOT add extra columns, rows, headers, or footers that are not present in the image.

Reconstruction rules:
- Recreate the table layout as faithfully as possible, even if the image quality is imperfect.
- If the table has header rows, use <th> for header cells when appropriate.
- If the table has multiline text in a cell, preserve line breaks using <br>.
- If a cell is empty, keep it empty rather than filling it with guessed content.
- If a cell contains only a symbol, number, unit, or short code, preserve it exactly as shown.
- If borders are visible but text is hard to read, prefer structural accuracy over guessing text.
- If the table has multiple levels of headers, reconstruct the hierarchy with appropriate merged cells.
- If there are visually merged cells, reflect them with rowspan/colspan.
- If the table is irregular or partially cropped, reconstruct only the visible portion; do not infer missing parts.
- If there are no tables in the crop, output exactly: <table></table>

Text fidelity rules:
- Preserve casing, punctuation, digits, units, and symbols exactly as they appear.
- Keep technical notation unchanged, including slashes, hyphens, parentheses, arrows, and bracketed text.
- Preserve Vietnamese/English mixed text exactly as visible.
- Do not translate anything.

HTML rules:
- Return valid HTML only.
- Use only standard table tags: <table>, <thead>, <tbody>, <tr>, <th>, <td>.
- Do not add any surrounding tags such as <html>, <body>, <div>, or <p>.
- Do not include comments.
- Escape HTML special characters properly when needed.

Final output requirement:
Return only the raw HTML <table>...</table> string."""
    elif class_name.lower() == "note":
        sys_msg = "You are a technical data extraction engine."
        prompt = """Extract all text, notes, annotations, callouts, and labels from this image.

Rules:
- Return ONLY valid JSON.
- Do NOT use markdown fences.
- Do NOT add explanations.
- Do NOT hallucinate or infer missing text.
- Preserve line breaks when visually meaningful.
- Keep punctuation, symbols, and casing exactly as shown.
- If no text is found, return {"items": []}.

Output schema:
{
  "items": [
    {
      "text": "..."
    }
  ]
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.replace("```html", "").replace("```json", "").replace("```", "").strip()
        
        if class_name.lower() != "table":
            try: return json.loads(content)
            except: return content
            
        return content
    except Exception as e:
        return str(e)

def process_pipeline(image, use_clahe):
    if image is None or model is None:
        return None, "Model file not found or image missing.", None, None, None
    
    if use_clahe:
        image = apply_clahe_rgb(image)

    results = model(image)
    
    # Tạo bản sao ảnh để vẽ Bounding Box
    annotated_img = image.copy()
    
    output_json = {
        "image": "uploaded_drawing.jpg",
        "objects": []
    }
    cropped_images = []
    
    # OCR Panel
    ocr_panel_html = "<div style='font-family: monospace;'>"
    
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # Vẽ Bounding Box và Label
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        label = f"{class_name} {conf:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        crop_img = image[y1:y2, x1:x2]
        cropped_images.append((crop_img, f"{class_name}_{idx}"))
        
        ocr_content = ""
        if class_name.lower() in ["table", "note"]:
            ocr_content = extract_content(crop_img, class_name)
            
            # Đẩy kết quả vào OCR Panel
            ocr_panel_html += f"<h3> {class_name.upper()} {idx + 1}</h3>"
            if class_name.lower() == "table":
                ocr_panel_html += f"{ocr_content}<br><hr>"
            elif class_name.lower() == "note":
                # Chỉ trích xuất 'text' từ JSON
                extracted_texts = []
                if isinstance(ocr_content, dict) and "items" in ocr_content:
                    for item in ocr_content["items"]:
                        if "text" in item and item["text"]:
                            extracted_texts.append(item["text"])
                
                # Gộp các dòng text lại, hoặc giữ nguyên nếu không parse được
                display_text = "<br><br>".join(extracted_texts) if extracted_texts else str(ocr_content)
                ocr_panel_html += f"<div style='background:#f4f4f4; padding:10px; border-radius:5px;'>{display_text}</div><br><hr>"
            
        obj_data = {
            "id": idx + 1,
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "ocr_content": ocr_content
        }
        output_json["objects"].append(obj_data)
        
    ocr_panel_html += "</div>"
        
    json_path = "final_output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)
        
    return annotated_img, ocr_panel_html, output_json, json_path, cropped_images

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Engineering Drawing Pipeline (RT-DETR + GPT-4o)")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="numpy", label="1. Input Drawing")
            use_clahe = gr.Checkbox(label="Enable CLAHE (Enhance Contrast)", value=False) 
            btn = gr.Button("Run Extraction Pipeline", variant="primary")
            
        with gr.Column():
            annotated_out = gr.Image(type="numpy", label="2. Detected Regions", show_label=False)
    
    with gr.Tabs():
        with gr.TabItem("OCR Panel"):
            ocr_text_out = gr.HTML(label="Extracted Tables & Notes")
        with gr.TabItem("Cropped Regions"):
            gallery_out = gr.Gallery(label="Isolated Components", columns=4)
        with gr.TabItem("Master JSON Data"):
            json_out = gr.JSON(label="Full Pipeline Output")
            file_download = gr.File(label="Download final_output.json")
            
    btn.click(
        process_pipeline, 
        inputs=[img_in, use_clahe],
        outputs=[annotated_out, ocr_text_out, json_out, file_download, gallery_out]
    )

demo.launch()