# Engineering Drawing OD & OCR Pipeline

This repository contains the full source code for an automated pipeline that detects tables and notes in engineering drawings and extracts them into structured formats (HTML and JSON) using RT-DETR and GPT-4o.

## Quick Links
* **Live Web Demo:** [Hugging Face Space Demo](https://huggingface.co/spaces/KhanhAIDS/OD_OCR)
* **Model Weights (`best.pt`):** [Download Pre-trained Weights](https://huggingface.co/spaces/KhanhAIDS/OD_OCR/resolve/main/best.pt?download=true)

---

## 1. Environment Setup
To run this project locally, clone the repository and install the necessary dependencies via `requirements.txt`:

```bash
# Clone the repository
git clone [https://github.com/YourUsername/OD_OCR.git](https://github.com/YourUsername/OD_OCR.git)
cd OD_OCR

# Install required packages
pip install -r requirements.txt
```

---

## 2. Dataset Preparation
If you wish to train the model yourself, you must first prepare the dataset. Run the following scripts sequentially to process the raw data:

1. **Standardize the dataset:**
   ```bash
   python standardization.py
   ```
2. **Apply data augmentation** (This will generate the final dataset used for training):
   ```bash
   python augment.py
   ```

---

## 3. Model Training
Once the dataset is fully generated, you can train the RT-DETR model. 

1. Open the `training.ipynb` file in Jupyter Notebook or VS Code.
2. Run all the cells sequentially to initialize the training process. 
3. The newly trained weights will be saved in the `runs/detect/` folder.

---

## 4. Running the Inference Pipeline (Local App)
The inference application runs on Gradio and requires an active internet connection to communicate with OpenAI's API for the OCR tasks.

**Step 1: Obtain an OpenAI API Key**
You must have a valid OpenAI API key with access to the `gpt-4o` model.

**Step 2: Set your API Key as an Environment Variable**
Before running the app, set your key as an environment variable named `OPENAI_API_KEY`.
* **Windows (Command Prompt):**
  ```cmd
  set OPENAI_API_KEY=your_actual_api_key_here
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:OPENAI_API_KEY="your_actual_api_key_here"
  ```
* **Mac/Linux:**
  ```bash
  export OPENAI_API_KEY="your_actual_api_key_here"
  ```

**Step 3: Launch the Web App**
Make sure the `best.pt` model file is in the root directory, then run:
```bash
python app.py
```
A local link (e.g., `http://127.0.0.1:7860`) will appear in your terminal. Click it to open the GUI in your browser.

---

```
