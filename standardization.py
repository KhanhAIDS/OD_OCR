import os
from PIL import Image

input_dir = 'BOM-Dataset'
output_dir = 'BOM-Dataset-Standardized'
max_dimension = 2048

os.makedirs(output_dir, exist_ok=True)

processed_count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        input_path = os.path.join(input_dir, filename)
        
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        try:
            with Image.open(input_path) as img:
                if img.mode in ('RGBA', 'P', 'CMYK'):
                    img = img.convert('RGB')

                width, height = img.size
                if max(width, height) > max_dimension:
                    scaling_factor = max_dimension / float(max(width, height))
                    new_size = (int(width * scaling_factor), int(height * scaling_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                img.save(output_path, 'PNG', optimize=True)
                processed_count += 1
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print(f"Standardization complete. {processed_count} images saved to {output_dir}.")