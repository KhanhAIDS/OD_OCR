import json
import shutil
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

dataset_path = Path("BOM-Dataset-Standardized")
json_file = "instances_default.json"
output_root = Path("dataset")

TRAIN_RATIO = 0.5        
SYNTHETIC_SIZE = 800       
SYNTHETIC_COUNT = 250    
FORCED_TRAIN_NAMES = ["3", "4", "9", "10", "18", "27", "30", "45", "54", "55", "58"]

if output_root.exists(): shutil.rmtree(output_root)
for split in ["train", "val"]:
    (output_root / split / "images").mkdir(parents=True, exist_ok=True)
    (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

with open(json_file, 'r') as f: coco_data = json.load(f)
cat_map = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
part_drawing_id = next((cat_map[c['id']] for c in coco_data['categories'] if c['name'] == 'PartDrawing'), 0)

img_id_to_annos = {}
for anno in coco_data['annotations']:
    img_id_to_annos.setdefault(anno['image_id'], []).append(anno)

images = coco_data['images']
train_imgs, remaining_imgs = [], []

for img in images:
    if Path(img['file_name']).stem in FORCED_TRAIN_NAMES:
        train_imgs.append(img)
    else:
        remaining_imgs.append(img)

random.shuffle(remaining_imgs)
needed_for_train = max(0, int(len(images) * TRAIN_RATIO) - len(train_imgs))

train_imgs.extend(remaining_imgs[:needed_for_train])
val_imgs = remaining_imgs[needed_for_train:]

print(f"Tổng số ảnh: {len(images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")

# AUGMENTATION SETUP
clahe_transform = A.Compose(
    [A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=1.0)], 
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

def save_yolo(img, bboxes, labels, split, name):
    cv2.imwrite(str(output_root / split / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(output_root / split / "labels" / f"{name}.txt", 'w') as f:
        for bbox, label in zip(bboxes, labels):
            f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

# VALIDATION SET (RAW ONLY)
for img_info in val_imgs:
    img = cv2.imread(str(dataset_path / img_info['file_name']))
    if img is None: continue
    w, h = img_info['width'], img_info['height']
    bboxes, labels = [], []
    for anno in img_id_to_annos.get(img_info['id'], []):
        ax, ay, aw, ah = anno['bbox']
        bboxes.append([(ax + aw/2)/w, (ay + ah/2)/h, aw/w, ah/h])
        labels.append(cat_map[anno['category_id']])
    save_yolo(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes, labels, "val", Path(img_info['file_name']).stem)

# TRAIN SET (RAW + 1 CLAHE) & CROP EXTRACTION
all_crops = []
for img_info in train_imgs:
    img = cv2.imread(str(dataset_path / img_info['file_name']))
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    w, h = img_info['width'], img_info['height']
    bboxes, labels = [], []
    
    for anno in img_id_to_annos.get(img_info['id'], []):
        ax, ay, aw, ah = map(int, anno['bbox'])
        bboxes.append([(ax + aw/2)/w, (ay + ah/2)/h, aw/w, ah/h])
        labels.append(cat_map[anno['category_id']])
        
        if aw > 10 and ah > 10 and ax+aw <= w and ay+ah <= h:
            all_crops.append({'image': img_rgb[ay:ay+ah, ax:ax+aw], 'label': cat_map[anno['category_id']]})

    base_name = Path(img_info['file_name']).stem
    
    # Save Original
    save_yolo(img_rgb, bboxes, labels, "train", base_name)
    
    # Save CLAHE
    try:
        aug = clahe_transform(image=img_rgb, bboxes=bboxes, class_labels=labels)
        save_yolo(aug['image'], aug['bboxes'], aug['class_labels'], "train", f"{base_name}_clahe")
    except: continue

# SYNTHETIC DATA (RAW + 1 CLAHE)
def check_overlap(new_box, existing_boxes):
    nx, ny, nw, nh = new_box
    for (ex, ey, ew, eh, _) in existing_boxes:
        if not (nx + nw < ex or nx > ex + ew or ny + nh < ey or ny > ey + eh): return True
    return False

def apply_crop_augs(c_img, label):
    fx, fy = random.uniform(0.7, 1.3), random.uniform(0.7, 1.3)
    c_img = cv2.resize(c_img, (0,0), fx=fx, fy=fy)
    
    if label == part_drawing_id:
        morph_choice = random.choice(['none', 'dilate', 'erode'])
        kernel = np.ones((2,2), np.uint8)
        if morph_choice == 'dilate':
            c_img = cv2.dilate(c_img, kernel, iterations=1)
        elif morph_choice == 'erode':
            c_img = cv2.erode(c_img, kernel, iterations=1)

        angle = random.choice([0, 90, 180, 270])
        if angle == 90:   c_img = cv2.rotate(c_img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180: c_img = cv2.rotate(c_img, cv2.ROTATE_180)
        elif angle == 270: c_img = cv2.rotate(c_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        flip_code = random.choice([-2, -1, 0, 1]) 
        if flip_code != -2:
            c_img = cv2.flip(c_img, flip_code)
            
    return c_img

for i in range(SYNTHETIC_COUNT):
    canvas = np.full((SYNTHETIC_SIZE, SYNTHETIC_SIZE, 3), 255, dtype=np.uint8)
    placed_boxes = []
    
    selected_crops = random.sample(all_crops, min(random.randint(3, 8), len(all_crops)))
    for crop in selected_crops:
        c_img = apply_crop_augs(crop['image'].copy(), crop['label'])
        ch, cw = c_img.shape[:2]
        
        if cw > SYNTHETIC_SIZE or ch > SYNTHETIC_SIZE:
            scale = min(SYNTHETIC_SIZE/cw, SYNTHETIC_SIZE/ch) * 0.9
            cw, ch = int(cw * scale), int(ch * scale)
            c_img = cv2.resize(c_img, (cw, ch))
            
        for _ in range(50):
            cx, cy = random.randint(0, SYNTHETIC_SIZE - cw), random.randint(0, SYNTHETIC_SIZE - ch)
            if not check_overlap((cx, cy, cw, ch), placed_boxes):
                canvas[cy:cy+ch, cx:cx+cw] = c_img
                placed_boxes.append((cx, cy, cw, ch, crop['label']))
                break
                
    yolo_boxes, labels = [], []
    for (x, y, w, h, label) in placed_boxes:
        yolo_boxes.append([(x + w/2)/SYNTHETIC_SIZE, (y + h/2)/SYNTHETIC_SIZE, w/SYNTHETIC_SIZE, h/SYNTHETIC_SIZE])
        labels.append(label)
        
    # Save Synthetic Original
    save_yolo(canvas, yolo_boxes, labels, "train", f"synthetic_{i}")
    
    # Save Synthetic CLAHE
    try:
        aug = clahe_transform(image=canvas, bboxes=yolo_boxes, class_labels=labels)
        save_yolo(aug['image'], aug['bboxes'], aug['class_labels'], "train", f"synthetic_{i}_clahe")
    except: continue