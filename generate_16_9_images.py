import os
import numpy as np
import medmnist
from medmnist import INFO
from PIL import Image, ImageDraw

def create_16_9_image(dataset_flag, output_path, bg_color=(255, 255, 255)):
    info = INFO[dataset_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    print(f"Loading {dataset_flag}...")
    dataset = DataClass(split='train', download=True, size=224)
    images = dataset.imgs
    
    W, H = 1920, 1080
    canvas = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    num_imgs = len(images)
    large_idx = 0
    large_img_array = images[large_idx]
    if len(large_img_array.shape) == 2:
        large_img = Image.fromarray(large_img_array).convert('RGB')
    else:
        large_img = Image.fromarray(large_img_array)
        
    LARGE_SIZE = 1000
    large_img = large_img.resize((LARGE_SIZE, LARGE_SIZE), Image.NEAREST)
    
    x0, y0 = 40, 40
    # Draw border for large image
    border_w = 2
    draw.rectangle([x0 - border_w, y0 - border_w, x0 + LARGE_SIZE + border_w - 1, y0 + LARGE_SIZE + border_w - 1], fill=(0, 0, 0))
    canvas.paste(large_img, (x0, y0))
    
    COLS = 5
    ROWS = 6
    gap = 20
    cell_w = (800 - (COLS - 1) * gap) // COLS
    cell_h = (1000 - (ROWS - 1) * gap) // ROWS
    cell_size = min(cell_w, cell_h)
    
    start_x = 1080 + (800 - (COLS * cell_size + (COLS - 1) * gap)) // 2
    start_y = 40 + (1000 - (ROWS * cell_size + (ROWS - 1) * gap)) // 2
    
    grid_img_indices = np.random.RandomState(42).choice(num_imgs, size=COLS*ROWS + 1, replace=False)
    
    idx_ptr = 0
    if grid_img_indices[idx_ptr] == large_idx:
        idx_ptr += 1
        
    for r in range(ROWS):
        for c in range(COLS):
            if idx_ptr >= len(grid_img_indices):
                break
            
            i = grid_img_indices[idx_ptr]
            idx_ptr += 1
            if i == large_idx and idx_ptr < len(grid_img_indices):
                i = grid_img_indices[idx_ptr]
                idx_ptr += 1
                
            small_img_array = images[i]
            if len(small_img_array.shape) == 2:
                small_img = Image.fromarray(small_img_array).convert('RGB')
            else:
                small_img = Image.fromarray(small_img_array)
                
            small_img = small_img.resize((cell_size, cell_size), Image.NEAREST)
            
            px = start_x + c * (cell_size + gap)
            py = start_y + r * (cell_size + gap)
            
            draw.rectangle([px - border_w, py - border_w, px + cell_size + border_w - 1, py + cell_size + border_w - 1], fill=(0, 0, 0))
            canvas.paste(small_img, (px, py))
            
    canvas.save(output_path)
    print(f"Saved {output_path}")

if __name__ == '__main__':
    create_16_9_image('breastmnist', 'breastmnist_16_9.png')
    create_16_9_image('retinamnist', 'retinamnist_16_9.png')
