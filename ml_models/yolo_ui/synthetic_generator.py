#!/usr/bin/env python
"""
Synthetic UI Screenshot Generator for YOLO Training

Generates random UI layouts with known bounding boxes for:
  buttons, inputs, selects, cards, images, icons, text blocks,
  headers, sidebars, navbars, separators

Each generated image has a matching YOLO-format label file.

Usage:
    python synthetic_generator.py --count 500 --output dataset/
"""
import os
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# YOLO classes
CLASSES = [
    'button', 'input', 'select', 'card', 'image', 'icon',
    'text_block', 'header_bar', 'sidebar', 'navbar', 'separator'
]

# UI color palettes
PALETTES = [
    # Light theme
    {
        'bg': [(245, 245, 245), (255, 255, 255), (250, 250, 252)],
        'card': [(255, 255, 255), (248, 249, 250), (240, 242, 245)],
        'button': [(0, 123, 255), (40, 167, 69), (220, 53, 69), (255, 193, 7),
                   (23, 162, 184), (108, 117, 125)],
        'text': [(33, 37, 41), (73, 80, 87), (108, 117, 125)],
        'header': [(52, 58, 64), (0, 123, 255), (32, 34, 36), (33, 37, 41)],
        'sidebar': [(52, 58, 64), (33, 37, 41), (248, 249, 250), (255, 255, 255)],
        'input_bg': [(255, 255, 255)],
        'input_border': [(206, 212, 218), (200, 200, 200)],
        'separator': [(222, 226, 230), (200, 200, 200)],
    },
    # Dark theme
    {
        'bg': [(18, 18, 18), (33, 37, 41), (25, 25, 25)],
        'card': [(45, 50, 55), (52, 58, 64), (60, 65, 70)],
        'button': [(0, 123, 255), (40, 167, 69), (255, 193, 7), (220, 53, 69)],
        'text': [(248, 249, 250), (173, 181, 189), (222, 226, 230)],
        'header': [(33, 37, 41), (18, 18, 18)],
        'sidebar': [(25, 25, 30), (33, 37, 41), (52, 58, 64)],
        'input_bg': [(52, 58, 64), (73, 80, 87)],
        'input_border': [(73, 80, 87), (108, 117, 125)],
        'separator': [(73, 80, 87), (52, 58, 64)],
    },
    # Blue/corporate theme
    {
        'bg': [(240, 244, 248), (236, 240, 245)],
        'card': [(255, 255, 255), (245, 248, 252)],
        'button': [(13, 110, 253), (25, 135, 84), (239, 68, 68)],
        'text': [(30, 41, 59), (71, 85, 105)],
        'header': [(15, 23, 42), (30, 41, 59)],
        'sidebar': [(15, 23, 42), (255, 255, 255)],
        'input_bg': [(255, 255, 255)],
        'input_border': [(196, 204, 214)],
        'separator': [(215, 222, 232)],
    },
    # Purple/dashboard theme
    {
        'bg': [(245, 243, 249), (248, 246, 252)],
        'card': [(255, 255, 255), (250, 248, 255)],
        'button': [(124, 58, 237), (99, 102, 241), (236, 72, 153)],
        'text': [(30, 20, 50), (75, 65, 95)],
        'header': [(30, 20, 50), (55, 48, 163)],
        'sidebar': [(30, 20, 50), (55, 48, 163)],
        'input_bg': [(255, 255, 255)],
        'input_border': [(196, 190, 210)],
        'separator': [(220, 216, 228)],
    },
]

# Common UI text strings
BUTTON_TEXTS = [
    'Submit', 'Cancel', 'Save', 'Delete', 'Edit', 'Add', 'Search',
    'Login', 'Register', 'OK', 'Close', 'Next', 'Back', 'Apply',
    'Settings', 'Profile', 'Logout', 'Dashboard', 'Reset',
]

LABEL_TEXTS = [
    'Name', 'Email', 'Password', 'Username', 'Phone', 'Address',
    'Description', 'Title', 'Date', 'Status', 'Type', 'Category',
    'Amount', 'Price', 'Quantity', 'Total', 'Notes',
]

HEADING_TEXTS = [
    'Dashboard', 'Settings', 'User Management', 'Reports', 'Analytics',
    'Overview', 'Profile', 'Notifications', 'Activity', 'Projects',
    'System Status', 'Data Management', 'Configuration',
]

PLACEHOLDER_TEXTS = [
    'Enter text...', 'Search...', 'Type here...', 'Email address',
    'Your name', 'Select option...', 'Filter...',
]

NAV_ITEMS = [
    'Home', 'Dashboard', 'Reports', 'Users', 'Settings', 'Help',
    'Products', 'Orders', 'Analytics', 'Profile', 'Messages',
]


def get_font(size=14):
    """Get a font, falling back to default if custom font not found."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


class SyntheticUIGenerator:
    def __init__(self, width_range=(800, 1920), height_range=(600, 1080)):
        self.width_range = width_range
        self.height_range = height_range

    def generate(self):
        """Generate one synthetic UI screenshot with labels."""
        w = random.randint(*self.width_range)
        h = random.randint(*self.height_range)
        palette = random.choice(PALETTES)

        img = Image.new('RGB', (w, h), random.choice(palette['bg']))
        draw = ImageDraw.Draw(img)
        annotations = []  # List of (class_id, cx, cy, nw, nh)

        # Track occupied regions to avoid overlap
        occupied = []

        # Layout decision
        has_header = random.random() < 0.85
        has_sidebar = random.random() < 0.4
        has_navbar = random.random() < 0.3

        content_y = 0
        content_x = 0
        content_w = w
        content_h = h

        # --- Header ---
        if has_header:
            header_h = random.randint(48, 80)
            header_color = random.choice(palette['header'])
            draw.rectangle([0, 0, w, header_h], fill=header_color)
            annotations.append((7, w / 2 / w, header_h / 2 / h, 1.0, header_h / h))
            occupied.append((0, 0, w, header_h))

            # Add nav items in header
            if has_navbar:
                nav_x = random.randint(w // 6, w // 3)
                nav_y = header_h // 2 - 10
                nav_items = random.sample(NAV_ITEMS, random.randint(3, 6))
                font = get_font(14)
                for item in nav_items:
                    text_color = (255, 255, 255) if sum(header_color) < 380 else (33, 37, 41)
                    bbox = font.getbbox(item)
                    tw = bbox[2] - bbox[0]
                    draw.text((nav_x, nav_y), item, fill=text_color, font=font)
                    nav_x += tw + random.randint(20, 40)

            # Logo/title text
            title_font = get_font(random.randint(16, 24))
            title = random.choice(['MyApp', 'Dashboard', 'Admin Panel', 'CRM', 'Portal'])
            text_color = (255, 255, 255) if sum(header_color) < 380 else (33, 37, 41)
            draw.text((15, header_h // 2 - 10), title, fill=text_color, font=title_font)

            content_y = header_h
            content_h -= header_h

        # --- Sidebar ---
        if has_sidebar:
            sidebar_w = random.randint(180, 260)
            sidebar_color = random.choice(palette['sidebar'])
            sidebar_left = random.random() < 0.7  # 70% left sidebar

            if sidebar_left:
                sx = 0
                content_x = sidebar_w
            else:
                sx = w - sidebar_w
                content_w -= sidebar_w

            draw.rectangle([sx, content_y, sx + sidebar_w, h], fill=sidebar_color)
            scx = (sx + sidebar_w / 2) / w
            scy = (content_y + (h - content_y) / 2) / h
            snw = sidebar_w / w
            snh = (h - content_y) / h
            annotations.append((8, scx, scy, snw, snh))
            occupied.append((sx, content_y, sx + sidebar_w, h))

            if sidebar_left:
                content_w -= sidebar_w

            # Sidebar menu items
            menu_y = content_y + 20
            menu_font = get_font(14)
            text_color = (255, 255, 255) if sum(sidebar_color) < 380 else (33, 37, 41)
            for item in random.sample(NAV_ITEMS, random.randint(4, 8)):
                if menu_y + 30 > h - 20:
                    break
                mx = sx + 20
                draw.text((mx, menu_y), item, fill=text_color, font=menu_font)
                # Text block annotation
                bbox_text = menu_font.getbbox(item)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
                annotations.append((6,
                                    (mx + tw / 2) / w,
                                    (menu_y + th / 2) / h,
                                    tw / w, th / h))
                menu_y += random.randint(32, 48)

        # --- Content Area ---
        cx_start = content_x + 20
        cy_start = content_y + 20
        cx_end = content_x + content_w - 20
        cy_end = h - 20

        # Add content elements
        cursor_y = cy_start

        # Heading
        if random.random() < 0.8:
            heading = random.choice(HEADING_TEXTS)
            heading_font = get_font(random.randint(22, 36))
            text_color = random.choice(palette['text'])
            draw.text((cx_start, cursor_y), heading, fill=text_color, font=heading_font)
            hbbox = heading_font.getbbox(heading)
            tw = hbbox[2] - hbbox[0]
            th = hbbox[3] - hbbox[1]
            annotations.append((6, (cx_start + tw / 2) / w, (cursor_y + th / 2) / h,
                                tw / w, th / h))
            cursor_y += th + random.randint(15, 30)

        # Separator after heading
        if random.random() < 0.4:
            sep_color = random.choice(palette['separator'])
            sep_y = cursor_y
            draw.line([(cx_start, sep_y), (cx_end, sep_y)], fill=sep_color, width=1)
            annotations.append((10, (cx_start + (cx_end - cx_start) / 2) / w,
                                sep_y / h, (cx_end - cx_start) / w, 2 / h))
            cursor_y += 15

        # Cards row
        n_cards = random.randint(1, 4)
        card_w = (cx_end - cx_start - (n_cards - 1) * 15) // n_cards
        card_h = random.randint(80, 180)

        if cursor_y + card_h < cy_end:
            for i in range(n_cards):
                card_x = cx_start + i * (card_w + 15)
                card_color = random.choice(palette['card'])
                # Draw card with rounded corners (approximate with rectangle)
                draw.rectangle([card_x, cursor_y, card_x + card_w, cursor_y + card_h],
                               fill=card_color, outline=random.choice(palette['separator']),
                               width=1)
                annotations.append((3,
                                    (card_x + card_w / 2) / w,
                                    (cursor_y + card_h / 2) / h,
                                    card_w / w, card_h / h))
                occupied.append((card_x, cursor_y, card_x + card_w, cursor_y + card_h))

                # Card content
                text_y = cursor_y + 15
                text_color = random.choice(palette['text'])
                label_font = get_font(12)
                value_font = get_font(random.choice([24, 28, 32]))

                # Label
                label = random.choice(LABEL_TEXTS)
                draw.text((card_x + 15, text_y), label, fill=text_color, font=label_font)
                lbbox = label_font.getbbox(label)
                annotations.append((6,
                                    (card_x + 15 + (lbbox[2] - lbbox[0]) / 2) / w,
                                    (text_y + (lbbox[3] - lbbox[1]) / 2) / h,
                                    (lbbox[2] - lbbox[0]) / w,
                                    (lbbox[3] - lbbox[1]) / h))
                text_y += 25

                # Big number
                value = str(random.randint(10, 99999))
                draw.text((card_x + 15, text_y), value, fill=text_color, font=value_font)
                vbbox = value_font.getbbox(value)
                annotations.append((6,
                                    (card_x + 15 + (vbbox[2] - vbbox[0]) / 2) / w,
                                    (text_y + (vbbox[3] - vbbox[1]) / 2) / h,
                                    (vbbox[2] - vbbox[0]) / w,
                                    (vbbox[3] - vbbox[1]) / h))

            cursor_y += card_h + random.randint(20, 40)

        # Form elements
        if random.random() < 0.6 and cursor_y + 200 < cy_end:
            n_fields = random.randint(2, 5)
            field_w = min(400, cx_end - cx_start - 40)
            field_h = random.randint(32, 42)

            for i in range(n_fields):
                if cursor_y + field_h + 30 > cy_end:
                    break

                field_x = cx_start + 10

                # Label
                label = random.choice(LABEL_TEXTS)
                label_font = get_font(13)
                text_color = random.choice(palette['text'])
                draw.text((field_x, cursor_y), label, fill=text_color, font=label_font)
                lbbox = label_font.getbbox(label)
                annotations.append((6,
                                    (field_x + (lbbox[2] - lbbox[0]) / 2) / w,
                                    (cursor_y + (lbbox[3] - lbbox[1]) / 2) / h,
                                    (lbbox[2] - lbbox[0]) / w,
                                    (lbbox[3] - lbbox[1]) / h))
                cursor_y += 20

                # Input or Select
                is_select = random.random() < 0.25
                input_bg = random.choice(palette['input_bg'])
                input_border = random.choice(palette['input_border'])

                draw.rectangle([field_x, cursor_y, field_x + field_w, cursor_y + field_h],
                               fill=input_bg, outline=input_border, width=1)

                if is_select:
                    # Draw dropdown arrow
                    arrow_x = field_x + field_w - 25
                    arrow_y = cursor_y + field_h // 2
                    draw.polygon([(arrow_x, arrow_y - 4), (arrow_x + 10, arrow_y - 4),
                                  (arrow_x + 5, arrow_y + 4)], fill=text_color)
                    cls_id = 2  # select
                else:
                    # Placeholder text
                    placeholder = random.choice(PLACEHOLDER_TEXTS)
                    ph_color = tuple(min(255, c + 80) for c in text_color)
                    draw.text((field_x + 10, cursor_y + field_h // 2 - 7),
                              placeholder, fill=ph_color, font=get_font(13))
                    cls_id = 1  # input

                annotations.append((cls_id,
                                    (field_x + field_w / 2) / w,
                                    (cursor_y + field_h / 2) / h,
                                    field_w / w, field_h / h))

                cursor_y += field_h + random.randint(12, 25)

        # Buttons row
        if cursor_y + 50 < cy_end:
            n_buttons = random.randint(1, 4)
            btn_texts = random.sample(BUTTON_TEXTS, min(n_buttons, len(BUTTON_TEXTS)))
            btn_x = cx_start + 10
            btn_h = random.randint(34, 48)
            btn_font = get_font(14)

            for text in btn_texts:
                if btn_x + 120 > cx_end:
                    break
                bbox_t = btn_font.getbbox(text)
                btn_w = (bbox_t[2] - bbox_t[0]) + random.randint(30, 60)
                btn_color = random.choice(palette['button'])

                draw.rectangle([btn_x, cursor_y, btn_x + btn_w, cursor_y + btn_h],
                               fill=btn_color)
                # Button text
                text_c = (255, 255, 255) if sum(btn_color) < 380 else (33, 37, 41)
                tx = btn_x + (btn_w - (bbox_t[2] - bbox_t[0])) // 2
                ty = cursor_y + (btn_h - (bbox_t[3] - bbox_t[1])) // 2
                draw.text((tx, ty), text, fill=text_c, font=btn_font)

                annotations.append((0,
                                    (btn_x + btn_w / 2) / w,
                                    (cursor_y + btn_h / 2) / h,
                                    btn_w / w, btn_h / h))

                btn_x += btn_w + random.randint(10, 20)

            cursor_y += btn_h + random.randint(20, 40)

        # Image placeholders
        if random.random() < 0.4 and cursor_y + 100 < cy_end:
            img_w = random.randint(150, min(400, cx_end - cx_start))
            img_h = random.randint(100, 250)
            img_x = cx_start + random.randint(10, max(10, cx_end - cx_start - img_w - 10))

            if cursor_y + img_h < cy_end:
                # Draw a noisy image placeholder
                noise = np.random.randint(100, 200, (img_h, img_w, 3), dtype=np.uint8)
                noise_img = Image.fromarray(noise)
                noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=3))
                img.paste(noise_img, (img_x, cursor_y))

                annotations.append((4,
                                    (img_x + img_w / 2) / w,
                                    (cursor_y + img_h / 2) / h,
                                    img_w / w, img_h / h))
                cursor_y += img_h + 20

        # Icons (small colored squares)
        if random.random() < 0.3:
            n_icons = random.randint(2, 6)
            icon_size = random.randint(20, 40)
            icon_x = cx_start + 10
            icon_y = min(cursor_y, cy_end - icon_size - 10)

            for _ in range(n_icons):
                if icon_x + icon_size > cx_end:
                    break
                icon_color = random.choice(palette['button'])
                draw.rectangle([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size],
                               fill=icon_color)
                # Draw simple shape inside (circle or line)
                inner = 4
                if random.random() < 0.5:
                    draw.ellipse([icon_x + inner, icon_y + inner,
                                  icon_x + icon_size - inner, icon_y + icon_size - inner],
                                 outline=(255, 255, 255), width=2)
                else:
                    mid = icon_size // 2
                    draw.line([(icon_x + inner, icon_y + mid),
                               (icon_x + icon_size - inner, icon_y + mid)],
                              fill=(255, 255, 255), width=2)

                annotations.append((5,
                                    (icon_x + icon_size / 2) / w,
                                    (icon_y + icon_size / 2) / h,
                                    icon_size / w, icon_size / h))
                icon_x += icon_size + random.randint(15, 30)

        # Add slight noise/texture to make it more realistic
        if random.random() < 0.3:
            noise_arr = np.array(img)
            noise = np.random.normal(0, 3, noise_arr.shape).astype(np.int16)
            noise_arr = np.clip(noise_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noise_arr)

        return img, annotations

    def generate_batch(self, count, output_dir, split_ratio=0.8):
        """Generate a batch of synthetic UI images with labels."""
        for split in ('train', 'val'):
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

        split_idx = int(count * split_ratio)

        for i in range(count):
            split = 'train' if i < split_idx else 'val'
            img, annotations = self.generate()

            img_name = f'synthetic_{i:04d}.png'
            lbl_name = f'synthetic_{i:04d}.txt'

            img.save(os.path.join(output_dir, 'images', split, img_name))

            with open(os.path.join(output_dir, 'labels', split, lbl_name), 'w') as f:
                for ann in annotations:
                    cls_id, cx, cy, nw, nh = ann
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            if (i + 1) % 50 == 0 or i == count - 1:
                print(f"  Generated {i + 1}/{count}")

        # Write data.yaml
        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(CLASSES)}\n")
            f.write(f"names: {CLASSES}\n")

        print(f"\nDataset written to {output_dir}")
        print(f"  {split_idx} train + {count - split_idx} val images")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic UI training data')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of images to generate')
    parser.add_argument('--output', default='dataset',
                        help='Output directory')
    parser.add_argument('--width-min', type=int, default=800)
    parser.add_argument('--width-max', type=int, default=1920)
    parser.add_argument('--height-min', type=int, default=600)
    parser.add_argument('--height-max', type=int, default=1080)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    print(f"Generating {args.count} synthetic UI images...")
    gen = SyntheticUIGenerator(
        width_range=(args.width_min, args.width_max),
        height_range=(args.height_min, args.height_max),
    )
    gen.generate_batch(args.count, output_dir)


if __name__ == '__main__':
    main()
