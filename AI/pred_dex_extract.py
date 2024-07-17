import os
import zipfile
from PIL import Image
import shutil


def create_subdirectory(extract_to, apk_name):
    subdirectory = os.path.join(extract_to, os.path.splitext(os.path.basename(apk_name))[0])
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    return subdirectory


def extract_dex_files(apk_path, extract_to):
    subdirectory = create_subdirectory(extract_to, apk_path)
    with zipfile.ZipFile(apk_path, 'r') as apk:
        dex_files = [file for file in apk.namelist() if file.endswith('.dex')]
        apk.extractall(subdirectory, dex_files)
    return [os.path.join(subdirectory, file) for file in dex_files], subdirectory


def parse_dex_file(dex_path):
    with open(dex_path, 'rb') as file:
        content = file.read()

    header = content[:0x70]

    string_ids_size = int.from_bytes(content[0x38:0x3C], byteorder='little')
    string_ids_off = int.from_bytes(content[0x3C:0x40], byteorder='little')

    type_ids_size = int.from_bytes(content[0x40:0x44], byteorder='little')
    type_ids_off = int.from_bytes(content[0x44:0x48], byteorder='little')

    proto_ids_size = int.from_bytes(content[0x48:0x4C], byteorder='little')
    proto_ids_off = int.from_bytes(content[0x4C:0x50], byteorder='little')

    field_ids_size = int.from_bytes(content[0x50:0x54], byteorder='little')
    field_ids_off = int.from_bytes(content[0x54:0x58], byteorder='little')

    method_ids_size = int.from_bytes(content[0x58:0x5C], byteorder='little')
    method_ids_off = int.from_bytes(content[0x5C:0x60], byteorder='little')

    class_defs_size = int.from_bytes(content[0x60:0x64], byteorder='little')
    class_defs_off = int.from_bytes(content[0x64:0x68], byteorder='little')

    string_ids = content[string_ids_off:string_ids_off + string_ids_size * 4]
    type_ids = content[type_ids_off:type_ids_off + type_ids_size * 4]
    proto_ids = content[proto_ids_off:proto_ids_off + proto_ids_size * 12]
    field_ids = content[field_ids_off:field_ids_off + field_ids_size * 8]
    method_ids = content[method_ids_off:method_ids_off + method_ids_size * 8]
    class_defs = content[class_defs_off:class_defs_off + class_defs_size * 32]

    return {
        'header': header,
        'string_ids': string_ids,
        'type_ids': type_ids,
        'proto_ids': proto_ids,
        'field_ids': field_ids,
        'method_ids': method_ids,
        'class_defs': class_defs,
    }


def merge_dex_files(dex_files):
    merged_parts = {}
    print('Merging dex files...')
    for dex_file in dex_files:
        parts = parse_dex_file(dex_file)
        for part_name, part_content in parts.items():
            if part_name not in merged_parts:
                merged_parts[part_name] = part_content
            else:
                merged_parts[part_name] += part_content
    return merged_parts


def dex_to_image(merged_parts, width, height):
    byte_array = b''.join(merged_parts.values())
    pixel_array = [(0, 0, 0)] * (width * height)
    print('Converting dex file to image...')
    for i in range(height):
        for j in range(width):
            idx = width * i + j
            rgb_idx = idx * 3
            if rgb_idx + 2 < len(byte_array):
                red = byte_array[rgb_idx]
                green = byte_array[rgb_idx + 1]
                blue = byte_array[rgb_idx + 2]
                pixel_array[idx] = (red, green, blue)
    return pixel2image(pixel_array, width, height)


def pixel2image(pixel_array, width, height):
    image = Image.new("RGB", (width, height))
    image.putdata(pixel_array)
    return image


def save_image(image, apk_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    apk_name = os.path.splitext(os.path.basename(apk_path))[0]
    image_path = os.path.join(save_dir, f"{apk_name}.png")
    image.save(image_path)
    return image_path


def process_single_apk(apk_path):
    extract_to = 'dex'
    width, height = 224, 224
    save_dir = 'img_2_pred'
    subdir = ''

    print('Processing ' + apk_path)
    try:
        dex_files, subdir = extract_dex_files(apk_path, extract_to)
        merged_parts = merge_dex_files(dex_files)
        image = dex_to_image(merged_parts, width, height)
        image_path = save_image(image, apk_path, save_dir)
        return image_path
    except (OSError, zipfile.BadZipFile, IOError) as e:
        print(f"Error processing {apk_path}: {e}")
        return None
    finally:
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
