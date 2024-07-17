import cv2
from pyzbar.pyzbar import decode
import sys


def decode_qrcode(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 使用 pyzbar 解码二维码
    decoded_objects = decode(img)

    if not decoded_objects:
        print("未找到二维码")
        return None

    for obj in decoded_objects:
        # 返回二维码数据
        return obj.data.decode('utf-8')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python decode_qrcode.py <图像文件路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = decode_qrcode(image_path)
    if result:
        print(f"二维码内容: {result}")
    else:
        print("无法解码二维码")
