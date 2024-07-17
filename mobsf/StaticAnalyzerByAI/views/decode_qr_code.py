from django.http import JsonResponse
import cv2
from pyzbar.pyzbar import decode
import numpy as np


def decode_qr_code_view(request):
    if request.method == 'POST' and request.FILES.get('qr_code'):
        qr_code_file = request.FILES['qr_code']
        qr_code_data = np.frombuffer(qr_code_file.read(), np.uint8)
        image = cv2.imdecode(qr_code_data, cv2.IMREAD_COLOR)

        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 解码二维码（不反转）
        decoded_objects_normal = decode(gray_image)

        # 反转颜色
        inverted_image = cv2.bitwise_not(gray_image)

        # 解码二维码（反转后）
        decoded_objects_inverted = decode(inverted_image)

        # 提取并返回 URL
        urls = [obj.data.decode('utf-8') for obj in decoded_objects_normal + decoded_objects_inverted]
        if urls:
            return JsonResponse({'status': 'success', 'urls': urls})
        else:
            return JsonResponse({'status': 'error', 'message': '未检测到二维码。'})
    return JsonResponse({'status': 'error', 'message': '请求方法不正确或未提供二维码图片。'})
