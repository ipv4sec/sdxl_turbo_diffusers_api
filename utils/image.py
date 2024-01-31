# -*- coding: utf-8 -*-
import io
import base64

def image_to_base64(image, format="JPEG"):
    # 将图像数据编码为 Base64 字符串
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # 这里可以根据实际情况选择图像格式
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_data