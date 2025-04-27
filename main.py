import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def text_detection(image_path, output_path=None, visualize=False):
    """
    识别图片中的文字并返回文字内容及其位置坐标
    
    参数:
        image_path: 图片路径
        output_path: 可选，可视化结果保存路径
        visualize: 是否显示可视化结果
    
    返回:
        list: 包含字典的列表，每个字典包含文字内容和其位置信息
    """
    # 读取图片
    pytesseract.pytesseract.tesseract_cmd = r"tesseract.exe路径"
    # https://tesseract-ocr.github.io/
    # https://github.com/tesseract-ocr/tesseract
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图片，请检查路径是否正确")
    
    # 转换为RGB (OpenCV使用BGR，但Tesseract需要RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用pytesseract获取文字和位置信息
    results = pytesseract.image_to_data(rgb_image, output_type=Output.DICT, lang='chi_sim+eng')
    
    # 提取有用信息
    text_info = []
    for i in range(len(results['text'])):
        # 只处理置信度大于0的文字
        if int(results['conf'][i]) > 0:
            text = results['text'][i]
            x = results['left'][i]
            y = results['top'][i]
            w = results['width'][i]
            h = results['height'][i]
            
            text_info.append({
                'text': text,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': int(results['conf'][i])
            })
    
    # 如果需要可视化结果
    if visualize or output_path:
        # 使用PIL绘制结果，因为PIL支持中文显示更好
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        
        # 加载字体 (Windows下可以使用simsun.ttc)
        try:
            font = ImageFont.truetype("simsun.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        for info in text_info:
            # 绘制矩形框
            draw.rectangle(
                [(info['x'], info['y']), 
                 (info['x'] + info['width'], info['y'] + info['height'])],
                outline="red",
                width=2
            )
            
            # 绘制文字
            draw.text(
                (info['x'], info['y'] - 15),
                f"{info['text']} ({info['confidence']}%)",
                fill="red",
                font=font
            )
        
        if visualize:
            pil_image.show()
        
        if output_path:
            pil_image.save(output_path)
    
    return text_info

# 使用示例
if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = r"请填写图片途径"
    
    # 识别文字并获取坐标
    detected_texts = text_detection(image_path, output_path="output.jpg", visualize=True)
    
    # 打印结果
    print("检测到的文字及其坐标:")
    for i, text_info in enumerate(detected_texts, 1):
        print(f"{i}. 文字: '{text_info['text']}'")
        print(f"   位置: 左上角({text_info['x']}, {text_info['y']}), "
              f"宽度: {text_info['width']}, 高度: {text_info['height']}, "
              f"置信度: {text_info['confidence']}%")
        print()
        
