import cv2
import numpy as np

def is_overlapping(rect1, rect2):  
    x1, y1, x2, y2 = rect1  
    x3, y3, x4, y4 = rect2  
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def draw_bounding_boxes(
        image,
        boxes,
        colors=[(0, 255, 0)],
        texts=[],
        thickness=2,
        lineType=cv2.LINE_AA,
        fontScale=0.5,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        bottomLeftOrigin=False,
):
    """
    在图像上绘制边界框。

    参数:
    image (numpy.ndarray): 输入图像。
    boxes (list[tuple[int, int, int, int]]): 表示边界框坐标的四元组列表。
    colors (list[tuple[int, int, int]], 可选): 边界框颜色，默认为绿色[(0, 255, 0)]
    thickness (int, 可选): 边界框线宽，默认为2。
    lineType (int, 可选): 线条类型，默认为cv2.LINE_AA。
    fontScale (float, 可选): 字体缩放因子，默认为1。
    fontFace (int, 可选): 字体类型，默认为cv2.FONT_HERSHEY_SIMPLEX。
    bottomLeftOrigin (bool, 可选): 如果为True，则表示坐标原点位于图像左下角，否则位于图像左上角。默认为False。
    """
    image_h,image_w=image.shape[0],image.shape[1]
    previous_locations = []
    previous_bboxes=[]
    text_offset = 10
    text_offset_original = 4
    text_size = max(0.07 * min(image_h, image_w) / 100, fontScale)
    text_line = int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = int(max(2 * min(image_h, image_w) / 512, thickness))
    text_height = text_offset # init
    
    assert len(boxes) == len(colors)

    for idx,(box, color_rgb) in enumerate(zip(boxes, colors)):
        # convert rgb to bgr
        color = (color_rgb[2], color_rgb[1], color_rgb[0])
        x1, y1, x2, y2 = map(int,box)
        if bottomLeftOrigin:
            x1, y1 = image.shape[1] - x1 - 1, image.shape[0] - y1 - 1

        new_image = cv2.rectangle(image, (x1, y1), (x2, y2), color, box_line, lineType)
        
        if texts:
            assert len(boxes) == len(texts)
            cur_text = texts[idx]
            (text_width, text_height), _ = cv2.getTextSize(cur_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)  
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - text_height - text_offset_original, x1 + text_width, y1  
            
            for prev_bbox in previous_bboxes:  
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):  
                    text_bg_y1 += text_offset  
                    text_bg_y2 += text_offset  
                    y1 += text_offset 
                    
                    if text_bg_y2 >= image_h:  
                        text_bg_y1 = max(0, image_h - text_height - text_offset_original)  
                        text_bg_y2 = image_h  
                        y1 = max(0, image_h - text_height - text_offset_original + text_offset)  
                        break 
            
            alpha = 0.5  
            for i in range(text_bg_y1, text_bg_y2):  
                for j in range(text_bg_x1, text_bg_x2):  
                    if i < image_h and j < image_w: 
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(color)).astype(np.uint8) 
            
            cv2.putText(  
                new_image, cur_text, (x1, y1 - text_offset_original), fontFace, text_size, (0, 0, 0), text_line, cv2.LINE_AA  
            )  
            previous_locations.append((x1, y1))  
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
    return image
