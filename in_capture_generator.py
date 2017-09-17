#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random

import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_letter_cases = "abcdefghijklmnopqrstuvwxyz"
_upper_cases = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_numbers = "1234567890"
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))


def generate_verify_image(index,
                          size=(50, 50),
                          chars=init_chars,
                          img_type="png",
                          mode="RGB",
                          bg_color=(0, 0, 0),
                          fg_color=(255, 255, 255),
                          draw_lines=False,
                          n_line=(1, 2),
                          draw_points=False,
                          point_chance=2):

    width, height = size
    img = Image.new(mode, size, bg_color)
    draw = ImageDraw.Draw(img)

    def get_chars():
        """生成给定长度的字符串，返回列表格式"""
        random_int = random.randint(0, 61)
        print(random_int)
        return chars[random_int:random_int+1]

    def create_lines():
        """绘制干扰线"""

        line_num = random.randint(*n_line)

        for i in range(line_num):
            # 起始点
            begin = (random.randint(0, size[0]), random.randint(0, size[1]))
            # 结束点
            end = (random.randint(0, size[0]), random.randint(0, size[1]))
            draw.line([begin, end], fill=(0, 0, 0))

    def create_points():
        """绘制干扰点"""

        chance = min(100, max(0, int(point_chance)))

        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=(0, 0, 0))

    def create_strs():
        """绘制验证码字符"""

        c_chars = get_chars()
        strs = ' %s ' % ' '.join(c_chars)

        font = ImageFont.truetype("D:/INCP/arial.ttf", size=28)

        draw.text((0, 0), strs, fill=fg_color, font=font)

        return ''.join(c_chars)

    if draw_lines:
        create_lines()
    if draw_points:
        create_points()

    capture_char = create_strs()

    # 图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500
              ]
    img = img.transform(size, Image.PERSPECTIVE, params)

    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    img = img.rotate(random.randint(-20, 20), expand=0)

    print(capture_char)
    folder = 'D:/workspace/tmp/IN/Generate/%s_%s/' % (capture_char.lower(), isupper(capture_char))
    os.makedirs(folder, exist_ok=True)
    file_name = folder + '/%d.png'
    img.save(file_name % index, img_type)


def isupper(capture_char):
    if capture_char.isupper():
        return 'u'
    else:
        return 'l'


if __name__ == "__main__":
    for i in range(50):
        print('index', i)
        generate_verify_image(i)
