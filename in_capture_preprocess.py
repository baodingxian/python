import fileinput
import glob
import os

import cv2
import numpy
import numpy as np
from PIL import Image


def ocr(s_input, s_output, text_path):
    preproccess(s_input, s_output, text_path)


def preproccess(s_input, s_output, text_path):
    cache = {}
    for line in fileinput.input(text_path):
        line = line.strip()
        split = line.split(':')
        cache[split[0]] = split[1]

    inputs = glob.glob(s_input + "/*.png")
    if len(inputs) > 0:
        print("Found input image %d" % len(inputs))
        count = 1
        for i in inputs:
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            _, img_th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            img_mb = cv2.medianBlur(img_th, 3)

            img_mb, contours, hierarchy = cv2.findContours(img_mb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = hierarchy[0]

            img_mb = cv2.cvtColor(img_mb, cv2.COLOR_GRAY2RGB)
            bbox = []
            for c in zip(contours, hierarchy):
                contour = c[0]
                hierarchy = c[1]

                if hierarchy[2] == -1 and hierarchy[3] > 0:
                    rect = cv2.minAreaRect(contour)
                    r_box = cv2.boxPoints(rect)
                    r_box = np.int32(r_box)

                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 5 and h > 5:
                        cv2.rectangle(img_mb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        cv2.drawContours(img_mb, [r_box], 0, (0, 0, 255), 1)
                        bbox.append([x, y, w, h, r_box])

            if len(bbox) == 5:
                bbox = sorted(bbox, key=lambda k: k[0])

                basename = os.path.basename(i)
                print(basename)
                cv2.imwrite(s_output + '/' + basename, img_mb)

                name = os.path.splitext(basename)[0]
                output_name = s_output + '/' + name
                os.makedirs(output_name, exist_ok=True)
                os.makedirs("D:/workspace/tmp/IN/LaNetDataSet", exist_ok=True)
                index = 0
                img_th = cv2.bitwise_not(img_th)
                for box in bbox:
                    stencil = numpy.zeros(img_th.shape).astype(img_th.dtype)
                    contours = [box[4]]
                    color = [255, 255, 255]
                    cv2.fillPoly(stencil, contours, color)
                    result = cv2.bitwise_and(img_th, stencil)

                    bbox_img = result[box[1] - 2:box[1] + box[3] + 2, box[0] - 2:box[0] + box[2] + 2]
                    pil_img = Image.fromarray(bbox_img)
                    print (pil_img);
                    with Image.new('L', pil_img.size, 0) as image:
                        image.paste(pil_img, (0, 0))
                        bbox_img = numpy.array(image)
                        print(str(index))
                        cv2.imwrite(output_name + '/' + cache[name][index:index + 1] + '_' + str(index) + '.jpg', bbox_img)
                        cv2.imwrite('D:/workspace/tmp/IN/LaNetDataSet/'+ cache[name][index:index + 1] + '_' + name + '.jpg', bbox_img)
                        index += 1
                count += 1
        print(count)
    else:
        print("No input image found")


if __name__ == '__main__':
    ocr('D:/workspace/tmp/IN/image_in', 'D:/workspace/tmp/IN/image_out', 'D:/workspace/tmp/IN/1.txt')
