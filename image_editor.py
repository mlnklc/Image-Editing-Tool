from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5 import QtGui
from qcrop.ui import QCrop
import sys
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PIL import Image,ImageEnhance, ImageQt

from PyQt5 import QtCore
class image_editor(QDialog):
    def __init__(self):
        super(image_editor, self).__init__()
        loadUi("image_editor_gui.ui",self)
        self.exist_img=False
        self.image = None
        self.original_image=None
        self.will_change_img=None
        self.temp = None
        self.original_path=None
        self.iconName="kakuna.png"
        self.window_title="Kakuna Image Editor"
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.window_title)
        self.tabWidget.hide()
        self.image_label1.hide()
        self.image_label2.hide()
        self.label.hide()
        self.label_5.hide()

        self.image_label1.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label1.setScaledContents(True)

        self.image_label2.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label2.setScaledContents(True) # to scale image on label
        self.newimg_button.clicked.connect(self.browse_image)
        self.saveimg_button.clicked.connect(self.save_image)
        self.reset_button.clicked.connect(self.reset_img)
        self.flip_button.clicked.connect(self.flipImage)
        self.mirror_button.clicked.connect(self.mirrorImg)
        self.inverse_button.clicked.connect(self.negative)
        self.crop_button.clicked.connect(self.crop)

        self.hand_button.clicked.connect(self.handDraw)
        self.sepia_button.clicked.connect(self.sepia)
        self.black_button.clicked.connect(self.black)
        self.nega_button.clicked.connect(self.negative)
        self.cartoon_button.clicked.connect(self.cartoon)
        self.blackrus_button.clicked.connect(self.black_russian)
        self.weedle_button.clicked.connect(self.greenDemon)
        self.kakuna_button.clicked.connect(self.kakuna_filter)
        self.carnaval_button.clicked.connect(self.carnaval_filter)
        self.head_button.clicked.connect(self.head)
        self.beedril_button.clicked.connect(self.beedril_filter)
        self.avatar_button.clicked.connect(self.avatar_blue)
        self.dither_button.clicked.connect(self.dither)
        self.contrast_button.clicked.connect(self.contrast)
        self.gray_button.clicked.connect(self.gray_pic)
        self.lemon_button.clicked.connect(self.shaky_lemon)
        self.blueShape_button.clicked.connect(self.blueShape)
        self.green_button.clicked.connect(self.greenLife)
        self.wan_button.clicked.connect(self.wan)
        self.davinci_button.clicked.connect(self.daVinci)
        self.burgundy_button.clicked.connect(self.burgundy)
        self.ocean_button.clicked.connect(self.ocean)
        self.magma_button.clicked.connect(self.magma)
        self.greenish_button.clicked.connect(self.greenish)
        self.cotton_button.clicked.connect(self.cotton_candy)
        self.dream_button.clicked.connect(self.dream)
        self.white_button.clicked.connect(self.white_russian)
        self.focus_button.clicked.connect(self.focus)

        self.auto_enchacement.clicked.connect(self.auto_en)
        self.rotate_button.clicked.connect(self.rotate_img)
        self.brightness_slider.valueChanged.connect(self.brightness_img)
        self.contrast_slider.valueChanged.connect(self.change_contrast)
        self.exit_button.clicked.connect(self.quit)

    def browse_image(self):
        #print("browse image")
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.original_image = self.image
        self.update_img2(self.image)

        self.temp = self.image
        self.exist_img = True
        self.tabWidget.show()
        self.image_label1.show()
        self.image_label2.show()
        self.label.show()
        self.label_5.show()
        self.label_6.hide()

    def quit(self):
        QApplication.instance().quit()

    def save_image(self):
        if self.will_change_img:
            options = QFileDialog.Options()
            filter = 'JPEG (*.jpg);;PNG (*.png);;Bitmap (*.bmp)'
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '',
                                                            filter, options=options)
            self.will_change_img.save(self.save_path)
        else:
            QMessageBox.warning(self, 'Saving Error', 'There is no image that have been changed.')

    def reset_img(self):
        if self.exist_img == True:
            self.image_label2.clear()
            self.update_img(self.original_image)
            self.rotate_slider.setValue(0)
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
        else:
            QMessageBox.warning(self, 'Reset Image Error', 'There is no image that can be reset.')

    def rotate_img(self):
        rotate_val = self.rotate_slider.value()
        transform = QTransform().rotate(rotate_val/10)
        pixmap = QPixmap(self.will_change_img)
        rotated = pixmap.transformed(transform, mode=Qt.SmoothTransformation)
        self.will_change_img = QImage(rotated)
        self.image_label2.setPixmap(QPixmap.fromImage(self.will_change_img))

    def auto_en(self):
        clip = 1
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        maximum = accumulator[-1]
        clip *= (maximum / 100.0)
        clip /= 2.0
        minimum_gray = 0
        while accumulator[minimum_gray] < clip:
            minimum_gray += 1
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip):
            maximum_gray -= 1

        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        self.temp = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
        self.update_img(self.temp)

    def mirrorImg(self):
        mirror = QTransform().scale(-1, 1)
        pixmap = QPixmap(self.will_change_img)
        mirrored = pixmap.transformed(mirror)
        self.will_change_img = QImage(mirrored)
        self.image_label2.setPixmap(QPixmap.fromImage(self.will_change_img))

    def reverse(self):
        rever = QTransform().scale(1, -1)
        pixmap = QPixmap(self.image)
        revers = pixmap.transformed(rever)
        self.image = QImage(revers)
        self.image_label2.setPixmap(QPixmap.fromImage(self.image))

    def flipImage(self):
        transform90 = QTransform().rotate(90)
        pixmap = QPixmap(self.will_change_img)
        rotated = pixmap.transformed(transform90, mode=Qt.SmoothTransformation)
        self.will_change_img = QImage(rotated)
        self.image_label2.setPixmap(QPixmap.fromImage(self.will_change_img))

    def change_contrast(self):
        contrast_value = self.contrast_slider.value()
        self.contrast_text.setText(str(contrast_value))
        image = self.cv2PIL(self.original_image)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_value)
        image = self.img_to_cv(image)
        self.update_img(image)

    def brightness_img(self):
        bright_val=self.brightness_slider.value()
        self.brightness_text.setText(str(bright_val))
        image = self.cv2PIL(self.original_image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(bright_val/10)

        image = self.img_to_cv(image)
        self.update_img(image)

    def crop(self):
        pixmap = QPixmap(self.will_change_img)
        crop_tool = QCrop(pixmap)
        status = crop_tool.exec()
        if status == 1:
            cropped_image = crop_tool.image
            self.will_change_img = QImage(cropped_image)
            self.image_label2.setPixmap(QPixmap.fromImage(self.will_change_img))

    def cartoon(self):
        samp = 2
        filternum = 50

        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)

        for _ in range(samp):
            img = cv2.pyrDown(img)

        for _ in range(filternum):
            img = cv2.bilateralFilter(img, 9, 9, 7)

        for _ in range(samp):
            img = cv2.pyrUp(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        (x, y, z) = img.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        self.temp = cv2.bitwise_and(img, img_edge)
        self.update_img(self.temp)

    def black_russian(self):
        img = ImageQt.fromqimage(self.will_change_img)
        image = self.img_to_cv(img)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                red = image[x, y, 2]
                green = image[x, y, 1]
                blue = image[x, y, 0]

                if red > 180:
                    image[x, y, 2] = 200
                else:
                    image[x, y, 2] = 0
                if green > 180:
                    image[x, y, 1] = 200
                else:
                    image[x, y, 1] = 0
                if blue > 180:
                    image[x, y, 0] = 200
                else:
                    image[x, y, 0] = 0

        self.update_img(image)
        return image

    def c_edit(self, img, param=50):

        param = min(max(-100, param), 100)
        c_val = (param + 100.0) / 100.0
        c_val **= 2
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        width, height = img.size
        pixels = img.load()
        for i in range(width):
            for j in range(height):
                r, g, b, a = pixels[i, j]

                r = int(((r / 255.0 - 0.5) * c_val + 0.5) * 255)
                g = int(((g / 255.0 - 0.5) * c_val + 0.5) * 255)
                b = int(((b / 255.0 - 0.5) * c_val + 0.5) * 255)

                r = min(max(r, 0), 255)
                g = min(max(g, 0), 255)
                b = min(max(b, 0), 255)

                pixels[i, j] = r, g, b, a

        return img

    def kakuna_filter(self):
        img = ImageQt.fromqimage(self.will_change_img)
        image = self.c_edit(img)
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                if (i + j) % 2 == 0:
                    red, green, blue, alpha = pixels[i, j]
                    pixels[i, j] = min(100, int(red ** .15 / 255)), \
                                   min(100, int(blue ** .45 / 255)), \
                                   max(75, int(red * green / 255)), \
                                   alpha
        image = self.img_to_cv(image)
        self.update_img(image)

    def head(self):
        image = ImageQt.fromqimage(self.will_change_img)
        image = self.img_to_cv(image)
        filt = image < 128
        image[filt] = 80
        rows, cols, layer = image.shape

        X, Y = np.ogrid[:rows, :cols]
        centrow, centcol = rows / 2, cols / 2
        distance = (X - centrow) ** 2 + (Y - centcol) ** 2
        radius = (rows / 2) ** 2
        circular_mask = (distance > radius)

        image[circular_mask] = 255,
        self.update_img(image)

    def avatar_blue(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        image.point(lambda i: i ^ 0x8B if i < 128 else i)
        image.point(lambda i: i ^ 0x3D if i < 256 else i)
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]
                pixels[i, j] = min(100, int(green * blue / 255)), \
                               min(100, int(blue * red / 255)), \
                               min(255, int(red * green / 255))

        image = self.img_to_cv(image)
        self.update_img(image)
        return image


    def carnaval_filter(self):
        image = ImageQt.fromqimage(self.will_change_img)
        image = self.img_to_cv(image)
        width = image.shape[1]
        height = image.shape[0]
        i = 0
        j = 0
        init_value = 10  # 100
        inc_value = 100  # 100
        x = 0

        for i in range(height):
            for j in range(width):
                image[j:x, x + 10:x + 20, 2] = 80
                image[i:init_value, init_value:x, 0] = 50
                image[x:x + 50, x + 50:x + 150, 1] = 180

                i += 10
                init_value += 30
                x += 30
                j += 10

        self.update_img(image)

    def get_saturation(self, value, quadrant):
        if value > 223:
            return 255
        elif value > 159:
            if quadrant != 1:
                return 255

            return 0
        elif value > 95:
            if quadrant == 0 or quadrant == 3:
                return 255

            return 0
        elif value > 32:
            if quadrant == 1:
                return 255

            return 0
        else:
            return 0


    def burgundy(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img_rgb = self.img_to_cv(image)
        aspect_ratio = img_rgb.shape[1] / img_rgb.shape[0]
        window_width = 500 / aspect_ratio
        image = cv2.resize(img_rgb, (500, int(window_width)))
        img_color = image
        newImage = img_color.copy()
        i, j, k = img_color.shape
        for x in range(i):
            for y in range(j):
                R = img_color[x, y, 2] * 0.125 + img_color[x, y, 1] * 0.102 + img_color[x, y, 0] * 0.135
                G = img_color[x, y, 2] * 0.256 + img_color[x, y, 1] * 0.106 + img_color[x, y, 0] * 0.96
                B = img_color[x, y, 2] * 0.565 + img_color[x, y, 1] * 0.300 + img_color[x, y, 0] * 0.206
                if R > 255:
                    newImage[x, y, 2] = 102
                else:
                    newImage[x, y, 2] = B
                if G > 255:
                    newImage[x, y, 1] = 102
                else:
                    newImage[x, y, 1] = R
                if B > 255:
                    newImage[x, y, 0] = 102
                else:
                    newImage[x, y, 0] = R

        self.update_img(newImage)


    def dither(self):
        image = ImageQt.fromqimage(self.will_change_img)
        width, height = image.size
        img = self.create_image(width, height)
        pixels = img.load()
        for i in range(0, width, 2):
            for j in range(0, height, 2):
                # Get Pixels
                p1 = self.get_pixel(image, i, j)
                p2 = self.get_pixel(image, i, j + 1)
                p3 = self.get_pixel(image, i + 1, j)
                p4 = self.get_pixel(image, i + 1, j + 1)

                red = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
                green = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
                blue = (p1[2] + p2[2] + p3[2] + p4[2]) / 4

                r = [0, 0, 0, 0]
                g = [0, 0, 0, 0]
                b = [0, 0, 0, 0]

                for x in range(0, 4):
                    r[x] = self.get_saturation(red, x)
                    g[x] = self.get_saturation(green, x)
                    b[x] = self.get_saturation(blue, x)

                pixels[i, j] = (r[0], g[0], b[0])
                pixels[i, j + 1] = (r[1], g[1], b[1])
                pixels[i + 1, j] = (r[2], g[2], b[2])
                pixels[i + 1, j + 1] = (r[3], g[3], b[3])

        image = self.img_to_cv(img)
        self.update_img(image)
        return img

    def shaky_lemon(self):
        image = self.dither()
        if image.mode != "RGBA":
            image.convert("RGBA")
        width, height = image.size
        pixels = image.load()

        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]

                pixels[i, j] = min(255, int((green - blue) ** 2 / 128) + red), \
                               min(255, int((red - green) ** 2 / 128) + blue), \
                               min(255, int((blue - red) ** 2 / 128))
        image = image.point(lambda i: int(i ** 2 / 255))
        image = self.img_to_cv(image)
        self.update_img(image)

    def beedril_filter(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGBA":
            image.convert("RGBA")

        width, height = image.size
        pixels = image.load()

        for i in range(0, width, 2):
            for j in range(0, height, 2):
                p1 = self.get_pixel(image, i, j)
                p2 = self.get_pixel(image, i, j + 1)
                p3 = self.get_pixel(image, i + 1, j)
                p4 = self.get_pixel(image, i + 1, j + 1)

                # to grayscale
                gray1 = (p1[0] * 0.299) + (p1[1] * 0.587) + (p1[2] * 0.114)
                gray2 = (p2[0] * 0.299) + (p2[1] * 0.587) + (p2[2] * 0.114)
                gray3 = (p3[0] * 0.299) + (p3[1] * 0.587) + (p3[2] * 0.114)
                gray4 = (p4[0] * 0.299) + (p4[1] * 0.587) + (p4[2] * 0.114)

                # Saturation
                sat = (gray1 + gray2 + gray3 + gray4) / 4

                if sat > 223:
                    pixels[i, j] = (255, 208, 2)
                    pixels[i, j + 1] = (255, 208, 2)
                    pixels[i + 1, j] = (255, 208, 2)
                    pixels[i + 1, j + 1] = (255, 208, 2)
                elif sat > 159:
                    pixels[i, j] = (255, 255, 255)
                    pixels[i, j + 1] = (0, 0, 0)
                    pixels[i + 1, j] = (255, 255, 255)
                    pixels[i + 1, j + 1] = (255, 255, 255)
                elif sat > 95:
                    pixels[i, j] = (182, 66, 30)  # (248, 142, 33) orange
                    pixels[i, j + 1] = (182, 66, 30)
                    pixels[i + 1, j] = (182, 66, 30)
                    pixels[i + 1, j + 1] = (182, 66, 30)
                elif sat > 32:
                    pixels[i, j] = (0, 0, 0)
                    pixels[i, j + 1] = (255, 255, 255)
                    pixels[i + 1, j] = (0, 0, 0)
                    pixels[i + 1, j + 1] = (0, 0, 0)
                else:
                    pixels[i, j] = (0, 0, 0)
                    pixels[i, j + 1] = (0, 0, 0)
                    pixels[i + 1, j] = (0, 0, 0)
                    pixels[i + 1, j + 1] = (0, 0, 0)

        image = self.img_to_cv(image)
        self.update_img(image)

    def negative(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        k = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k.append(img[i, j])

        maxi = np.max(k)
        self.temp = img.copy()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.temp[i, j] = maxi - self.temp[i, j]
        self.update_img(self.temp)

    def black(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGBA":
            image.convert("RGBA")

        width, height = image.size
        pixels = image.load()

        separator = 255 / 1.2 / 2 * 3
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                total = r + g + b
                if total > separator:
                    image.putpixel((x, y), (255, 255, 255))
                else:
                    image.putpixel((x, y), (0, 0, 0))

        image = self.img_to_cv(image)
        self.update_img(image)
        return image

    def wan(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                red = int(r * 0.796 + g * 0.769 + b * 0.189)
                green = int(r * 0.349 + g * 0.686 + b * 0.500)
                blue = int(r * 0.272 + g * 0.563 + b * 0.566)
                image.putpixel((x, y), (red, green, blue))
        image = self.img_to_cv(image)
        self.update_img(image)

    def greenDemon(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()

        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]

                pixels[i, j] = min(255, int((green - blue) ** 2 / 128) + red), \
                               min(255, int((red - green) ** 2 / 128) + blue), \
                               min(255, int((blue - red) ** 2 / 128))
        image = image.point(lambda i: int(i ** 2 / 255))
        image = self.img_to_cv(image)
        self.update_img(image)
        return image

    def sepia(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                red = int(r * 0.393 + g * 0.769 + b * 0.189)
                green = int(r * 0.349 + g * 0.686 + b * 0.168)
                blue = int(r * 0.272 + g * 0.534 + b * 0.131)
                image.putpixel((x, y), (red, green, blue))
        image = self.img_to_cv(image)
        self.update_img(image)

    def greenLife(self):
        image = self.cv2PIL(self.avatar_blue())
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]
                pixels[i, j] = min(255, int((green - blue) ** 2 / 128) + red), \
                               min(255, int((red - green) ** 2 / 128) + blue), \
                               min(255, int((blue - red) ** 2 / 128))
        image = image.point(lambda i: int(i ** 2 / 255))
        image = self.img_to_cv(image)
        self.update_img(image)

    def contrast(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        avg = 0
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                avg += r * 0.299 + g * 0.587 + b * 0.114
        avg /= image.size[0] * image.size[1]

        palette = []
        for i in range(256):
            temp = int(avg + 2 * (i - avg))
            if temp < 0:
                temp = 0
            elif temp > 255:
                temp = 255
            palette.append(temp)

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                image.putpixel((x, y), (palette[r], palette[g], palette[b]))

        image = self.img_to_cv(image)
        self.update_img(image)

    def gray_pic(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                gray = int(r * 0.2126 + g * 0.7152 + b * 0.0722)
                image.putpixel((x, y), (gray, gray, gray))
        image = self.img_to_cv(image)
        self.update_img(image)


    def daVinci(self):
        img = ImageQt.fromqimage(self.will_change_img)
        pixels = list(img.getdata())
        width, height = img.size
        image_array = []
        for j in range(height):
            row = []
            for i in range(width):
                if i < width - 1 and (abs(pixels[j * width + i][0] - pixels[j * width + i + 1][0])
                                      + abs(pixels[j * width + i][1] - pixels[j * width + i + 1][1])
                                      + abs(
                            pixels[j * width + i][2] - pixels[j * width + i + 1][2])) / 3 > 35:

                    row.append((255, 140, i // ((width + 255) // 255)))

                elif j < height - 1 and (abs(pixels[j * width + i][0] - pixels[(j + 1) * width + i][0])
                                         + abs(pixels[j * width + i][1] - pixels[(j + 1) * width + i][1])
                                         + abs(
                            pixels[j * width + i][2] - pixels[(j + 1) * width + i][0])) / 3 > 35:

                    row.append((255, 140, i // ((width + 255) // 255)))

                else:
                    row.append((102,102,255))

            image_array.append(row)
        array = np.array(image_array, dtype=np.uint8)
        new_image = Image.fromarray(array)
        image = self.img_to_cv(new_image)
        self.update_img(image)

    def blueShape(self):
        image = self.cv2PIL(self.black())
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()

        image.point(lambda i: i ^ 0x8B if i < 128 else i)
        image.point(lambda i: i ^ 0x3D if i < 256 else i)
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]
                pixels[i, j] = min(100, int(green * blue / 255)), \
                               min(100, int(blue * red / 255)), \
                               min(255, int(red * green / 255))

        image = self.img_to_cv(image)
        self.update_img(image)

    def handDraw(self):
        image = ImageQt.fromqimage(self.will_change_img)
        imga = self.img_to_cv(image)
        gray = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray)
        smooth = cv2.GaussianBlur(invert, (21,21),sigmaX=0, sigmaY=0)

        def dodge(x, y):
            return cv2.divide(x, 255 - y, scale=256)
        img = dodge(gray, smooth)
        self.update_img(img)

    def ocean(self):
        red = 50
        green = 70
        blue = 180
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        height = img.shape[0]
        width = img.shape[1]
        color_rgb = (blue, green, red)
        overlay = np.full((height, width, 3), color_rgb, dtype='uint8')
        cv2.addWeighted(overlay, 1, img, 1.0, 0, img)
        ocean = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.update_img(ocean)

    def magma(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        kernel = np.array([[1, 1, 0],
                           [1, 0, -1],
                           [0, -1, -1]])
        new_image = cv2.filter2D(img, -1, kernel)
        new_image = cv2.applyColorMap(new_image, cv2.COLORMAP_HOT)
        self.update_img(new_image)

    def greenish(self):
        image = ImageQt.fromqimage(self.will_change_img)
        if image.mode != "RGB":
            image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                red = int(r * 0.0025)
                green =int(g * 0.8852)
                blue = int(b * 0.0018)
                image.putpixel((x, y), (red, green, blue))
        image = self.img_to_cv(image)
        self.update_img(image)

    def cotton_candy(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        b, g, r = cv2.split(img)
        b = b * 1.35
        g = g * 0.5
        r = r * 2.25
        rbr_img = cv2.merge((r, g, b))
        image = cv2.convertScaleAbs(rbr_img, alpha=1.2, beta=-20)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        self.update_img(image)

    def dream(self):
        image = ImageQt.fromqimage(self.will_change_img)
        img = self.img_to_cv(image)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        b, g, r = cv2.split(img)
        rbr_img = cv2.merge((r, b, g))
        morphology = cv2.morphologyEx(rbr_img, cv2.MORPH_OPEN, element)
        canvas = cv2.normalize(morphology, None, 20, 255, cv2.NORM_MINMAX)
        new_image = cv2.stylization(canvas, sigma_s=60, sigma_r=0.6)
        self.update_img(new_image)

    def white_russian(self):
        image = ImageQt.fromqimage(self.will_change_img)
        width, height = image.size
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                red, green, blue = pixels[i, j]
                if red < 20:
                    red = 0
                else:
                    red = 200
                if green < 20:
                    green = 0
                else:
                    green = 200
                if blue < 20:
                    blue = 0
                else:
                    blue = 200
                pixels[i, j] = (int(red), int(green), int(blue))
        image = self.img_to_cv(image)
        self.update_img(image)


    def focus(self):
        image = ImageQt.fromqimage(self.will_change_img)
        image = self.img_to_cv(image)
        rows, cols = image.shape[:2]
        X_kernel = cv2.getGaussianKernel(cols, 200)
        Y_kernel = cv2.getGaussianKernel(rows, 200)
        kernel = Y_kernel * X_kernel.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        output = np.copy(image)
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        self.update_img(output)

    def get_pixel(self, image, i, j):
        width, height = image.size
        if i > width or j > height:
            return None

        pixel = image.getpixel((i, j))
        return pixel

    def create_image(self, i, j):
        image = Image.new("RGB", (i, j), "white")
        return image

    def img_to_cv(self, image):
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()
        return cv_image

    def cv2PIL(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        return pil_image

    def update_img2(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.will_change_img = image
        self.image_label1.setPixmap(QtGui.QPixmap.fromImage(image))

    def update_img(self,image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.will_change_img = image.copy()
        self.image_label2.setPixmap(QtGui.QPixmap.fromImage(image))

run = QApplication(sys.argv)
panel=image_editor()
panel.show()
run.exec_()