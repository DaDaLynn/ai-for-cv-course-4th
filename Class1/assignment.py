#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np

class DataAugmentation:
	def __init__(self, img_path = r'.\Eddie.jpg'):
		self.img = cv2.imread(img_path)
		self.shape = self.img.shape
		
	def show(self):
		cv2.imshow("image", self.img)
		key = cv2.waitKey()
		if key == 27:
			cv2.destroyAllWindows()
			
	def crop(self, scale):
		img_crop = self.img[0:int(self.shape[0] * scale), 0:int(self.shape[1] * scale)]
		return img_crop
		
	def get_single_channel(self, ch = 0):
		if self.shape[2] == 1:
			return self.img
		channel_data = list(cv2.split(self.img))
		return channel_data[ch]
		
	def random_modify_color(self):
		if self.shape[2] == 1:
			return self.img
		else:			
			for input in list(cv2.split(self.img)):
				value = random.randint(-50, 50)
				if value == 0:
					pass
				elif value > 0:
					thre = 255 - value
					input[input >= thre] = 255
					input[input < thre] = input[input < thre] + value
				else:
					thre = 0 - value
					input[input < thre] = 0
					input[input > thre] = input[input > thre] + value
	
	def img_rotate(self, angle, scale):
		trans_kernel = cv2.getRotationMatrix2D((self.shape[1] / 2, self.shape[0] / 2), angle, scale)
		img_trans = cv2.warpAffine(self.img, trans_kernel, (self.shape[1], self.shape[0]))
		return img_trans
		
	def img_affine_trans(self, pt1, pt2):
		M = cv2.getAffineTransform(pt1, pt2)
		img_trans = cv2.warpAffine(self.img, M, (self.shape[1], self.shape[0]))
		return img_trans
		
	def random_warp(self):
		height, width, channels = self.img.shape
		
		random_margin = 60
		x1 = random.randint(-random_margin, random_margin)
		y1 = random.randint(-random_margin, random_margin)
		x2 = random.randint(width - random_margin - 1, width - 1)
		y2 = random.randint(-random_margin, random_margin)
		x3 = random.randint(width - random_margin - 1, width - 1)
		y3 = random.randint(height - random_margin - 1, height - 1)
		x4 = random.randint(-random_margin, random_margin)
		y4 = random.randint(height - random_margin - 1, height - 1)

		dx1 = random.randint(-random_margin, random_margin)
		dy1 = random.randint(-random_margin, random_margin)
		dx2 = random.randint(width - random_margin - 1, width - 1)
		dy2 = random.randint(-random_margin, random_margin)
		dx3 = random.randint(width - random_margin - 1, width - 1)
		dy3 = random.randint(height - random_margin - 1, height - 1)
		dx4 = random.randint(-random_margin, random_margin)
		dy4 = random.randint(height - random_margin - 1, height - 1)

		pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
		M_warp = cv2.getPerspectiveTransform(pts1, pts2)
		img_warp = cv2.warpPerspective(self.img, M_warp, (width, height))
		return M_warp, img_warp
		
	def hist_equalize(self):
		height, width, channels = self.img.shape
		if channels == 1:
			return cv2.equalizeHist(self.img)
		else:
			img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
			return img_gray
		
	def gamma_adjust(self, gamma=1.0):
		inv_gamma = 1 / gamma
		table = []
		for i in range(256):
			table.append((i / 255) ** inv_gamma * 255)
		table = np.array(table).astype("uint8")
		return cv2.LUT(self.img, table)
		
if __name__ == '__main__':
	print("main")
	test_case = DataAugmentation()
	test_case.show()
	img1 = test_case.crop(0.8)
	img2 = test_case.gamma_adjust(2)
	img3 = test_case.get_single_channel(1)
	img4 = test_case.hist_equalize()
	img5 = test_case.img_rotate(30, 0.5)
	img6 = test_case.random_modify_color()
	img7 = test_case.random_warp()