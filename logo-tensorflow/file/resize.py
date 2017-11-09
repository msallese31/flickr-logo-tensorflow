from PIL import Image
import os, sys

def resize_all_in_dir(path, size):
	dirs = os.listdir( path )
	for item in dirs:
		if os.path.isfile(path+item):
			im = Image.open(path+item)
			f, e = os.path.splitext(path+item)
			imResize = im.resize((size,size), Image.ANTIALIAS)
			imResize.save(f + '-resized-small.jpg', 'JPEG', quality=100)

def main():
	resize_all_in_dir("/home/sallese/flickr-logo-tensorflow/logo-tensorflow/images/", 64)

if __name__== "__main__":
  main()