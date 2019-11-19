from PIL import Image

# Replace with converted image
im = Image.open('TestPic.jpeg')

def crop_center(pil_img):
    minimum = min(pil_img.size)
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - minimum) // 2,
                         (img_height - minimum) // 2,
                         (img_width + minimum) // 2,
                         (img_height + minimum) // 2))

im_new = crop_center(im)
# im_new.save('cropped.jpg', quality=100)
