from PIL import Image
from rembg import remove


input = Image.open(r"tmp\processed_1000034474.jpg")
output = remove(input)
output.save(r"tmp\output.jpg")
