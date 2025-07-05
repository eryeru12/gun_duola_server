from PIL import Image
from rembg import remove


input = Image.open(r"tmp\a8773912b31bb0512deea5b7247adab44bede063.png")
output = remove(input)
output.save(r"tmp\output.png")
