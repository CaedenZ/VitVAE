from PIL import Image
import os

for name in os.listdir('blackened'):
    
    image = Image.open('blackened/' + name)
    # print(f"Original size : {image.size}") # 5464x3640

    sunset_resized = image.resize((384, 384))
    sunset_resized.save('resized384/'+ name)