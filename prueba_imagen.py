from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open("images/cuarto.jpg")
image.save("images/cuarto.jpg", quality=95)
image_array = np.asarray(image)


sam = sam_model_registry["vit_l"](checkpoint="checkpoints/sam_vit_l_0b3195.pth")

sam.to(device=device)

input_point = np.array([[307, 2909]])
input_label = np.array([1])

predictor = SamPredictor(sam)
predictor.set_image(image_array)
masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label)

i = 0
for mask in masks:
    plt.figure(figsize=(40,40))
    plt.imshow(image)
    print(scores[i])
    i =+ 1
    show_mask(mask, plt.gca())
    plt.scatter(input_point[0][0], input_point[0][1], c='red')
    plt.axis('on')
    plt.show()

