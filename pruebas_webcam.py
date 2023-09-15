import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import numpy as np
import torch
import threading

image = None
new_image = 0
masks = None

device = "cuda" if torch.cuda.is_available() else "cpu"

#Carga del checkpoint del modelo
sam = sam_model_registry["vit_l"](checkpoint="checkpoints/sam_vit_l_0b3195.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

input_point = np.array([[320, 250]])
input_label = np.array([1])

def process_image():
    
    while True:
        
        global image
        global masks
        
        if image is not None:
            img_array = np.asarray(image)
            predictor.set_image(img_array)
            masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label)
        else:
            print("Imagen sigue siendo nula")
            
#Dibujar la máscara sobre la imagen original, quedandonos con los pixeles de la máscara o imagen original
def draw_mask(img, generated_mask):
    masked_image = img.copy()
    
    generated_mask = np.reshape(generated_mask, (480, 640, 1))
    
    x = np.zeros(shape=(480, 640, 3))
    
    masked_image = np.where(generated_mask,
                            np.array([0,255,0], dtype='uint8'),
                            masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    
    return cv2.addWeighted(img, 0.3, masked_image, 0.7, 0)
    
            


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if __name__ == "__main__":
    
    x = threading.Thread(target=process_image, daemon=True)
    x.start()
    
    while True:
        ret, image = cam.read()
        if masks is not None:
            masked_final_image = draw_mask(image, masks[0])
            cv2.imshow("window", masked_final_image)
        else:
            cv2.imshow("window", image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            cam.release()
            break
        
    

