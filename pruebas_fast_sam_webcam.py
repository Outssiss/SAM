import cv2
from fastsam import FastSAM, FastSAMPrompt
import torch
import threading
import numpy as np

image = None
mask = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_image():
    global image
    global mask
        
    model = FastSAM('./checkpoints/FastSAM-x.pt')

    model.to(device=DEVICE)
    
    while True:

        if image is not None:
            
            everything_results = model(
            source=image,
            device=DEVICE,
            retina_masks=True,
            imgsz=256,
            conf=0.4,
            iou=0.9,
            verbose=False
            )
            
            prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
            
            mask = prompt_process.point_prompt(points=[[320, 250]], pointlabel=[1])
        
        else:
            print("Imagen sigue siendo nula")
        
def draw_mask(img, generated_mask):
    
    if len(generated_mask) != 0:
    
        masked_image = img.copy()
        
        generated_mask = np.reshape(generated_mask, (480, 640, 1))
        
        masked_image = np.where(generated_mask,
                                np.array([0,255,0], dtype='uint8'),
                                masked_image)
        
        masked_image = masked_image.astype(np.uint8)
        
        return cv2.addWeighted(img, 0.3, masked_image, 0.7, 0)
    
    else:
        return img
            
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if __name__ == "__main__":
    
    mask_generation = threading.Thread(target=process_image, daemon=True)
    mask_generation.start()
    
    while True:
            ret, image = cam.read()
            if mask is not None:
                masked_final_image = draw_mask(image, mask)
                cv2.imshow("window", masked_final_image)
            else:
                cv2.imshow("window", image)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                cam.release()
                break