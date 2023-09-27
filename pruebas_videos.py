import cv2
from fastsam import FastSAM, FastSAMPrompt
import torch
import threading
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_image(image):
        
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
            
            #mask = prompt_process.point_prompt(points=[[320, 250]], pointlabel=[1])
            mask = prompt_process.text_prompt(text='bottle')
            
            return mask
        
        else:
            print("Imagen sigue siendo nula")
            
            return None

def draw_mask(img, generated_mask):
    
    if len(generated_mask) != 0:
    
        masked_image = img.copy()
        
        generated_mask = np.reshape(generated_mask, (1920, 1080, 1))
        
        masked_image = np.where(generated_mask,
                                np.array([0,255,0], dtype='uint8'),
                                masked_image)
        
        masked_image = masked_image.astype(np.uint8)
        
        return cv2.addWeighted(img, 0.3, masked_image, 0.7, 0)
    
    else:
        return img







if __name__ == "__main__":
    
    cap = cv2.VideoCapture('./videos/cuarto.mp4')

    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        mask = process_image(frame)
        if mask is not None:
            masked_final_image = draw_mask(frame, mask)
            cv2.imshow('Frame',masked_final_image)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()