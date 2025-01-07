import torch 
from torchvision.transforms import v2 


def get_transforms(train=True): 
    
    if train: 
        transforms = v2.Compose([
            v2.ToImage() , 
            v2.ToDtype(torch.uint8 , scale=True),
            v2.Resize((256,256)) ,
            v2.RandomHorizontalFlip(p=0.5) ,
            v2.RandomRotation(degrees=5), 
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage() , 
            v2.ToDtype(torch.uint8 , scale=True),
            v2.Resize((256,256)) ,
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return transforms