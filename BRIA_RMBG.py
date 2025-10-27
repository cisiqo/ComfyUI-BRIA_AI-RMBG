
import torch, os
import torch.nn.functional as F
from PIL import Image
from .briarmbg import BriaRMBG
from torchvision.transforms.functional import normalize
import numpy as np
from .blurfusion_foreground_estimation import FB_blur_fusion_foreground_estimator_2

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BRIA_RMBG_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ("RMBGMODEL",)
    RETURN_NAMES = ("rmbgmodel",)
    FUNCTION = "load_model"
    CATEGORY = "🧹BRIA RMBG"
  
    def load_model(self):
        net = BriaRMBG()
        model_path = os.path.join(current_directory, "RMBG-1.4/model.pth")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval() 
        return [net]


class BRIA_RMBG_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rmbgmodel": ("RMBGMODEL",),
                "image": ("IMAGE",),
            },
           "optional": {
                "edge_optimization": ("BOOLEAN", {"default": True}),
                "blur_ksize": ("INT", {"default": 90, "min": 0, "max": 90, "step": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "remove_background"
    CATEGORY = "🧹BRIA RMBG"
  
    def remove_background(self, rmbgmodel, image, edge_optimization=True, blur_ksize=90):
        processed_images = []
        processed_masks = []

        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            result=rmbgmodel(im_tensor)
            result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            
            if edge_optimization:
                new_im = refine_foreground(orig_image, pil_im, r=blur_ksize)
            else:
                new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
                new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks

def refine_foreground(image: Image.Image, mask: Image.Image, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    alpha_array = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(
        image_array, alpha_array, r=r
    )
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    image_masked.putalpha(mask.resize(image.size))
    return image_masked
      

NODE_CLASS_MAPPINGS = {
    "BRIA_RMBG_ModelLoader_Zho": BRIA_RMBG_ModelLoader_Zho,
    "BRIA_RMBG_Zho": BRIA_RMBG_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BRIA_RMBG_ModelLoader_Zho": "🧹BRIA_RMBG Model Loader",
    "BRIA_RMBG_Zho": "🧹BRIA RMBG",
}
