from PIL import Image

def save_image(img_tensor, save_path):
        from PIL import Image
        img = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
        img.save(save_path)