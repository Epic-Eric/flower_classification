import json
import model
from PIL import Image
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        # Resize the image
        if im.size[0] > im.size[1]:
            im.thumbnail((256, 256 * im.size[0] / im.size[1]))
        else:
            im.thumbnail((256 * im.size[1] / im.size[0], 256))
        
        # Crop the center of the image
        left = (im.width - 224) / 2
        top = (im.height - 224) / 2
        right = left + 224
        bottom = top + 224
        im = im.crop((left, top, right, bottom))
        
        # Convert image to numpy array
        np_image = np.array(im) / 255.0
        
        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        
        # Reorder dimensions
        np_image = np_image.transpose((2, 0, 1))
        print(np_image.shape)
        
        return np_image
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    plt.show()
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("mps" if torch.backends.mps.is_built() and args.gpu else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.tensor(image).float().to(device)
        image = image.unsqueeze(0)
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p, top_class
    
def plot_result(probs, classes):
    probs = probs.cpu().numpy().squeeze()
    classes = classes.cpu().numpy().squeeze()
    
    # Invert the class_to_idx dictionary to get a mapping from index to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_names = [cat_to_name[idx_to_class[cls]] for cls in classes]
    
    imshow(torch.tensor(new_img))
    fig, ax = plt.subplots()
    y_pos = np.arange(len(probs))
    
    # Sort the probabilities and class names
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    
    ax.barh(y_pos, sorted_probs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_class_names)
    ax.invert_yaxis()  # Invert y-axis to have the most popular category at the top
    plt.show()
    print("=====================================")
    print(f"It is a {class_names[0]} with a probability of {probs[0] * 100:.2f}%")  


if __name__ == "__main__":
    # Argument parsing
    image_path = "" # flowers/external/Erica_tetralix_002.jpg
    checkpoint = "" # checkpoints/vgg19_checkpoint.pth
    if (len(sys.argv) > 2):
        image_path = sys.argv[1]
        checkpoint = sys.argv[2]
    else:
        print("Please provide image_path and checkpoint as argument")
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[3:]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--top_k', type=int, default=5)
    argparser.add_argument('--gpu', action='store_true')
    args = argparser.parse_args()

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load model
    model = model.load_filepath(checkpoint)

    # Show image
    new_img = process_image(image_path)
    imshow(torch.tensor(new_img))

    # Predict & Display an image along with the top 5 classes
    probs, classes = predict(image_path, model, args.top_k)
    plot_result(probs, classes)
    ## "It is a pincushion flower with a probability of 39.53%"
