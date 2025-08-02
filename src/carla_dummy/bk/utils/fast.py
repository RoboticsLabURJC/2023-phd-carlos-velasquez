import cv2
import numpy as np
import torch
from fastseg import MobileV3Small  # Asegúrate de que esta importación es correcta según tu modelo real

class LaneDetector:
    def __init__(self, model_path="/home/canveo/Documents/carla_laneddetection/fastai_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(model_path, self.device)

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image   

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def _predict(self, img):
        with torch.no_grad():
            image_tensor = img.transpose(2, 0, 1).astype('float32') / 255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    def detect(self, img_array):
        model_output = self._predict(img_array)
        background, left, right = model_output[0, 0, :, :], model_output[0, 1, :, :], model_output[0, 2, :, :] 
        return background, left, right

    def fit_poly(self, probs):
        # Aquí debes definir `self.cut_v` y `self.grid` correctamente si los necesitas
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0., 0., 0., 0.])
        return np.poly1d(coeffs)

    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_imagefile_to_array(image)
        left_poly, right_poly, _, _ = self.get_fit_and_probs(image)
        return left_poly, right_poly

    def get_fit_and_probs(self, img):
        _, left, right = self.detect(img)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

def load_model(model_path, device):
    # Load the entire model from the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the model architecture
    model = MobileV3Small(num_classes=3, use_aspp=True, num_filters=8)

    # Check if 'state_dict' key exists in the checkpoint, assuming it is saved with torch.save({'state_dict': model.state_dict()})
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # If the checkpoint is not a dictionary containing 'state_dict', it might be the state_dict itself
        state_dict = checkpoint

    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    
    # Move the model to the appropriate device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Verifica que el directorio 'code' esté en el PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    code_directory = os.path.join(project_root, 'code')
    if code_directory not in sys.path:
        sys.path.append(code_directory)

    lane_detector = LaneDetector(model_path="/home/canveo/Documents/carla_laneddetection/fastai_model.pth")
    left_poly, right_poly = lane_detector.detect_from_file(image_path)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img_rgb.shape
    y_values = np.linspace(height // 2, height - 1, num=100)

    left_x_values = left_poly(y_values)
    right_x_values = right_poly(y_values)

    for x, y in zip(left_x_values, y_values):
        cv2.circle(img_rgb, (int(x), int(y)), 2, (255, 0, 0), -1)

    for x, y in zip(right_x_values, y_values):
        cv2.circle(img_rgb, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imshow('Detected Lanes', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
