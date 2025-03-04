import pickle
import numpy as np
from model.util.renderer import Renderer
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def visualMeshPoint(vertices):
    # Assuming pred_vertices is your (6890, 3) array
    # pred_vertices shape should be (6890, 3) with x, y, z coordinates

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from pred_vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Plot the points in 3D
    ax.scatter(x, y, z, s=1)  # s=1 is the size of each point

    # Optional: You can set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    print('Start')
    root = '/media/imaginarium/12T/TIANMA/visualResult/'

    exampleName = 'person_19_frame_769'

    ourResults = root + exampleName + '/our/output_data.pkl'

    # Open and load the pickle file
    with open(ourResults, 'rb') as file:
        data = pickle.load(file)

    # Print the loaded data to check its contents
    # print(data)
    GT_npy = data['GT_npy']
    GT_cam = GT_npy['pred_cam_t'].float().view(-1, 3)
    GT_vertices = GT_npy['pred_vertices'].float().view(-1, 6890, 3)
    # Convert NumPy arrays to PyTorch tensors
    # GT_cam_tensor = torch.tensor(GT_cam).float()
    # GT_vertices_tensor = torch.tensor(GT_vertices).float()

    pred_vertices = data['pred_vertices']
    # visualMeshPoint(pred_vertices)

    faceArray = np.load('./model_faces.npy')
    renderer = Renderer(faces=faceArray)
    # camera_translation = GT_cam.numpy().copy()
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    LIGHT_RED = (1.0, 0.6, 0.6)  # Light red color (RGB)
    # tmesh = renderer.vertices_to_trimesh(GT_vertices.cpu().numpy()[0], camera_translation, LIGHT_BLUE)

    regression_img = renderer(pred_vertices,  # GT_npy['pred_vertices'][0][0]
                              GT_cam[0],  # GT_cam[0].detach().cpu().numpy()
                              torch.zeros((3, 512, 512), dtype=torch.float32).cpu(),  # Images[0].cpu(),
                              mesh_base_color=LIGHT_BLUE,
                              scene_bg_color=(1, 1, 1),
                              )
    # regression_img = (regression_img * 255).astype(np.uint8)
    # print(f"Min value: {regression_img.min()}, Max value: {regression_img.max()}")

    # Save the image using matplotlib
    plt.imshow(regression_img)
    plt.axis('off')  # Hide axis
    plt.savefig(root + exampleName + '/our/pred.png')
    plt.show()

    # Construct the path to the image
    image_path = root + exampleName + '/our/pred.png'

    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to RGB if it's not already (ensure 3 channels)
    image = image.convert('RGB')

    # Resize the image to (512, 512) if needed
    image = image.resize((512, 512))

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Normalize the image (optional) to [0, 1] range, if needed (assuming values in [0, 255])
    image_np = image_np.astype(np.float32) / 255.0

    # Convert to PyTorch tensor and reorder dimensions to (C, H, W)
    regression_img = torch.tensor(image_np).permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)

    # Ensure the tensor has the correct dtype (float32)
    regression_img = regression_img.float()


    regression_img = renderer(GT_vertices,  # GT_npy['pred_vertices'][0][0]
                              GT_cam[0],  # GT_cam[0].detach().cpu().numpy()
                              torch.zeros((3, 512, 512), dtype=torch.float32).cpu(),  # Images[0].cpu(),
                              mesh_base_color=LIGHT_RED,
                              scene_bg_color=(1, 1, 1),
                              )

    plt.imshow(regression_img)
    plt.axis('off')


print("Image saved as 'regression_image.png'")
