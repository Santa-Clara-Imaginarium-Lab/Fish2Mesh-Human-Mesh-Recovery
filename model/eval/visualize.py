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
    faceArray = np.load('./model_faces.npy')
    renderer = Renderer(faces=faceArray)
    # camera_translation = GT_cam.numpy().copy()
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    LIGHT_RED = (1.0, 0.6, 0.6)  # Light red color (RGB)

    root = '/media/imaginarium/12T/TIANMA/visualResult/'

    # person_25_frame_24055 person_19_frame_769 person_12_frame_8813
    exampleName = 'person_12_frame_8813'

    models = ['our', '4Dhuman', 'egohmr']

    for model in models:

        ourResults = root + exampleName + '/' + model + '/output_data.pkl'

        # Open and load the pickle file
        with open(ourResults, 'rb') as file:
            data = pickle.load(file)

        if model != 'egohmr':

            GT_npy = data['GT_npy']
            GT_cam = GT_npy['pred_cam_t'].float().view(-1, 3)
            GT_vertices = GT_npy['pred_vertices'].float().view(-1, 6890, 3).numpy()

            regression_img = renderer(GT_vertices[0],  # GT_npy['pred_vertices'][0][0]
                                      GT_cam[0],  # GT_cam[0].detach().cpu().numpy()
                                      torch.zeros((3, 512, 512), dtype=torch.float32).cpu(),  # Images[0].cpu(),
                                      mesh_base_color=LIGHT_RED,
                                      scene_bg_color=(1, 1, 1),
                                      )
            plt.axis('off')
            plt.imshow(regression_img)
            plt.savefig(root + exampleName + '/' + model + '/GT.png')


        pred_vertices = data['pred_vertices']
    # visualMeshPoint(pred_vertices)


        # tmesh = renderer.vertices_to_trimesh(GT_vertices.cpu().numpy()[0], camera_translation, LIGHT_BLUE)
        plt.clf()
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
        plt.savefig(root + exampleName + '/' + model + '/pred.png')
    # plt.show()



        print("Image saved as 'regression_image.png'")
