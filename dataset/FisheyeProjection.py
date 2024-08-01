# from google.colab import drive
import os
import subprocess

# mount google drive
# drive.mount('/content/drive')

# path to your folder in google drive
input_folder = '/media/imaginarium/12T/Dataset/temp_headset_videos/'
output_folder = '/media/imaginarium/12T/Dataset/temp_convert_headset_videos/'

# get list of all .insv files in the input folder
# I download the insv file from the google drive , the video auto convert to mp4
input_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]


for input_file in input_files:
    file_base = os.path.splitext(input_file)[0]
    file_number = file_base.split('_')[-1]

    # create a temporary output file path
    temp_output1 = os.path.join(output_folder, f'{file_base}_temp1.mp4')
    temp_output2 = os.path.join(output_folder, f'{file_base}_temp2.mp4')
    final_output = os.path.join(output_folder, f'processed_headset_{file_number}.mp4')
    # print(temp_output1)
    print(temp_output2)
    print(final_output)

    # 1st command: convert .insv to .mp4
    # command1 = [
    #     'ffmpeg', '-i', os.path.join(input_folder, input_file),
    #     '-c', 'copy', temp_output1
    # ]
    # subprocess.run(command1)
    # print("first processing done!")

    # 2nd command: convert .mp4 to panoramic .mp4
    # command2 = [
    #     'ffmpeg', '-i', os.path.join(input_folder, input_file),
    #     '-vf', 'v360=input=dfisheye:output=equirect:ih_fov=180:iv_fov=180,scale=1920:1640',
    #     temp_output2
    # ]
    command2 = [
        'ffmpeg', '-i', os.path.join(input_folder, input_file),
        '-vf', 'v360=input=fisheye:output=equirect:ih_fov=180:iv_fov=180,transpose=2',
        temp_output2
    ]
    # v360=fisheye:equirect:ih_fov=360:iv_fov=360

    subprocess.run(command2)
    print("second processing done!")

    # 3rd command: crop the panoramic video
    # command3 = [
    #     'ffmpeg', '-i', temp_output2,
    #     '-filter:v', 'crop=800:1280:480:120', '-c:a', 'copy',
    #     final_output
    # ]
    command3 = [
        'ffmpeg', '-i', temp_output2,
        '-filter:v', 'crop=2000:2000:440:1880', '-c:a', 'copy',
        final_output
    ]
    subprocess.run(command3)
    print("final processing done!")

    # remove temporary files
    # os.remove(temp_output1)
    os.remove(temp_output2)

print("done. all files processed!")
