import os
#import ffmpeg
import json
import torch
import pickle
import cv2
from torchvision import transforms
'''
Downsample from 398x224 at 25fps to 112x112 at 25fps
'''

#create list of goal times in the game from labels
def create_goal_times(game_label):
    goals = []
    for i in range(len(game_label)):
        label_dict = game_label[i]
        if label_dict["label"] == "soccer-ball":
            # ie label_dict["gameTime"] = "1 - 13:10"
            gameTime = label_dict["gameTime"].split(' ') 
            half = gameTime[0]
            time = gameTime[-1]
            minute = int(time.split(':')[0])
            #goals.append((half, time))
            if half == "1":
                goals.append(minute)
    return goals

def main():
    stage = input("Train, test, or val?:\n")
    data_path = os.path.join(stage, "/Data/SoccerNet/england_epl")
    save_path = os.path.join(stage, "/Data/SoccerNet/Tensors")
    # define transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((112, 112)), transforms.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
    num_blocks = 0
    block_num_to_label = {}
    for season in os.listdir(data_path):
        for game in os.listdir(os.path.join(data_path, season)):
            with open(os.path.join(data_path, season, game, "Labels.json")) as outfile:
                game_label = json.load(outfile)["annotations"]
            # create list of all goal times in game
            goal_times = create_goal_times(game_label)
            print(goal_times)
            minute = 0
            for half in range(1):
                #file_path = os.path.join(data_path, season, game, "1.mkv")
                file_path = os.path.join(data_path, season, game, "1.mp4")
                print("Found file: %s" % file_path)
                #mp4_file = convert_to_mp4(file_path)
                mp4_file = file_path
                vid = cv2.VideoCapture(mp4_file)
                frame_counter = 0
                num_frames_in_block = 0
                block = torch.zeros(60, 3, 112, 112)
                
                while True:
                    ret, frame = vid.read()
                    if not ret:
                        break
                    if frame_counter % 25 == 0:
                        # downsample, normalize, convert to torch tensor
                        block[num_frames_in_block] = transform(frame)
                        num_frames_in_block += 1
                    if num_frames_in_block == 59:
                        print(minute)
                        # save block
                        torch.save(block, os.path.join(save_path, f"block_{num_blocks}.pt"))
                        block_num_to_label[num_blocks] = 1 if minute in goal_times else 0
                        num_blocks += 1
                        minute += 1
                        num_frames_in_block = 0

                    frame_counter += 1
    # Save block_num_to_label map
    with open(os.path.join(save_path, "block_num_to_label.pkl"), 'wb') as fd:
        pickle.dump(block_num_to_label, fd)

def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    with_mp4 = name + ".mp4"
    ffmpeg.input(mkv_file).output(with_mp4).run()
    print("Finished converting {}".format(mkv_file))
    return with_mp4

if __name__ == "__main__":
    main()
