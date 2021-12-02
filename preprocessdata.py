#! /usr/bin/env python
import os
import sys
import json
import torch
import pickle
import cv2
from torchvision import transforms
import ffmpeg
import subprocess
'''
Downsample from 398x224 at 25fps to 112x112 at 25fps
'''

root_path = '/scratch/users/mpike27/CS230/data/'
ACTIONS = ['Kick-off', 'Ball out of play', 'Throw-in', 'Corner', 'Shots on target', \
    'Offside', 'Clearance', 'Goal', 'Foul', 'Indirect free-kick', \
    'Shots off target', 'Substitution', 'Yellow card', 'Direct free-kick', \
    'Penalty', 'Red card', 'Yellow then red card', 'None']

ACTION_TO_IND = {
    'Kick-off': 0, 
    'Ball out of play': 1, 
    'Throw-in': 2, 
    'Corner': 3, 
    'Shots on target': 4, 
    'Offside': 5, 
    'Clearance': 6, 
    'Goal': 7, 
    'Foul': 8, 
    'Indirect free-kick': 9, 
    'Shots off target': 10, 
    'Substitution': 11, 
    'Yellow card': 12, 
    'Direct free-kick': 13, 
    'Penalty': 14, 
    'Red card': 15, 
    'Yellow then red card': 16, 
    'None': 17
}

def timeToBlockNum(time, clip_size):
    minutes = int(time.split(':')[0])
    seconds = int(time.split(':')[1])
    totalSeconds = minutes * 60 + seconds
    return totalSeconds // clip_size

#create list of goal times in the game from labels
def create_action_times(game_label, clip_size):
    actions = {}
    for i in range(len(game_label)):
        label = game_label[i]['label']
        if label in ACTION_TO_IND:
            # ie label_dict["gameTime"] = "1 - 13:10"
            gameTime = game_label[i]["gameTime"].split(' ') 
            half = gameTime[0]
            time = gameTime[-1]
            # minute = int(time.split(':')[0])
            # actions.append((half, time))
            block_num = timeToBlockNum(time, clip_size)
            if half == "1":
                actions[block_num] = ACTION_TO_IND[label]
    return actions

def main():

    # log = open("log.out", "a")
    # sys.stdout = log
    # sys.stderr = log
    # stage = input("Train, test, or val?:\n")
    # define transform
    clip_size = int(sys.argv[1])
    #Make sure clip size if even on the minute
    assert(60 % clip_size == 0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((112, 112)), transforms.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
    for split in ['train', 'val', 'test']:
        # num_blocks = 0
        blocks = []
        block_num_to_label = []
        data_path = root_path + split + "/SoccerNet/england_epl"
        save_path = root_path + split + "/SoccerNet/Tensors"
        for season in os.listdir(data_path):
            for game in os.listdir(os.path.join(data_path, season)):
                with open(os.path.join(data_path, season, game, "Labels-v2.json")) as outfile:
                    game_label = json.load(outfile)["annotations"]
                # create list of all goal times in game
                action_times = create_action_times(game_label, clip_size)
                game_block_num = 0
                for half in range(1):
                    file_path = os.path.join(data_path, season, game)
                    print(file_path)
                    mkv_file = os.path.join(file_path, '1.mkv')
                    mp4_file = os.path.join(file_path, '1.mp4')
                    if not os.path.exists(mp4_file):
                        convert_to_mp4(mkv_file, mp4_file)

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
                        if num_frames_in_block == clip_size - 1:
                            # breakpoint()
                            # save block
                            blocks.append(block.unsqueeze(0))
                            # torch.save(block, os.path.join(save_path, f"block_{num_blocks}.pt"))
                            if game_block_num in action_times:
                                block_num_to_label.append(action_times[game_block_num])
                            else: 
                                block_num_to_label.append(ACTION_TO_IND['None'])
                            # num_blocks += 1
                            game_block_num += 1
                            num_frames_in_block = 0
                        frame_counter += 1
                print(len(blocks))
                
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        blocks_tensor = torch.Tensor(len(blocks), *(blocks[0].shape))
        torch.cat(blocks, out=blocks_tensor)
        block_num_to_label = torch.Tensor(block_num_to_label)
        tensor_path = os.path.join(save_path, f"blocks_{clip_size}.pt")
        labels_path = os.path.join(save_path, f"block_num_to_label_{clip_size}.pt")
        print(f"Saving {tensor_path} with shape {blocks_tensor.shape}")
        torch.save(blocks_tensor, tensor_path)
        print(f"Saving {labels_path} with shape {block_num_to_label.shape}")
        torch.save(block_num_to_label, labels_path)
        # # Save block_num_to_label map
        # with open(os.path.join(save_path, f"block_num_to_label_{clip_size}.pkl"), 'wb') as fd:
        #     pickle.dump(block_num_to_label, fd)

def convert_to_mp4(mkv_file, mp4_file):
    subprocess.run(["ffmpeg", "-i", mkv_file, "-codec", "copy", mp4_file])
    print("Finished converting {}".format(mkv_file))

if __name__ == "__main__":
    main()
