import argparse
import pickle as pkl
import json
import shutil


# Defining main function 
def main(args):
    # load global meta data
    index_file = 'index_ade20k.pkl'
    with open('{}/{}'.format(args.meta_sourse_dir, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)
    nfiles = len(index_ade20k['filename'])
    print("Dataset Size: ", nfiles)
    out_door_ls = []
    for i in range(nfiles):
        if "train" not in index_ade20k['filename'][i] and "val" not in index_ade20k['filename'][i]:
            continue
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i]).replace(".jpg", ".json")
        # parse image meta data
        try:
            with open(full_file_name, 'r') as meta_file:
                data = json.load(meta_file)
                if "outdoor" in data["annotation"]["scene"]:
                    out_door_ls.append(i)
        except:
            print("{} load error".format(full_file_name))

    # record outdoor image
    print("Outdoor Dataset Size: ", len(out_door_ls))
    for i in out_door_ls:
        # get filename
        if "train" in index_ade20k['filename'][i]:
            sub_dir = "training"
        else:
            sub_dir = "validation"
        # copy image
        try:
            source_file = '{}/images/{}/{}'.format(args.image_sourse_dir, sub_dir, index_ade20k['filename'][i])
            destination_file = '{}/images/{}/{}'.format(args.image_target_dir, sub_dir, index_ade20k['filename'][i])
            shutil.copyfile(source_file, destination_file)

            # copy annotation
            source_file = '{}/annotations/{}/{}'.format(args.image_sourse_dir, sub_dir, index_ade20k['filename'][i].replace(".jpg", ".png"))
            destination_file = '{}/annotations/{}/{}'.format(args.image_target_dir, sub_dir, index_ade20k['filename'][i].replace(".jpg", ".png"))
            shutil.copyfile(source_file, destination_file)
        except:
            print("Copy error: {}".format(index_ade20k['filename'][i]))

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description='A simple program to demonstrate argparse')
    # Add arguments
    parser.add_argument('--meta_sourse_dir', type=str, default='/home/haydeew0102/ADE20K_2021_17_01')
    parser.add_argument('--image_sourse_dir', type=str, default='/home/haydeew0102/ADEChallengeData2016') 
    parser.add_argument('--image_target_dir', type=str, default='/home/haydeew0102/ADEChallengeData2016_outdoors')

    # Parse the arguments
    args = parser.parse_args()
    main(args)