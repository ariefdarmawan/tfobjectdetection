import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--template', action="store",dest="template", required=True)
    parser.add_argument('--output', action="store",dest="output", required=True)
    parser.add_argument('--label', action="store", dest="label", default="../data/label_map.csv")
    parser.add_argument('--map', action="store", dest="map", default="../data/label.pbtxt")
    parser.add_argument('--ckpt', action="store", dest="ckpt", default="../data/model.ckpt")
    parser.add_argument('--train', action="store", dest="train", default="../data/train.record")
    parser.add_argument('--test', action="store", dest="test", default="../data/test.record")
    parser.add_argument('--test_images_dir', action="store", dest="test_images_dir", default="../images/test")
    flags = parser.parse_args()

    try:
        template = open(flags.template,'r')
        tmp = template.read()

        # get label count
        labels = pd.read_csv(flags.label)
        tmp = tmp.replace("{--numclass--}", str(len(labels)))

        if flags.map!='':
            f = open(flags.map, "w")
            for _, row in labels.iterrows():
                f.write("item {\n")
                f.write("\tid: {}\n".format(row["tagid"]))
                f.write("\tname: '{}'\n".format(row["label"]))
                f.write("}\n\n")
            f.close()
        tmp = tmp.replace("{--labelmap_path--}", flags.map)

        # template replace files
        tmp = tmp.replace("{--model_ckpt_path--}", flags.ckpt)
        tmp = tmp.replace("{--tfrecord_train_path--}", flags.train)
        tmp = tmp.replace("{--tfrecord_test_path--}", flags.test)

        # test images
        image_count = 0
        image_exts = ['.jpg','.jpeg','.png']
        files = os.listdir(flags.test_images_dir)
        for filename in files:
            _, ext = os.path.splitext(filename)
            for image_ext in image_exts:
                if ext==image_ext:
                    image_count +=1
                    break
        tmp = tmp.replace("{--test_image_num--}", str(image_count))

        template.close()
        
        # write tmp to config
        cfg = open(flags.output, 'w')
        cfg.write(tmp)
        cfg.close()

        print("successfully write config file {}\n".format(flags.output))
    except Exception as e:
        print("error on open template file {}: {}".format(flags.template, str(e)))

main()