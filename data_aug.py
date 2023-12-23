import cv2
import argparse
import os
import pandas as pd


def get_label_frequence(genre_count):
    l = []
    for k, v in genre_count.items():
        if v > 600:
            l.append(k)

    return l

def check_aug(gs, fre_label):
    for g in gs:
        if g in fre_label:
            return False
    
    return True

def count_genre(df):
    genre_counts = {}
    for i in range(len(df)):
        label = df.iloc[i]["genre"]
        label = label.split("|")
        for l in label:
            if not l in genre_counts.keys():
                genre_counts[l] = 1
            else:
                genre_counts[l] += 1

    return genre_counts

def data_aug(old_df, save_path):
    
    #movieid,title,genre,id,img_path
    new_df = pd.DataFrame(columns=["movieid", "title", "genre", "id", "img_path"])
    index = 0

    
    genre_counts = count_genre(old_df)

    for s in range(3):
        f_label = get_label_frequence(genre_counts)
        
        for i in range(len(old_df)):
            d = old_df.iloc[i]
            m = d["movieid"]
            t = d["title"]
            g = d["genre"]
            id = d["id"]
            p = d["img_path"]

            if s == 0:
                new_df.loc[index] = [m, t, g, id, p]
                index += 1

            gs = g.split("|")
            
            if not check_aug(gs, f_label):
                continue

            new_p = p.split("\\")[-1][:-4]
            
            for g in gs:
                genre_counts[g] += 1

            if os.path.exists(p):
                img = cv2.imread(p)

                if s == 0:
                    img_v = cv2.flip(img, 1)
                    p_v = os.path.join(save_path, new_p + "_v1.jpg")
                    cv2.imwrite(p_v, img_v)
                    new_df.loc[index] = [m, t, g, id, p_v]
                    index += 1

                if s == 1:
                    img_h = cv2.flip(img, 0)
                    p_h = os.path.join(save_path, new_p + "_v2.jpg")
                    cv2.imwrite(p_h, img_h)
                    new_df.loc[index] = [m, t, g, id, p_h]
                    index += 1

                if s == 2:
                    img_h_v = cv2.flip(img, -1)
                    p_h_v = os.path.join(save_path, new_p + "_v3.jpg")
                    cv2.imwrite(p_h_v, img_h_v)
                    new_df.loc[index] = [m, t, g, id, p_h_v]
                    index += 1
                # if s == 3:
    
    return new_df

    

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--path_data", type=str, default="data/dataset")
    parse.add_argument("--save_path", type=str, default="data/dataset")
    args = parse.parse_args()

    path_csv = os.path.join(args.path_data, "movies_train.csv")
    data = pd.read_csv(path_csv)

    save_path = os.path.join(args.save_path, "aug")
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    new_df = data_aug(data, save_path)
    new_df.to_csv(os.path.join(args.path_data, "movies_train_aug.csv"), index=False)

    # data_aug(args)