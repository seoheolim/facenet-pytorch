from model.facenet import detect_mosaic, detect_box

if __name__ == "__main__":
    my_path = "test/"
    i = 4
    detect_mosaic(f"{my_path}test{i}.mp4", i)
    detect_box(f"{my_path}test{i}.mp4", i)
