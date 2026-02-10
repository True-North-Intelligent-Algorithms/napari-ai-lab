import os


def zero_pad_index(index, width=5):
    return f"{index:0{width}d}"


def generate_next_name(image_path, name, ext=".tif"):

    index = 0
    image_name = image_path + "/" + name + "_" + zero_pad_index(index) + ext

    while os.path.exists(image_name):
        index = index + 1
        image_name = (
            image_path + "/" + name + "_" + zero_pad_index(index) + ext
        )

    base_name = os.path.basename(image_name)
    base_name = os.path.splitext(base_name)[0]

    return base_name
