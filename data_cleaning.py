import os
import shutil
import re


def copy_all_file(source_folder, destination_folder):
    # go through all file and copy it
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".java"):
                shutil.copy(os.path.join(root, file), os.path.join(destination_folder, file))
                print("Copied", os.path.join(root, file))


def clean_data(text):
    reg_clean_comment = r"/\*[\s\S]*?\*/";
    text = re.sub(reg_clean_comment, "", text, flags=re.M)

    reg_clean_package = r"(package|import)[\s\S]*?;"
    text = re.sub(reg_clean_package, "", text)

    return text


def clean_all_files(folder):
    for file in os.listdir(folder):
        programFile = open(os.path.join(folder, file), encoding="utf-8", errors='ignore')
        programText = programFile.read()
        programFile.close()

        programText = clean_data(programText)

        programFile = open(os.path.join(folder, file), mode="w+", encoding="utf-8")
        programFile.write(programText)
        programFile.close()
        print("Cleaned", file)


if __name__ == "__main__":
    dataset = "xalan/xalan-j_2_4_0"

    source_folder = "./data/PROMISE-backup/source code/" + dataset
    destination_folder = "./data/Clean Data/" + dataset

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, mode=0o777)
        print("Create", destination_folder)

    copy_all_file(source_folder, destination_folder)
    clean_all_files(destination_folder)
