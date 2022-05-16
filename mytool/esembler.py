import os, shutil, glob
def choose_anno(file1, file2):
        f1 = open(file1, "r")
        r1 = f1.read().split("\n")

        f2 = open(file2, "r")
        r2 = f2.read().split("\n")

        if len(r1) > len(r2):
            return file1
        else:
            return file2

def merge_anno(file1, file2):
        data1 = data2 = ""

        with open(file1, "r") as fp:
            data1 = fp.read()
            fp.close()

        with open(file2, "r") as fp:
            data2 = fp.read()
            fp.close()

        if (data1[-1] == "\n") and (data2[-1] == "\n"):
            data = data1 + data2[:-1]

        if (data1[-1] == "\n") and (data2[-1] != "\n"):
            data = data1 + data2

        if (data1[-1] != "\n") and (data2[-1] != "\n"):
            data = data1 + "\n" + data2

        with open(file1, "w") as fp:
            fp.write(data)
            fp.close()

class DetEsembler():
    def __init__(self, saved_result_dirs):
        self.saved_result_dirs = saved_result_dirs
    
    def stack_result(self):
        #create esemble saved_result_dir
        ids = [sub_dir.split("_")[-1] for sub_dir in self.saved_result_dirs]
        esemble_saved_result_dir = self.saved_result_dirs[0].split("_")[:-1]
        esemble_saved_result_dir.extend(ids)
        esemble_saved_result_dir = "_".join(esemble_saved_result_dir)

        #create esemble bkai
        esemble_saved_bkai = esemble_saved_result_dir + "/bkai"
        temp_saved_bkai = os.path.join(self.saved_result_dirs[0], "bkai")

        if not os.path.exists(esemble_saved_bkai):
            shutil.copytree(temp_saved_bkai, esemble_saved_bkai)

        for anno_file in os.listdir(esemble_saved_bkai):
            temp_sub_anno_file = os.path.join(esemble_saved_bkai, anno_file)
            for saved_result_dir in self.saved_result_dirs[1:]:
                sub_anno_file = os.path.join(saved_result_dir, "bkai", anno_file)
                merge_anno(temp_sub_anno_file, sub_anno_file)

        zip_file = os.path.join(esemble_saved_result_dir, "prediction")
        os.chdir(esemble_saved_bkai)
        shutil.make_archive(zip_file, format='zip', root_dir='.')

    def greater_result(self):
        #create esemble saved_result_dir
        ids = [sub_dir.split("_")[-1] for sub_dir in self.saved_result_dirs]
        esemble_saved_result_dir = self.saved_result_dirs[0].split("_")[:-1]
        esemble_saved_result_dir.extend(ids)
        esemble_saved_result_dir = "_".join(esemble_saved_result_dir)

        esemble_saved_bkai = esemble_saved_result_dir + "/bkai"
        os.makedirs(esemble_saved_bkai, exist_ok = True)

        temp_saved_bkai = os.path.join(self.saved_result_dirs[0], "bkai")
        for anno_file in os.listdir(temp_saved_bkai):
            temp_sub_anno_file = os.path.join(temp_saved_bkai, anno_file)
            for saved_result_dir in self.saved_result_dirs[1:]:
                sub_anno_file = os.path.join(saved_result_dir, "bkai", anno_file)
                temp_sub_anno_file = choose_anno(temp_sub_anno_file, sub_anno_file)
            shutil.copy(temp_sub_anno_file, esemble_saved_bkai)

        zip_file = os.path.join(esemble_saved_result_dir, "prediction")
        os.chdir(esemble_saved_bkai)
        shutil.make_archive(zip_file, format='zip', root_dir='.')