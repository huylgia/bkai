import pickle, torch
import numpy as np
import glob, os, shutil
import json, imagesize
from format_processor import FormatReader
#=============================Backup model - For detector, recognitor=============================
class StateGetter():
    def __init__(self, model_state):
        self.model_state = model_state
    
    def get_paddle_info(self, problem_type = "det"):
        with open(self.model_state, 'rb') as f:
            info = pickle.load(f)
            epoch = info['epoch']

            if problem_type == "det":
                score = round(info['best_model_dict']['hmean'], 4) * 100
                post_fix = f"{epoch}_hmean:{score}"

            if problem_type == "rec":
                score = round(info['best_model_dict']['acc'], 4) * 100
                post_fix = f"{epoch}_acc:{score}"

            f.close()
            
        return post_fix
    
    def get_mmocr_info(self, problem_type = "det"):
        info = torch.load(self.model_state)
        epoch = info['meta']['epoch']

        score = round(info['mea']['hook_msgs']['best_score'], 4) * 100
        if problem_type == "det":
            post_fix = f"{epoch}_best_hmean:{score}"

        if problem_type == "rec":
            post_fix = f"{epoch}_best_acc:{score}"
            
        return post_fix

def backup(model_filename, saved_model_dir, config_filename = None, module = "paddle", problem_type = "det"):
    module = module.lower()
    problem_type = problem_type.lower()

    #Save model
    post_fix = ""
    config_ext = ""
    
    if module == "paddle":
        model_state = model_filename + ".states"
        config_ext = ".yml"
        
        if os.path.exists(model_state):
            state_getter = StateGetter(model_state)
            post_fix = state_getter.get_paddle_info(problem_type)
            
            #checkpoint
            for model_file in glob.glob(model_filename + "*"):
                model_name_ext = os.path.basename(model_file)
                model_name, ext = os.path.splitext(model_name_ext)
                
                saved_model_file = os.path.join(saved_model_dir, f"{model_name}_{post_fix}{ext}")
                shutil.copy(model_file, saved_model_file)
                
    if module == "mmocr":
        model_state = model_filename + ".pth"
        config_ext = ".py"
        
        if os.path.exists(model_state):
            if "latest" in model_state:
                state_getter = StateGetter(model_state)
                post_fix = state_getter.get_mmocr_info(problem_type)
            
            #checkpoint
            model_name_ext = os.path.basename(model_state)
            model_name, ext = os.path.splitext(model_name_ext)
            
            saved_model_file = os.path.join(saved_model_dir, f"{model_name}_{post_fix}{ext}")
            shutil.copy(model_file, saved_model_file)
            
    #config
    if config_filename and config_ext:
        config_name = os.path.basename(config_filename)
        config_file = config_filename + config_ext
        saved_config_file = os.path.join(saved_model_dir, f"{config_name}_{post_fix}{config_ext}")
        shutil.copy(config_file, saved_config_file)
        
    return post_fix
    

#==================================Some other function===========================
def process_point(point, h, w):
    x, y = point

    x = int(x) if x >= 0 else 0
    y = int(y) if y >= 0 else 0
    x = int(x) if x <= w-1 else w-1
    y = int(y) if y <= h-1 else h-1

    return [x,y]

    return polygon

def dict_list_to_text(dic_list):
      paddle_results = json.dumps(dic_list, indent = 4, ensure_ascii=False).encode("utf8")
      paddle_results = paddle_results.decode()
      text = str(paddle_results).replace("\n","")
      text = text.replace(" ","")
      text = text.replace(",",", ")
      text = text.replace(":",": ")

      return text

def convert_to_bbox(polygon, image_path):
    width, height = imagesize.get(image_path)

    polygon = np.reshape(polygon[:-1], (-1,2))
    top_left = np.min(polygon, axis = 0)
    bottom_right = np.max(polygon, axis = 0)

    top_left = process_point(top_left, height, width)
    bottom_right = process_point(bottom_right, height, width)
    polygon = [top_left,
               [bottom_right[0], top_left[1]],
               bottom_right,
               [top_left[0], bottom_right[1]]
              ]
              
def merge_two_file(file1, file2, mer_file):
        data = data2 = ""
        
        # Reading data from file1
        with open(file1) as fp:
            data = fp.read()
            
        # Reading data from file2
        with open(file2) as fp:
            contents = fp.read().split("\n")
                
        # Merging 2 files
        # To add the data of file2
        # from next line
        data += "\n"
        data += data2
            
        with open(mer_file, 'w') as fp:
            fp.write(data) 
            
def rename_cropimage(output_file, croped_dir):
    reader = FormatReader(output_file)
    dictionary = reader.read_recog(croped_dir)
    
    for image_file, annotation in dictionary.items():
        if annotation["label"] == "":
          continue
        annotation["label"] = annotation["label"].replace("/","-")
        image_name = os.path.basename(image_file)
        new_image_name = annotation["label"] + "_" + image_name
        
        new_image_file = os.path.join(croped_dir, new_image_name)
        os.rename(image_file, new_image_file)
