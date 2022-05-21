import pickle, torch
import numpy as np
import glob, os, shutil
import json, imagesize
from subprocess import call
from format_processor import FormatReader
import cv2
import copy
from shapely.geometry import Polygon, Point

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

        if problem_type == "det":
            post_fix = f"{epoch}"

        if problem_type == "rec":
            post_fix = f"{epoch}"
            
        return post_fix

def write_log(saved_dir, model_dict):
    model_name = os.path.basename(model_dict['directory'])
    log_file = os.path.join(saved_dir, model_name + ".json")

    log_object = json.dumps(model_dict, indent = 4)
    f = open(log_file, "w")
    f.write(log_object)
    f.close()

def backup(model_dict, root_model_dir):
    model_dir = model_dict['directory']
    checkpoint_name = model_dict['checkpoint_name']
    checkpoint_filename = os.path.join(model_dir, checkpoint_name)

    module = model_dict['module']
    problem_type = model_dict['problem_type']

    #Save model
    model_name = os.path.basename(model_dir)        
    saved_model_dir = os.path.join(root_model_dir, problem_type, model_name)
    os.makedirs(saved_model_dir, exist_ok = True)

    post_fix = ""
    if module == "paddle":
        config_ext = ".yml"
        state = checkpoint_filename + ".states"
        if os.path.exists(state):
            #================================Model================================
            #get info from state
            state_getter = StateGetter(state)
            post_fix = state_getter.get_paddle_info(problem_type)

            #update config
            saved_checkpoint_name = f"{model_name}_{post_fix}"
            model_dict['checkpoint_name'] = saved_checkpoint_name

            #backup checkpoint
            model_exts = [".pdopt", ".pdparams", ".states"]
            for ext in model_exts:
                checkpoint_file = os.path.join(model_dir, checkpoint_name + ext) 
                saved_checkpoint_file = os.path.join(saved_model_dir, saved_checkpoint_name + ext) 

                shutil.copy(checkpoint_file, saved_checkpoint_file)

            #===============================Other================================

    if module == "mmocr":
        config_ext = ".py"
        state = checkpoint_filename + ".pth"        
        if os.path.exists(state):
            #================================Model================================
            #get info from state
            if "latest" == checkpoint_name:
                state_getter = StateGetter(state)
                post_fix = state_getter.get_mmocr_info(problem_type)

            #update config
            saved_checkpoint_name = f"{checkpoint_name}_{post_fix}"
            model_dict['checkpoint_name'] = saved_checkpoint_name

            #backup checkpoint
            checkpoint_file = os.path.join(model_dir, checkpoint_name + ".pth")
            saved_checkpoint_file = os.path.join(saved_model_dir, saved_checkpoint_name + ".pth")
            shutil.copy(checkpoint_file, saved_checkpoint_file)

    #config
    if "config_name" in model_dict.keys():
        config_name = model_dict['config_name']
        saved_config_name = f"{config_name}_{post_fix}"
        model_dict['config_name'] = saved_config_name

        #backup config
        config_file = os.path.join(model_dir, config_name + config_ext)
        saved_config_file = os.path.join(saved_model_dir, saved_config_name + config_ext)

        shutil.copy(config_file, saved_config_file)
        
    return model_dict
    

#==================================Some other function===========================
def bash_script(command):
    with open("script.sh", "w") as f:
            f.write(command)
            f.close()
    call(["bash", "script.sh"])
    # os.remove("script.sh")

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

#Detector - process det
def plot_polygon(image, polygon):
    pts = np.array(polygon, np.int32)
    
    pts = pts.reshape((-1, 1, 2))
    
    isClosed = True
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.polylines() method
    # Draw a Blue polygon with 
    # thickness of 1 px
    image = cv2.polylines(image, [pts], 
                        isClosed, color, thickness)
    
    return image

def ios(poly1, poly2):
    poly1 = np.float32(poly1)
    poly1 = np.reshape(poly1, (-1,2))

    poly2 = np.float32(poly2)
    poly2 = np.reshape(poly2, (-1,2))

    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    flag = True
    max_area = max(polygon1.area, polygon2.area)
    if polygon1.area == max_area:
        for point in poly2:
            point = Point(point)
            if not point.within(polygon1):
                flag = False
                break
                
    if polygon2.area == max_area:
        for point in poly1:
            point = Point(point)
            if not point.within(polygon2):
                flag = False
                break

    if flag == False:
        intersect = polygon1.intersection(polygon2).area
        smaller = min(polygon1.area, polygon2.area)

        ios = intersect / smaller

        return ios
    else:
        return -1


def merge_box(poly1, poly2):    
    poly1 = np.float32(poly1)
    poly1 = np.reshape(poly1, (-1,2)).tolist()

    poly2 = np.float32(poly2)
    poly2 = np.reshape(poly2, (-1,2)).tolist()

    poly12 = copy.deepcopy(poly1)
    poly12.extend(poly2)
    xmin, ymin = np.min(poly12, axis = 0)
    xmax, ymax = np.max(poly12, axis = 0)

    bbox = [[xmin, ymin], [xmax, ymin],
            [xmax, ymax], [xmin, ymax]]

    return bbox


#Detector - mmocr
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
            
def rename_cropimage(rec_dict_list, croped_dir):
    for image_file, annotation in rec_dict_list[0].items():
        if annotation["label"] == "":
            continue

        label1 = annotation["label"].replace("/","-")
        post_fix = [label1]
        for dictionary in rec_dict_list[1:]:
            if dictionary[image_file]["label"] == "":
                continue
            label2 = dictionary[image_file]["label"].replace("/","-")
            post_fix.append(label2)
        post_fix = "_".join(post_fix)

        image_name_ext = os.path.basename(image_file)
        image_name, ext = os.path.splitext(image_name_ext)

        new_image_name = image_name + "_" + post_fix + ext
        new_image_file = os.path.join(croped_dir, new_image_name)
        os.rename(image_file, new_image_file)
