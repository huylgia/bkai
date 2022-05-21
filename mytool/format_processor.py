import ast
import numpy as np
import os, glob, shutil, json
import re

#============================================================Reader============================================================
def process_annotation(annotation):
        '''
            annotation: [{"transcription": None, "points": [..], "difficult": false}, {...}]
        '''
        annotation = re.sub("^\[|\]$", "", annotation)
        annotation = re.sub(", {", ",  {", annotation)
        annotation = re.sub('"difficult": false', '"difficult": False', annotation)
        annotation = re.sub('"difficult": true', '"difficult": True', annotation)
        annotation_list = annotation.split(",  ")
        annotation_list = [ast.literal_eval(anno) for anno in annotation_list]

        return annotation_list

class FormatReader():
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        
    def read_det(self, image_dir):
        dictionary = {}
        with open(self.annotation_file, "r") as f:
            for row in f.readlines():
                if row == "\n":
                    continue
                row = row.replace("\n","").split("\t")
                #Get info
                image_path = row[0]
                annotations = row[1]
                
                #process image_path
                image_path = os.path.join(image_dir, os.path.basename(image_path))

                #process annotation
                if annotations == "[]":
                    annotation_list = None
                else:
                    annotation_list = process_annotation(annotations)
                dictionary[image_path] = annotation_list
        return dictionary
    
    def read_recog(self, croped_dir):
        dictionary = {}
        with open(self.annotation_file, "r") as f:
            for row in f.readlines():
                row = row.replace("\n","")
                image_path, label, score = row.split("\t")
                
                image_path = os.path.join(croped_dir, os.path.basename(image_path))
                dictionary[image_path] = {"label": label,
                                            "score": float(score)}
          
        return dictionary

#============================================================Writer============================================================
def dict_list_to_text(dic_list):
      paddle_results = json.dumps(dic_list, indent = 4, ensure_ascii=False).encode("utf8")
      paddle_results = paddle_results.decode()
      text = str(paddle_results).replace("\n","")
      text = text.replace(" ","")
      text = text.replace(",",", ")
      text = text.replace(":",": ")

      return text

def init_bk(image_dir, annotation_dir):
    image_name_ext_list = os.listdir(image_dir)
    image_name_list = [os.path.splitext(image_name_ext)[0] for image_name_ext in image_name_ext_list]
    anno_name_ext_list = ["res_%s.txt"%image_name for image_name in image_name_list]
    for anno_name_ext in anno_name_ext_list:
        anno_file = os.path.join(annotation_dir, anno_name_ext) 
        with open(anno_file, "w") as f:
            f.write("")
            f.close()

class FormatWriter():
    def __init__(self):
      self.paddle_dict = {}
      self.bk_dict = {}

    def record_paddle(self, image_filename, anno_dict = None, idx = None, transcription = None):
      if image_filename not in self.paddle_dict.keys():
        self.paddle_dict[image_filename] = []
      
      if anno_dict != None:
        self.paddle_dict[image_filename].append(anno_dict)
      
      if transcription != None:
        if type(transcription) != int:
            self.paddle_dict[image_filename][idx]["transcription"] = transcription
        else:
            self.paddle_dict[image_filename][idx] = None

    def record_bk(self, image_filename, polygon = None, idx = None, transcription = None):
      if image_filename not in self.bk_dict.keys():
        self.bk_dict[image_filename] = []

      if polygon != None:
        polygon = np.ravel(polygon).tolist()
        info = ",".join([str(cor) for cor in polygon]) + ",\n"
        self.bk_dict[image_filename].append(info)
      
      if transcription != None:
        if type(transcription) != int:
            self.bk_dict[image_filename][idx] = self.bk_dict[image_filename][idx].replace("\n","") + f"{transcription}\n"
        else:
            self.bk_dict[image_filename][idx] = None
    
    def write_paddle(self, annotation_file):
        contents = ""
        for image_filename, annotation_list in self.paddle_dict.items():
            #remove None
            annotation_list = [i for i in annotation_list if i]

            annotations = dict_list_to_text(annotation_list)
            text = image_filename + "\t" + annotations + "\n"
            contents+= text

        with open(annotation_file, "w") as f:
            f.write(contents[:-1])
            f.close()

    def write_bk(self, image_dir, annotation_dir):
        os.makedirs(annotation_dir, exist_ok = True)
        init_bk(image_dir, annotation_dir)
        
        for image_filename, annotation_list in self.bk_dict.items():
            #remove None
            annotation_list = [i for i in annotation_list if i]

            image_name, _ = os.path.splitext(image_filename)
            annotation_filename = "res_%s.txt"%image_name
            annotation_file = os.path.join(annotation_dir, annotation_filename)
            
            if annotation_list:
                annotation_list[-1] = annotation_list[-1].replace("\n","")

            with open(annotation_file, "w") as f:
                f.writelines(annotation_list)
                f.close()
        zip_file = os.path.join(os.path.dirname(annotation_dir), "prediction")
        os.chdir(annotation_dir)
        shutil.make_archive(zip_file, format='zip', root_dir='.')
