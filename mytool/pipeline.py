import imagesize
import os, glob, shutil, json
import cv2
import numpy as np

from detector import DetPredictor, process_missed_det
from recognitor import RecPredictor, esemble_rec
from format_processor import FormatReader, FormatWriter
from image_processor import ImageCroper
from utility import rename_cropimage

class Predictor():
    def __init__(self, image_dir, saved_result_dir, root_model_dir = None):
        self.saved_result_dir = saved_result_dir
        self.root_model_dir = root_model_dir
        
        self.image_dir = image_dir
        self.croped_dir = os.path.join(saved_result_dir, "croped")
        
        self.writer = FormatWriter()
        
        os.makedirs(saved_result_dir, exist_ok = True)
        
    def run(self, det_dicts, rec_dicts):
        #Detector
        deter = DetPredictor(self.image_dir, self.saved_result_dir, root_model_dir = self.root_model_dir)
        det_file = deter.detect(det_dicts[0])
        missed_images = deter.process_det(self.croped_dir, det_file, writer = self.writer)
        
        if missed_images:
            missed_det_file, missed_dir = process_missed_det(missed_images, det_dicts[1],
                                                             det_file, root_model_dir = self.root_model_dir)
            deter.process_det(self.cropped_dir, missed_det_file, writer = self.writer)
            shutil.rmtree(missed_dir)

        #Recognitor
        recer = RecPredictor(self.croped_dir, self.saved_result_dir, root_model_dir = self.root_model_dir)
        
        rec_dict_list = []
        for idx, rec_dict in enumerate(rec_dicts):
            rec_file = recer.recognize(rec_dict, f"rec_{idx}.txt")
            
            reader = FormatReader(rec_file)
            dictionary = reader.read_recog(self.croped_dir)
            rec_dict_list.append(dictionary)
        
        rec_name_ext = "esemble_" + os.path.basename(rec_file).split("_")[0] + ".txt"
        rec_file = os.path.join(os.path.dirname(rec_file), rec_name_ext)
        esemble_rec(rec_file, rec_dict_list)
        
        lower_confidence_images = recer.process_rec(rec_file, writer = self.writer)
        rename_cropimage(rec_file, self.croped_dir)
        
        #Save result
        self.writer.write_paddle(self.saved_result_dir + "/Label.txt")
        self.writer.write_bk(self.saved_result_dir + "/bkai")
    