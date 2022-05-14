import os, glob, shutil, json, cv2, imagesize
from utility import backup, merge_two_file, dict_list_to_text, convert_to_bbox, process_point
from format_processor import FormatReader
from image_processor import ImageCroper
from subprocess import call
class PaddleDetector():
    def __init__(self, det_dict):
        self.det_dict = det_dict

    def predict(self, image_dir, output_file, use_gpu = True):
        model_dir = self.det_dict['directory']
        checkpoint_filename =  os.path.join(model_dir, self.det_dict['checkpoint_name'])
        config_file = os.path.join(model_dir, self.det_dict['config_name'] + ".yml")
        
        modify = ["python", "/content/drive/MyDrive/Busmap/OCR/SceneText/script/PaddleOCR_2.4/tools/infer_det.py",
                  f'-c {config_file}',
                  f'-o Global.infer_img={image_dir}',
                  f'Global.checkpoints={checkpoint_filename}',
                  f'Global.save_res_path={output_file}',
                  f'Global.use_gpu={use_gpu}']
        command = " ".join(modify)
        with open("script.sh", "w") as f:
            f.write(command)
            f.close()
        call(["bash", "script.sh"])

    def infer(self, image_dir, output_dir, use_gpu = True):
        model_dir = self.det_dict['directory']
        algorithm = self.det_dict['algorithm']
        
        modify = ['python script/PaddleOCR_2.4/tools/predict_det.py',
                  f'--image_dir={image_dir}',
                  f'--det_algorithm={algorithm}',
                  f'--det_model_dir={model_dir}',
                  f'--draw_img_save_dir={output_dir}',
                  f'--use_gpu={use_gpu}']
        command = " ".join(modify)
        os.system(command)

    def call(self, image_dir, saved_result_dir, root_model_dir, output_name = "det.txt"):
        output_file = ""
        log = {}
        
        #predict 
        checkpoint_name = self.det_dict['checkpoint_name']
        
        if checkpoint_name == "inference":
            output_dir = os.path.join(saved_result_dir, "det_results")
            output_file = os.path.join(output_dir, output_name)

            if not os.path.exists(output_file):
                self.infer(image_dir, output_dir)
                
                #back_up
                model_dir = self.det_dict['directory']
                saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
                os.makedirs(saved_model_dir, exist_ok = True)
                
                ##checkpoint
                checkpoint_filename =  os.path.join(self.det_dict['directory'], checkpoint_name)
                post_fix = backup(checkpoint_filename, saved_model_dir,
                                   module = "paddle", problem_type = "det")
                
        else:
            output_file = os.path.join(saved_result_dir, output_name)

            if not os.path.exists(output_file):
                self.predict(image_dir, output_file)
                
                #back_up
                model_dir = self.det_dict['directory']
                saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
                os.makedirs(saved_model_dir, exist_ok = True)
                
                ##checkpoint
                config_name = self.det_dict['config_name']
                config_filename = os.path.join(self.det_dict['directory'], config_name)
                
                checkpoint_filename = os.path.join(self.det_dict['directory'], checkpoint_name)
                post_fix = backup(checkpoint_filename, saved_model_dir,
                                  config_filename = config_filename,
                                  module = "paddle", problem_type = "det")
                
                log['config_name'] = config_name + f"_{post_fix}"
        
        #write_log      
        log['model_dir'] = saved_model_dir
        log['checkpoint_name'] = checkpoint_name + f"_{post_fix}"
        
        log_file = os.path.join(saved_result_dir, "log.json")
        log_object = json.dumps(log, indent = 4)
        f = open(log_file, "w").write(log_object)
        
        return output_file

class MMOCRDetector():
    def __init__(self, det_dict):
        self.det_dict = det_dict

    def predict(self, image_dir, output_file):
        from mmocr.utils.ocr import MMOCR
        
        model_dir = self.det_dict['directory']
        checkpoint_file =  os.path.join(model_dir, self.det_dict['checkpoint_name'] + ".pth")
        config_file = self.det_dict['config_filename'] + ".py"
        
        contents = ""
        ocr = MMOCR(recog=None, det_config=config_file, det_ckpt=checkpoint_file)
        for image_path in glob.glob(image_dir + "/*"):
            results = ocr.readtext(image_path)

            dic_list = []
            for polygon in results[0]['boundary_result']:
                polygon = convert_to_bbox(polygon, image_path)

                dic = {}
                dic['transcription'] = ""
                dic['points'] = polygon
                dic['difficult'] = False

                dic_list.append(dic)
            
            text = dict_list_to_text(dic_list)
            contents += image_path + "\t" + text + "\n"
                
        with open(output_file, "w") as f:
            f.write(contents[:-1])
            f.close()

    def call(self, image_dir, save_result_dir, root_model_dir, output_name = "det.txt"):
        log = {}
        
        #predict
        output_file = os.path.join(save_result_dir, output_name)
        if not os.path.exists(output_file):
            self.predict(image_dir, output_file)

            #back_up
            model_dir = self.det_dict['directory']
            saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
            os.makedirs(saved_model_dir, exist_ok = True)
            
            ##checkpoint
            config_filename = self.det_dict['config_filename']
            config_name = os.path.basename(config_filename)
            
            checkpoint_name = self.det_dict['checkpoint_name']
            checkpoint_filename =  os.path.join(model_dir, checkpoint_name)
            post_fix = backup(checkpoint_filename, saved_model_dir,
                              config_filename = config_filename,
                              module = "mmocr", problem_type = "det")
        
        #write_log      
        log['model_dir'] = saved_model_dir
        log['checkpoint_name'] = checkpoint_name + f"_{post_fix}"
        
        log_file = os.path.join(self.saved_result_dir, "log.json")
        log_object = json.dumps(log, indent = 4)
        f = open(log_file, "w").write(log_object)
        
        return output_file

#======================================================================Predictor================================================
class DetPredictor():
    def __init__(self, image_dir, saved_result_dir, root_model_dir):
        self.saved_result_dir = saved_result_dir
        self.root_model_dir = root_model_dir
        
        self.image_dir = image_dir

    def detect(self, det_dict, output_name = "det.txt"):
        if det_dict['module'] == "paddle":
            det = PaddleDetector(det_dict)
            output_file = det.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
        
        if det_dict['module'] == "mmocr":
            det = MMOCRDetector(det_dict)
            output_file = det.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
        
        return output_file
    
    def process_det(self, writer, croped_dir, output_file):
        reader = FormatReader(output_file)
        dictionary = reader.read_det(self.image_dir)

        missed_images = []
        for image_path, annotation_list in dictionary.items():
            if annotation_list == None:
                missed_images.append(image_path)
                continue
            
            for idx, anno in enumerate(annotation_list):
                #process polygon
                polygon = anno["points"]

                #crop image
                image_filename = os.path.basename(image_path)
                image_name, extension = os.path.splitext(image_filename)
                
                crop_image_filename = image_name + f"_{idx}{extension}"
                crop_image_file = os.path.join(croped_dir, crop_image_filename)

                if not os.path.exists(crop_image_file):
                    croper = ImageCroper(image_path)
                    croped_image = croper.rectangle(polygon)
                    cv2.imwrite(crop_image_file, croped_image)
    
                    #store info
                    width, height = imagesize.get(image_path)
                    anno["points"] = [process_point(point, height, width) for point in polygon]
    
                    writer.record_paddle(image_filename, anno)
                    writer.record_bk(image_filename, anno["points"])
    
        return missed_images
        
def process_missed_det(missed_images, missed_dir, missed_det_dict,
                       saved_model_dir, output_file):
                           
    #Move missed_image to dir
    os.makedirs(missed_dir, exist_ok = True)
    os.makedirs(saved_model_dir, exist_ok = True)
    for image_path in missed_images:
        shutil.copy(image_path, missed_dir)
    
    deter = DetPredictor(missed_dir, missed_dir, saved_model_dir)
    missed_output_file = deter.detect(missed_dir, missed_det_dict)
    merge_two_file(output_file,  missed_output_file, output_file)