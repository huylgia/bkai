import os, glob, shutil, json, cv2, imagesize
import numpy as np
from format_processor import FormatReader
from image_processor import ImageCroper
from subprocess import call
import os, glob, shutil, json, cv2, imagesize
import utility as u
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
        
        modify = ["python", "tools/infer_det.py",
                  f'-c {config_file}',
                  f'-o Global.infer_img={image_dir}',
                  f'Global.checkpoints={checkpoint_filename}',
                  f'Global.save_res_path={output_file}',
                  f'Global.use_gpu={use_gpu}']
        command = " ".join(modify)

        #Run script
        os.chdir("/content/bkai/PaddleOCR_2.5")
        u.bash_script(command)

    def infer(self, image_dir, output_dir, use_gpu = True):
        os.makedirs(output_dir, exist_ok = True)
        model_dir = self.det_dict['directory']
        algorithm = self.det_dict['algorithm']
        
        modify = ['python', 'tools/infer/predict_det_tinh.py',
                  f'--image_dir={image_dir}',
                  f'--det_algorithm={algorithm}',
                  f'--det_model_dir={model_dir}',
                  f'--draw_img_save_dir={output_dir}',
                  f'--use_gpu={use_gpu}']
        command = " ".join(modify)
        
        #Run script
        os.chdir("/content/bkai/PaddleOCR_2.4")
        u.bash_script(command)

    def call(self, image_dir, saved_result_dir, root_model_dir = None, output_name = "det.txt"):
        if self.det_dict['checkpoint_name'] == "inference":
            output_dir = os.path.join(saved_result_dir, "det_results")
            output_file = os.path.join(output_dir, output_name)

            if not os.path.exists(output_file):
                self.infer(image_dir, output_dir)
        else:
            output_file = os.path.join(saved_result_dir, output_name)

            if not os.path.exists(output_file):
                self.predict(image_dir, output_file)

        if root_model_dir:
            self.det_dict = u.backup(self.det_dict, root_model_dir)                        
        u.write_log(saved_result_dir, self.det_dict)
        
        return output_file

class MMOCRDetector():
    def __init__(self, det_dict):
        self.det_dict = det_dict

    def predict(self, image_dir, output_file, saved_result_dir = None):
        os.chdir("/content/bkai/mmocr")
        from mmocr.utils.ocr import MMOCR
        
        model_dir = self.det_dict['directory']
        checkpoint_file =  os.path.join(model_dir, self.det_dict['checkpoint_name'] + ".pth")
        config_file = os.path.join(model_dir, self.det_dict['config_name'] + ".py")

        det_results = os.path.join(saved_result_dir, "det_results")
        os.makedirs(det_results, exist_ok = True)
        
        contents = ""
        ocr = MMOCR(recog=None, det_config=config_file, det_ckpt=checkpoint_file)
        for image_path in glob.glob(image_dir + "/*"):
            results = ocr.readtext(image_path)

            image_name_ext = os.path.basename(image_path)
            saved_image_path = os.path.join(det_results, image_name_ext)

            dic_list = []
            image = cv2.imread(image_path)
            for polygon in results[0]['boundary_result']:
                polygon = polygon[:-1]
                score = polygon[-1]

                if score < 0.7:
                    continue
                polygon = np.reshape(polygon, (-1,2)).tolist()
                image = u.plot_polygon(image, polygon)

                dic = {}
                dic['transcription'] = ""
                dic['points'] = polygon
                dic['difficult'] = False

                dic_list.append(dic)
            
            cv2.imwrite(saved_image_path, image)
            text = u.dict_list_to_text(dic_list)
            contents += image_path + "\t" + text + "\n"

                
        with open(output_file, "w") as f:
            f.write(contents[:-1])
            f.close()

    def call(self, image_dir, saved_result_dir, root_model_dir = None, output_name = "det.txt"):    
        output_file = os.path.join(saved_result_dir, output_name)

        if not os.path.exists(output_file):
            self.predict(image_dir, output_file, saved_result_dir)

        if root_model_dir:
            self.det_dict = u.backup(self.det_dict, root_model_dir)  
                              
        u.write_log(saved_result_dir, self.det_dict)
        return output_file

#======================================================================Predictor================================================
class DetPredictor():
    def __init__(self, image_dir, saved_result_dir, root_model_dir = None):
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
    
    def process_det(self, croped_dir, output_file, writer = None):
        os.makedirs(croped_dir, exist_ok = True)
        reader = FormatReader(output_file)
        dictionary = reader.read_det(self.image_dir)

        saved_vis = os.path.join(self.saved_result_dir, "det_merge_results")
        os.makedirs(saved_vis, exist_ok = True)

        image_list = glob.glob(self.image_dir + "/*")
        predict_image_list = []
        for image_path, annotation_list in dictionary.items():
            if annotation_list == None:
                continue
            predict_image_list.append(image_path)

            image = cv2.imread(image_path)

            temp_idxs = []
            for idx1, anno1 in enumerate(annotation_list):
                if idx1 in temp_idxs:
                    continue
                #process polygon
                poly1 = anno1["points"]
                for idx2, anno2 in enumerate(annotation_list[idx1+1:]):
                    poly2 = anno2["points"]

                    try:
                        ios = u.ios(poly1, poly2)
                    except:
                        continue

                    if ios < 0:
                        temp_idxs.append(idx1+1+idx2)
                    else:
                        if ios > 0.2:
                            poly1 = u.merge_box(poly1, poly2)
                            temp_idxs.append(idx1+1+idx2)

                image = u.plot_polygon(image, poly1)

                #crop image
                image_filename = os.path.basename(image_path)
                image_name, extension = os.path.splitext(image_filename)
                
                crop_image_filename = image_name + f"_{idx1}{extension}"
                crop_image_file = os.path.join(croped_dir, crop_image_filename)

                if not os.path.exists(crop_image_file):
                    croper = ImageCroper(image_path)
                    croped_image = croper.crop_rectangle(poly1)
                    cv2.imwrite(crop_image_file, croped_image)

                    #store info
                    width, height = imagesize.get(image_path)
                    anno1["points"] = [u.process_point(point, height, width) for point in poly1]

                if writer:
                    # writer.record_paddle(image_filename, anno)
                    writer.record_bk(image_filename, anno1["points"])
            
            saved_image_path = os.path.join(saved_vis, image_filename)
            cv2.imwrite(saved_image_path, image)

        missed_images = list(set(image_list) - set(predict_image_list))
        
        return missed_images
        
def process_missed_det(missed_images, missed_dir, missed_det_dict,
                       output_file, root_model_dir = None):                        
    #Move missed_image to dir
    os.makedirs(missed_dir, exist_ok = True)
    for image_path in missed_images:
        shutil.copy(image_path, missed_dir)
    
    deter = DetPredictor(missed_dir, missed_dir, root_model_dir = root_model_dir)
    missed_output_file = deter.detect(missed_det_dict)
    u.merge_two_file(output_file,  missed_output_file, output_file)

    return missed_output_file
