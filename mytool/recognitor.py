import os, glob, shutil, json, copy
from utility import backup
from format_processor import FormatReader

class PaddleRecognitor():
    def __init__(self, rec_dict):
        self.rec_dict = rec_dict
        
    def predict(self, image_dir, output_file, use_gpu = True):
        model_dir = self.rec_dict['directory']
        checkpoint_filename =  os.path.join(model_dir, self.rec_dict['checkpoint_name'])
        config_file = os.path.join(model_dir, self.rec_dict['config_name'] + ".yml")
        
        modify = ['python script/PaddleOCR_2.4/tools/infer_rec.py',
                  f'-c {config_file}',
                  f'-o Global.infer_img={image_dir}',
                  f'Global.checkpoints={checkpoint_filename}',
                  f'Global.save_res_path={output_file}',
                  f'Global.use_gpu={use_gpu}',
                  f'Global.infer_mode=True']
        command = " ".join(modify)
        os.system(command)
        
    def infer(self, image_dir, output_file, use_gpu = True):
        model_dir = self.rec_dict['directory']
        algorithm = self.rec_dict['algorithm']
        
        image_shape = self.rec_dict['image_shape']
        max_length = self.rec_dict['max_length']
        char_dict = self.rec_dict['char_dict']
        use_space_char = self.rec_dict['use_space_char']
        
        modify = ['python script/PaddleOCR_2.4/tools/predict_det.py',
                  f'--image_dir={image_dir}',
                  f'--rec_algorithm={algorithm}',
                  f'--rec_model_dir={model_dir}',
                  f'--rec_saved_path={output_file}',
                  f'--rec_image_shape={image_shape}',
                  f'--max_text_length={max_length}',
                  f'--use_space_char={use_space_char}',
                  f'--use_gpu={use_gpu}']
        command = " ".join(modify)
        os.system(command)
        
    def call(self, image_dir, saved_result_dir, root_model_dir, output_name = "rec.txt"):
        output_file = ""
        log = {}
        
        #predict 
        checkpoint_name = self.rec_dict['checkpoint_name']
        output_file = os.path.join(saved_result_dir, output_name)
        
        if checkpoint_name == "inference":
            if not os.path.exists(output_file):
                self.infer(image_dir, output_file)
                
                #back_up
                model_dir = self.rec_dict['directory']
                saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
                os.makedirs(saved_model_dir, exist_ok = True)
                
                ##checkpoint
                checkpoint_filename =  os.path.join(model_dir, checkpoint_name)
                post_fix = backup(checkpoint_filename, saved_model_dir,
                                   module = "paddle", problem_type = "rec")
                
        else:
            if not os.path.exists(output_file):
                self.predict(image_dir, output_file)
                
                #back_up
                model_dir = self.rec_dict['directory']
                saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
                os.makedirs(saved_model_dir, exist_ok = True)
                
                ##checkpoint
                config_name = self.rec_dict['config_name']
                config_filename = os.path.join(model_dir, config_name)
                
                checkpoint_filename = os.path.join(model_dir, checkpoint_name)
                post_fix = backup(checkpoint_filename, saved_model_dir,
                                  config_filename = config_filename,
                                  module = "paddle", problem_type = "rec")
                
                log['config_name'] = config_name + f"_{post_fix}"
        
        #write_log      
        log['model_dir'] = saved_model_dir
        log['checkpoint_name'] = checkpoint_name + f"_{post_fix}"
        
        log_file = os.path.join(saved_result_dir, "log.json")
        log_object = json.dump(log, indent = 4)
        f = open|(log_file).write(log_object)
        
        return output_file
        
class MMOCRRecognitor():
    def __init__(self, rec_dict):
        from mmocr.utils.ocr import MMOCR
        self.rec_dict = rec_dict

    def predict(self, image_dir, output_file):
        from mmocr.utils.ocr import MMOCR
        
        model_dir = self.rec_dict['directory']
        checkpoint_file =  os.path.join(model_dir, self.rec_dict['checkpoint_name'] + ".pth")
        config_file = self.rec_dict['config_filename'] + ".py"
        
        text_list = []
        ocr = MMOCR(recog=None, det_config=config_file, det_ckpt=checkpoint_file)
        for image_path in glob.glob(image_dir + "/*"):
            image_filename = os.path.basename(image_path)
            results = self.predict_one_image(ocr, image_path)

            content = [image_filename, results["text"], str(results["score"])]
            text = "\t".join(content) + "\n"
            text_list.append(text)

            print(text)
        
        text_list[-1] = text_list[-1][:-1]
        with open(output_file, "w") as f:
            f.writelines(text_list)
            f.close()

    def call(self, image_dir, saved_result_dir, root_model_dir, output_name = "rec.txt"):
        log = {}
        
        #predict
        output_file = os.path.join(saved_result_dir, output_name)
        if not os.path.exists(output_file):
            self.predict(image_dir, output_file)

            #back_up
            model_dir = self.rec_dict['directory']
            saved_model_dir = os.path.join(root_model_dir, os.path.basename(model_dir))
            os.makedirs(saved_model_dir, exist_ok = True)
            
            ##checkpoint
            config_filename = self.rec_dict['config_filename']
            config_name = os.path.basename(config_filename)
            
            checkpoint_name = self.rec_dict['checkpoint_name']
            checkpoint_filename =  os.path.join(model_dir, checkpoint_name)
            post_fix = backup(checkpoint_filename, saved_model_dir,
                              config_filename = config_filename,
                              module = "mmocr", problem_type = "rec")
        
        #write_log      
        log['model_dir'] = saved_model_dir
        log['checkpoint_name'] = checkpoint_name + f"_{post_fix}"
        
        log_file = os.path.join(saved_result_dir, "log.json")
        log_object = json.dump(log, indent = 4)
        f = open|(log_file).write(log_object)
        
        return output_file
        
class RecPredictor():
    def __init__(self, image_dir, saved_result_dir, root_model_dir):
        self.saved_result_dir = saved_result_dir
        self.root_model_dir = root_model_dir
        
        self.image_dir = image_dir
        
    
    def recognize(self, rec_dict, output_name = "rec.txt"):
        if rec_dict['module'] == "paddle":
            rec = PaddleRecognitor(rec_dict)
            rec_file = rec.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
        
        if rec_dict['module'] == "mmocr":
            rec = MMOCRRecognitor(rec_dict)
            rec_file = rec.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
            
        return rec_file
        
    def process_rec(self, writer, output_file, threshold = 0.9):
        reader = FormatReader(output_file)
        dictionary = reader.read_recog(self.croped_dir)
        
        lower_confidence_images = []
        for croped_image_path, annotation in dictionary.items():
            if annotation["score"] <= threshold :
                transcription = 0
                lower_confidence_images.append(croped_image_path)
            else:
                transcription = annotation["label"]
            
            #record rec info
            croped_image_filename = os.path.basename(croped_image_path)
            croped_image_name, extension = os.path.splitext(croped_image_filename)
            
            idx = int(croped_image_name.split("_")[-1])
            original_image_filename = "_".join(croped_image_name.split("_")[:-1]) + extension
            
            writer.record_paddle(original_image_filename, idx = idx, transcription = transcription)
            writer.record_bk(original_image_filename, idx = idx, transcription = transcription)
        
        return lower_confidence_images
        
def esemble_rec(rec_file, rec_dict_list):
    dictionary = copy.deepcopy(rec_dict_list[0])
    for image_path, info in dictionary.items():
        for rec_dict in rec_dict_list[1:]:
            if rec_dict[image_path]['score'] > info['score']:
                dictionary[image_path] = rec_dict[image_path]
                print(info, rec_dict[image_path])
                    
    contents = []
    for image_path, info in dictionary.items():
        text = [image_path, info['label'], str(info['score'])]
        text = "\t".join(text) + "\n"
        contents.append(text)
    
    contents[-1] = contents[-1][:-1]
    with open(rec_file, "w") as f:
        f.writelines(contents)
        f.close()