import os, glob, shutil, json, copy
import utility as u
from format_processor import FormatReader

class PaddleRecognitor():
    def __init__(self, rec_dict):
        self.rec_dict = rec_dict
        
    def predict(self, image_dir, output_file, use_gpu = True):
        model_dir = self.rec_dict['directory']
        checkpoint_filename =  os.path.join(model_dir, self.rec_dict['checkpoint_name'])
        config_file = os.path.join(model_dir, self.rec_dict['config_name'] + ".yml")
        
        modify = ['python', 'tools/infer_rec.py',
                  f'-c {config_file}',
                  f'-o Global.infer_img={image_dir}',
                  f'Global.checkpoints={checkpoint_filename}',
                  f'Global.save_res_path={output_file}',
                  f'Global.use_gpu={use_gpu}',
                  f'Global.infer_mode=True']
        command = " ".join(modify)
        
        #Run script
        os.chdir("../PaddleOCR_2.4")
        u.bash_script(command)        

    def infer(self, image_dir, output_file, use_gpu = True):
        model_dir = self.rec_dict['directory']
        algorithm = self.rec_dict['algorithm']
        
        image_shape = self.rec_dict['image_shape']
        max_length = self.rec_dict['max_length']
        char_dict = self.rec_dict['char_dict']
        use_space_char = self.rec_dict['use_space_char']
        
        modify = ['python','tools/predict_rec.py',
                  f'--image_dir={image_dir}',
                  f'--rec_algorithm={algorithm}',
                  f'--rec_model_dir={model_dir}',
                  f'--rec_saved_path={output_file}',
                  f'--rec_image_shape={image_shape}',
                  f'--max_text_length={max_length}',
                  f'--use_space_char={use_space_char}',
                  f'--use_gpu={use_gpu}']
        command = " ".join(modify)

        #Run script
        os.chdir("../PaddleOCR_2.4")
        u.bash_script(command)        
        
    def call(self, image_dir, saved_result_dir, root_model_dir = None, output_name = "rec.txt"):
        output_file = os.path.join(saved_result_dir, output_name)    
        if self.rec_dict['checkpoint_name'] == "inference":
            if not os.path.exists(output_file):
                self.infer(image_dir, output_file)

        else:
            if not os.path.exists(output_file):
                self.predict(image_dir, output_file)

        if root_model_dir:
            self.rec_dict = u.backup(self.rec_dict, root_model_dir)                        
        u.write_log(saved_result_dir, self.rec_dict)
        
        return output_file
        
class MMOCRRecognitor():
    def __init__(self, rec_dict):
        self.rec_dict = rec_dict

    def predict(self, image_dir, output_file):
        os.chdir("/content/bkai/mmocr")
        from mmocr.utils.ocr import MMOCR
        
        model_dir = self.rec_dict['directory']
        checkpoint_file =  os.path.join(model_dir, self.rec_dict['checkpoint_name'] + ".pth")
        config_file = os.path.join(model_dir, self.rec_dict['config_name'] + ".py")
        
        text_list = []
        ocr = MMOCR(recog=None, det_config=config_file, det_ckpt=checkpoint_file)
        for image_path in glob.glob(image_dir + "/*"):
            image_filename = os.path.basename(image_path)
            results = ocr.readtext(image_path, details=True)[0]

            content = [image_filename, results["text"], str(results["score"])]
            text = "\t".join(content) + "\n"
            text_list.append(text)

            print(text)
        
        text_list[-1] = text_list[-1][:-1]
        with open(output_file, "w") as f:
            f.writelines(text_list)
            f.close()

    def call(self, image_dir, saved_result_dir, root_model_dir = None, output_name = "rec.txt"):
        output_file = os.path.join(saved_result_dir, output_name)    
        if not os.path.exists(output_file):
            self.predict(image_dir, output_file)

        if root_model_dir:
            self.rec_dict = u.backup(self.rec_dict, root_model_dir)                        
        u.write_log(saved_result_dir, self.rec_dict)
        
        return output_file
class RecPredictor():
    def __init__(self, image_dir, saved_result_dir, root_model_dir = None):
        self.saved_result_dir = saved_result_dir
        self.root_model_dir = root_model_dir
        
        self.image_dir = image_dir
        
    
    def recognize(self, rec_dict, output_name = "rec.txt"):
        if rec_dict['module'] == "paddle":
            rec = PaddleRecognitor(rec_dict)
            rec_file = rec.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
            print("OK")
        if rec_dict['module'] == "mmocr":
            rec = MMOCRRecognitor(rec_dict)
            rec_file = rec.call(self.image_dir, self.saved_result_dir, self.root_model_dir, output_name)
            
        return rec_file
        
    def process_rec(self, output_file, threshold = 0.8, writer = None):
        reader = FormatReader(output_file)
        dictionary = reader.read_recog(self.image_dir)
        
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
            
            if writer:
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