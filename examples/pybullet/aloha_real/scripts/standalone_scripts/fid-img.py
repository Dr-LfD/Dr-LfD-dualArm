import os, sys, cv2
import numpy as np
import torch
from pytorch_fid import fid_score, inception

class sam_module():
    def __init__(self, text_prompt):
        self.text_prompt = text_prompt
        self.sam_path = '/home/xuhang/Desktop/yzchen_ws/Grounded-SAM-2/'
        sys.path.insert(0, self.sam_path)
        os.chdir(self.sam_path)
        from aloha_segmenter import SAMTrackPipeline
        self.network = SAMTrackPipeline(text_prompt = self.text_prompt, source_video_frame_dir = "./custom_video_frames", save_tracking_results_dir = "./tracking_results" )
        sys.path.pop()
        print("SAM server initialized")



    def save_masked_img(self, img_path, output_dir):
        def fill_mask_holes(mask):
            # 将 float32 类型的 mask 转换为二值图像 (uint8)
            _, mask_binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            mask_binary = mask_binary.astype(np.uint8)
            
            mask_inv = cv2.bitwise_not(mask_binary)
            
            contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_mask = np.zeros_like(mask_binary)
            
            # 遍历所有轮廓
            for i, contour in enumerate(contours):
                # 只填充没有子轮廓的轮廓（即真正的空洞）
                if hierarchy[0][i][2] == -1:  # 如果没有子轮廓
                    cv2.drawContours(filled_mask, [contour], 0, 255, -1)
            
            filled_mask = cv2.bitwise_not(filled_mask)
            
            result = cv2.bitwise_or(mask_binary, filled_mask)
            
            result = result.astype(np.float32) / 255.0
            return result

        try:
            rgb_image = cv2.imread(img_path)
            img_file = os.path.basename(img_path)
            self.network.save_images_frames([rgb_image])                
            masks, objects = self.network.get_img_masks()
            for i in range(len(objects)):

                # filled_mask = fill_mask_holes(masks[i])
                # ## dilate the mask
                # kernel = np.ones((5,5),np.uint8)
                # masks_dilated = cv2.dilate(filled_mask, kernel, iterations = 1)
                masked_img = rgb_image * masks[i][:, :, None]
                save_path = os.path.join(output_dir, objects[i]+ '_masked' +img_file)
                cv2.imwrite(save_path, masked_img)

        except:
            print("Error in SAM")
            return None, None
        
    def segment_imgs_from_demo(self, base_folder):
        for i in range(30):
            demo_img_folder = f'{base_folder}/episode_{i}_wrist_images'
            if not os.path.exists(demo_img_folder):
                print(f'{demo_img_folder} does not exist')
                continue
            sub_folders = os.listdir(demo_img_folder)
            for sub_folder in sub_folders:
                if 'masked' in sub_folder:
                    continue
                img_folder = f'{demo_img_folder}/{sub_folder}'
                output_dir = f'{demo_img_folder}/{sub_folder}_masked'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                img_files = os.listdir(img_folder)
                for img_file in img_files:
                    img_path = f'{img_folder}/{img_file}'
                    self.save_masked_img(img_path, output_dir)

def compute_fid_by_folders(img_folder1, img_folder2):
    fid_out = fid_score.calculate_fid_given_paths([img_folder1, img_folder2], 32, 'cuda', 2048)
    return fid_out

def compute_internal_fid_threthold(folder_list, pair_num = 10, threthold_percentile = 90):
    internal_fid = []
    for _ in range(pair_num):
        i, j = np.random.choice(len(folder_list), 2, replace=False)
        fid_out = compute_fid_by_folders(folder_list[i][1], folder_list[j][1])
        internal_fid.append((folder_list[i][0], folder_list[j][0], fid_out))


    threthold = np.percentile(internal_fid, threthold_percentile)
    return threthold, internal_fid
    
def get_demo_folders(base_folder):
    demo1_img_folders = []
    demo2_img_folders = []

    for i in range(30):
        # if i == 16: ## manual delete bad demo
        #     continue
        demo_img_folder = f'{base_folder}/episode_{i}_wrist_images'
        if not os.path.exists(demo_img_folder):
            print(f'{demo_img_folder} does not exist')
            continue
        sub_folders = os.listdir(demo_img_folder)
        for sub_folder in sub_folders:
            if 'masked' not in sub_folder:
                continue
            if 'right' in sub_folder:
                demo1_img_folders.append((i,f'{demo_img_folder}/{sub_folder}'))
            elif 'left' in sub_folder:
                demo2_img_folders.append((i,f'{demo_img_folder}/{sub_folder}'))
    return demo1_img_folders, demo2_img_folders

# def compute_gaussian_statistics(demo_folders, model):
#     demo_statistics = []
#     for i in range(len(demo_folders)):
#         id, folder_name = demo_folders[i]
#         mean, var = fid_score.compute_statistics_of_path(folder_name, model=model, batch_size=32, device='cuda', dims=2048)
#         demo_statistics.append((id, mean, var))
#     return demo_statistics

def compute_internal_fid(base_folder, demo1_img_folders, demo2_img_folders):
    demo1_threth, demo1_fids = compute_internal_fid_threthold(demo1_img_folders, pair_num=100, threthold_percentile = 90)
    ## save the threthold and fids into json
    import json
    with open(os.path.join(base_folder, 'demo1_masked_fid.json'), 'w') as f:
        json.dump({'threthold': demo1_threth, 'fids': demo1_fids}, f)
    demo2_threth, demo2_fids = compute_internal_fid_threthold(demo2_img_folders, pair_num= 100, threthold_percentile = 90)

    with open(os.path.join(base_folder, 'demo2_masked_fid.json'), 'w') as f:
        json.dump({'threthold': demo2_threth, 'fids': demo2_fids}, f)

    # print('demo1 threth:', demo1_threth, '; fid_list:', demo1_fids)
    # print('demo2 threth:', demo2_threth, '; fid_list:', demo2_fids)
def load_inception():
    # 加载预训练的 InceptionV3 模型
    model = inception.InceptionV3([3])  # [3] 表示获取 InceptionV3 的第三个池化层的特征
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置为推理模式
    return model

def copy_files_to_out_folder(demo_folders, out_folder):
    import shutil
    # 确保输出文件夹存在
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # 遍历每个文件夹
    for folder in demo_folders:
        # 遍历文件夹中的每个文件
        for root, dirs, files in os.walk(folder):
            for file in files:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 构造目标路径
                dest_path = os.path.join(out_folder, file)
                # 如果目标路径已存在同名文件，重命名
                if os.path.exists(dest_path):
                    base_name, extension = os.path.splitext(file)
                    counter = 1
                    # 生成新的文件名，直到找到一个不冲突的名字
                    while os.path.exists(dest_path):
                        new_name = f"{base_name}_{counter}{extension}"
                        dest_path = os.path.join(out_folder, new_name)
                        counter += 1
                # 复制文件
                shutil.copy2(file_path, dest_path)
                print(f"已复制文件 {file} 到 {dest_path}")

if __name__ == '__main__':

    # sam_module = sam_module('mug.')
    # sam_module.segment_imgs_from_demo('/home/xuhang/interbotix_ws/src/pddlstream_aloha/hdf5_out/transfer_cup')

    base_folder = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/hdf5_out/transfer_cup'
    demo1_img_folders, demo2_img_folders = get_demo_folders(base_folder)

    compute_internal_fid(base_folder, demo1_img_folders, demo2_img_folders)

    all_demo1_folder = os.path.join(base_folder, 'right_pre_masked')
    all_demo2_folder = os.path.join(base_folder, 'left_eff_masked')
    copy_files_to_out_folder([x[1] for x in demo1_img_folders], all_demo1_folder)
    copy_files_to_out_folder([x[1] for x in demo2_img_folders], all_demo2_folder)

    fid_all = compute_fid_by_folders(all_demo1_folder, all_demo2_folder)
    print(f'overall fid is {fid_all}')





                
