import os
from dashscope import MultiModalConversation

# https://help.aliyun.com/zh/model-studio/vision?spm=5176.28630291.0.0.24bf7eb5h3MdqT&disableWebsiteRedirect=true#6b5c3f098fjfc

import requests
import time
import cv2
import json
import click
from tqdm import tqdm

def video_understanding(video_path, prompt, fps):
    if isinstance(video_path, str):
        # print("loading a video:", video_path)
        # 处理视频时使用fps参数，表示每隔1/fps 秒抽取一帧
        content = [{'video': video_path, "fps": fps}, {'text': prompt}]
    else:
        # print("loading a list of images")
        # 处理图像列表时不使用fps参数，让API分析所有图像，to 避免二次降维
        content = [{'video': video_path, "fps": 1}, {'text': prompt}]

    messages = [
        {
            'role': 'system', 
            'content': [
                {'text': 'You are a helpful assistant.'}
            ]
        },
        {
            'role':'user',
            'content': content
        }
    ]

    # 添加重试机制
    max_retries = 3
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
                # api_key=os.getenv('DASHSCOPE_API_KEY'),
                api_key="sk-3415ed083d5349f990717cb39e4a0411",
                model='qwen2.5-vl-72b-instruct',
                messages=messages)
            
            # 如果调用成功，尝试解析响应
            json_output = response["output"]["choices"][0]["message"].content[0]["text"] # type: ignore
            return json_output
            
        except requests.exceptions.SSLError as e:
            print(f"SSL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("所有重试都失败了, 返回None")
                return None
                
        except Exception as e:
            print(f"其他错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("所有重试都失败了, 返回None")
                return None
    
    return None

def save_video_frames(video_path, output_folder, target_fps=None):
    """
    Save frames from a video file as images with a specific FPS.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save the extracted frames
        target_fps (float): Desired frames per second (if None, saves all frames)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # delete all files in the folder
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"Original FPS: {original_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # If target_fps is None, save all frames
    if target_fps is None:
        target_fps = original_fps
    
    # Calculate frame interval based on target FPS
    frame_interval = max(1, int(round(original_fps / target_fps)))
    print(f"Saving 1 frame every {frame_interval} frames")
    
    frame_count = 0
    saved_count = 0
    
    image_path_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            image_path_list.append(frame_filename)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Finished processing. Saved {saved_count} frames.")
    return image_path_list


def save_text_to_json(data, output_file, indent=4):
    """
    Save the data as a JSON file.
    
    Args:
    - data (dict): The data structure to be saved.
    - output_file (str): The path of the output JSON file.
    - indent (int): The number of spaces for JSON indentation, with a default value of 4.
    """
    try:
        # 写入 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        # print(f"\nSuccessfully saved the data to {output_file}")
    except Exception as e:
        print(f"\nError saving file: {e}")




def frameidx_to_timestep(frame_idx, fps, video_path):
    """
    Convert frame index to time step in seconds.
    
    Args:
        frame_idx (int): The index of the frame.
        fps (float): Frames per second of the video.
    
    Returns:
        float: Time step in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS) ## input_video 的原始帧率，即每秒帧数
    
    frame_interval = original_fps / fps
    time_step = frame_idx * frame_interval

    return time_step




if __name__ == "__main__":

    # task_name = 'two_arm_threading'
    task_name = 'two_arm_three_piece_assembly'
    # object_list = ['cream_cheese_box', 'butter', 'basket', 'table']
    robot_list = ['robot0'] if 'two_arm' not in task_name else ['robot_left', 'robot_right']

    action_template = {"grasp": [[["?r", "?o"], "add"], [["?s", "?o"], "remove"]], "place": [[["?r", "?o"], "remove"], [["?o", "?s"], "add"]]}

    cam_res_type = 'agentview128'
    local_path = f"/home/user/yzchen_ws/TAMP-ubuntu22/demo_understanding/dataset/{cam_res_type}/{task_name}_demovids/demo_0.mp4"
    qwen_vid_path =f"file://{local_path}"

    if not os.path.exists(local_path):
        raise ValueError(f"Error: {local_path} does not exist")


    use_image_list = True
    target_fps = 2  # Set to None to save all frames

    if use_image_list:
        output_folder = local_path.split('.')[0]
        input_path = save_video_frames(local_path, output_folder, target_fps)
    else:
        input_path = qwen_vid_path


# Objects in the scene: {object_list}
    # description = 'left robot picks up the tripod, right robot picks up the needle, and perform bimanual threading'
    description = 'left robot picks up piece_1, right robot picks up piece_2, and perform bimanual assembly to the base'

    prompt = f"""
Task: {task_name}
Robots in the scene: {robot_list}
Action template: {action_template}
Description: {description}

Answer the following questions based on the given video:

1. Using the task name, description, and action template, analyze the video and output a sequence of actions without placeholders in the action template. 
   - An action comprises a list of contact changes. 
   - A contact change must be a JSON object with the fields:
       - "subject": the first entity (robot or object)
       - "object": the second entity (robot or object)
       - "op": either "add" or "remove"
       - "frame": an integer frame number
   - "add" means the contact is established, "remove" means the contact is broken.
   - Variables (e.g., ?r, ?o, ?s) must be replaced with actual robots or objects from the scene. 
   - No explanation needed.

2. Use temporal analysis to determine when each contact change happens. 
   - The time must be given as an integer frame number of the video.

The output must be valid JSON. Do not include comments or text outside the JSON.

Example output:
```json
{{
    "actions": [
        {{
            "action_number": 1,
            "contact_changes": [
                {{ "subject": "robot0", "object": "obj1", "op": "add", "frame": 12 }},
                {{ "subject": "obj1", "object": "obj2", "op": "remove", "frame": 25 }}
            ],
            "description": "robot0_grasp_obj1"
        }},
        {{
            "action_number": 2,
            "contact_changes": [
                {{ "subject": "obj4", "object": "obj3", "op": "add", "frame": 40 }}
            ],
            "description": "obj4_attach_obj3"
        }}
    ]
    
}}
```

"""


    result = video_understanding(input_path, prompt, fps=1) ## NOTE: fps can only be 1, as frame of fps has been extracted in the save_video_frames function.
    print(result)

    # Save the result to a JSON file
    if result is not None:
        json_str = "\n".join(result.split("```")[1].split("\n")[1:])
        json_output = json.loads(json_str)
    else:
        json_output = {}

    json_path = os.path.join(output_folder, f"seg_result.json")
    save_text_to_json(json_output, json_path, indent=4)