from qwen import QWen2_5VLInfer
from llava import LlavaInfer
import json, yaml, argparse, os, re
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

def read_json_file(file_path):
    """
    遍历文件内容，提取每行的键值对，并以原名保存到字典中。

    :param file_path: 要读取的文件路径
    :return: 包含所有键值对的字典列表
    """
    # try:
    with open(file_path, 'r', encoding='utf-8') as file:
        # 初始化一个空列表，用于存储每行的字典
        data_list = []
        # 按行读取文件内容
        for line in file:
            # 将每行内容解析为JSON对象
            data = json.loads(line)
            # 将解析后的字典添加到列表中
            data_list.append(data)
    return data_list
    # except FileNotFoundError:
    #     print(f"文件 {file_path} 未找到。")
    #     return []
    # except json.JSONDecodeError as e:
    #     print(f"解析JSON时出错: {e}")
    #     return []
    # except Exception as e:
    #     print(f"发生错误: {e}")
    #     return []

# # 调用函数，传入文件路径
# file_path = 'dataset/twitter/test.txt'  # 替换为你的文件路径
# result = read_json_file(file_path)

# # 打印结果
# for item in result:
#     print(item)
def qwen_generate_img_txt_comb(result_file = "qwen_result_infer.json"):
    text_file = "twitter_train.txt"
    image_root = "dataset/twitter/"
    items = read_json_file(text_file)
    qwenllm = QWen2_5VLInfer()
    qwenllm.initialize(model_id="models/qwen2.5vl-7b")
    for item in items:
        image_path = os.path.join(image_root, item['id'])
        txt = item['text']
        sd_label = [1,0][item['information_label']]
        ground_truth_image = item['image_label']
        ground_truth_text = item['text_label']
        ground_truth = item['label']
        message = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                ] 
                + 
                # [
                #     {"type": "text", "text": "Here is a demonstration:\n Image:"}
                # ] 
                # + 
                # [
                #     {"type": "image", "image": "demo_image.png"}
                # ] 
                # +
                # [
                #     {"type": "text", "text": "Text:\n The Windows background is on fire in California..."}
                # ] 
                # + 
                # [
                #     {"type": "text", "text": "Answer: The overall sentiment expressed by the combination is negative. Although the picture is just the classic Windows XP wallpaper, which conveys a plastic, sensitive feel, the text expresses a sense of negativity due to concerns about fire hazards on the landscape of the wallpaper. Therefore, the overall sentiment conveyed by the combination should be considered negative."}
                # ] 
                # + 
                [
                    {"type": "text", "text": "Please determine the sentiment expressed by the following combination (positive, negative, or neutral). "}
                ]
                +
                [
                    {"type": "text", "text": "Image: \n"}
                ]
                +
                [
                    {"type": "image", "image": image_path}
                ]
                +
                [
                    {"type": "text", "text": "Text: \n"}
                ]
                +
                [
                    {"type": "text", "text": txt}
                ]
                +
                [
                    {"type": "text", "text": "Your answer should be: Image:(sentiment), Text:(sentiment), Combination:(sentiment)"}
                ]
            }
        ]
        qwenllm.update(message = message)
        qwenllm.infer()
        print(qwenllm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image_path": image_path,
            "txt": txt,
            "sd_label": sd_label,
            "ground_truth": ground_truth,
            "ground_truth_image": ground_truth_image,
            "ground_truth_text": ground_truth_text,
            "qwen_output": qwenllm.output_text
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
            
def qwen_generate_demo(result_file = "qwen_result_demo_infer.json"):
    text_file = "dataset/twitter/test.txt"
    image_root = "dataset/twitter/"
    items = read_json_file(text_file)
    qwenllm = QWen2_5VLInfer()
    qwenllm.initialize(model_id="models/qwen2.5vl-7b")
    for item in items:
        image_path = os.path.join(image_root, item['id'])
        txt = item['text']
        sd_label = [1,0][item['information_label']]
        ground_truth = item['label']
        message = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                ] 
                + 
                [
                    {"type": "text", "text": "Here is a demonstration:\n Image:"}
                ] 
                + 
                [
                    {"type": "image", "image": "demo_image.png"}
                ] 
                +
                [
                    {"type": "text", "text": "Text:\n The Windows background is on fire in California..."}
                ] 
                + 
                [
                    {"type": "text", "text": "Answer: The overall sentiment expressed by the combination is negative. Although the picture is just the classic Windows XP wallpaper, which conveys a plastic, sensitive feel, the text expresses a sense of negativity due to concerns about fire hazards on the landscape of the wallpaper. Therefore, the overall sentiment conveyed by the combination should be considered negative."}
                ] 
                + 
                [
                    {"type": "text", "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral). "}
                ]
                +
                [
                    {"type": "text", "text": "Image: \n"}
                ]
                +
                [
                    {"type": "image", "image": image_path}
                ]
                +
                [
                    {"type": "text", "text": "Text: \n"}
                ]
                +
                [
                    {"type": "text", "text": txt}
                ]
            }
        ]
        qwenllm.update(message = message)
        qwenllm.infer()
        print(qwenllm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image_path": image_path,
            "txt": txt,
            "sd_label": sd_label,
            "ground_truth": ground_truth,
            "qwen_output": qwenllm.output_text
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

def qwen_generate(result_file = "qwen_result_infer_0307.json"):
    text_file = "dataset/twitter/test.txt"
    image_root = "dataset/twitter/"
    items = read_json_file(text_file)
    qwenllm = QWen2_5VLInfer()
    qwenllm.initialize(model_id="models/qwen2.5vl-7b")
    for item in items:
        image_path = os.path.join(image_root, item['id'])
        txt = item['text']
        sd_label = [1,0][item['information_label']]
        ground_truth = item['label']
        message = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                ]  
                + 
                [
                    {"type": "text", "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral)."}
                ]
                +
                [
                    {"type": "text", "text": "Image: \n"}
                ]
                +
                [
                    {"type": "image", "image": image_path}
                ]
                +
                [
                    {"type": "text", "text": "Text: \n"}
                ]
                +
                [
                    {"type": "text", "text": txt}
                ]
                # +
                # [
                #     {"type": "text", "text": "Your answer should be: ...(analyze process) , the overall sentiment of the combination is ..."}
                # ]
                +
                [
                    {"type": "text", "text": "Your answer should be: Image:(sentiment), Text:(sentiment), Combination:(sentiment)"}
                ]
            }
        ]
        qwenllm.update(message = message)
        qwenllm.infer()
        print(qwenllm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image_path": image_path,
            "txt": txt,
            "sd_label": sd_label,
            "ground_truth": ground_truth,
            "qwen_output": qwenllm.output_text
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

def qwen_generate_retrieval_json(
        result_file = "qwen_result_infer_0307.json",
        retrieval_json_file = "retrieval_1shot_sd_feature.json",
        sd_label=1):
    with open(retrieval_json_file, "r", encoding="utf-8") as f:
        retri_data = json.load(f)
    image_root = "dataset/twitter/data_image/"
    qwenllm = QWen2_5VLInfer()
    qwenllm.initialize(model_id="models/qwen2.5vl-7b")
    for item in retri_data:
        query_image_path = os.path.join(image_root, item['query']['image'])
        query_txt = item['query']['text']
        query_ground_truth = item['query']['label']

        retri_image_paths = [os.path.join(image_root, image_path) for image_path in item['retrieval']['images']]
        retri_txts = item['retrieval']['texts']
        retri_ground_truths = item['retrieval']['raw_labels']

        # 根据检索结果构造示例组
        demonstrations = []
        for img_path, txt, label in zip(retri_image_paths, retri_txts, retri_ground_truths):
            demonstrations.extend([
                {"type": "text", "text": "Retrieved demonstration:"},
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Text: " + txt},
                {"type": "text", "text": "Label: " + label},
                {"type": "text", "text": "\n"}  # 分隔空行
            ])

        message = [
            {
                "role": "user",
                "content":
                    [
                        {"type": "text",
                         "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                    ]
                    +
                    [
                        {"type": "text", "text": "Here are demonstrations:\n"}
                    ]
                    +
                    demonstrations +
                    [
                        {"type": "text",
                         "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral). "}
                    ]
                    +
                    [
                        {"type": "text", "text": "Image: \n"}
                    ]
                    +
                    [
                        {"type": "image", "image": query_image_path}
                    ]
                    +
                    [
                        {"type": "text", "text": "Text: \n"}
                    ]
                    +
                    [
                        {"type": "text", "text": query_txt}
                    ]
            }
        ]
        qwenllm.update(message = message)
        qwenllm.infer()
        print(qwenllm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image": query_image_path,
            "txt": query_txt,
            "sd_label": sd_label,
            "ground_truth": query_ground_truth,
            "qwen_output": qwenllm.output_text
        }

        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

def llava_generate(result_file = "llava_result_infer_0307.json"):
    text_file = "dataset/twitter/test.txt"
    image_root = "dataset/twitter/"
    items = read_json_file(text_file)
    llavallm = LlavaInfer()
    llavallm.initialize(model_id="models/llava-1.5-7b-hf")
    for item in items:
        image_path = os.path.join(image_root, item['id'])
        txt = item['text']
        sd_label = [1,0][item['information_label']]
        ground_truth = item['label']
        message = [
            {
                "role": "user",
                "content": 
                [
                    {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                ]  
                + 
                [
                    {"type": "text", "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral)."}
                ]
                +
                [
                    {"type": "text", "text": "Image: \n"}
                ]
                +
                [
                    {"type": "image", "image": image_path}
                ]
                +
                [
                    {"type": "text", "text": "Text: \n"}
                ]
                +
                [
                    {"type": "text", "text": txt}
                ]
                +
                [
                    {"type": "text", "text": "Your answer should be: Image:(sentiment), Text:(sentiment), Combination:(sentiment)"}
                ]
            }
        ]
        llavallm.update(message = message)
        llavallm.infer()
        print(llavallm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image_path": image_path,
            "txt": txt,
            "sd_label": sd_label,
            "ground_truth": ground_truth,
            "llava_output": llavallm.output_text
        }
        
        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
            
def qwen_generate_adjust_demo(
    demo_image_path:str,
    demo_text:str,
    result_file:str,
    ):
    text_file = "dataset/twitter/test.txt"
    image_root = "dataset/twitter/"
    items = read_json_file(text_file)
    qwenllm = QWen2_5VLInfer()
    qwenllm.initialize(model_id="models/qwen2.5vl-7b")
    for item in items:
        image_path = os.path.join(image_root, item['id'])
        txt = item['text']
        sd_label = [1,0][item['information_label']]
        ground_truth = item['label']
        message = [
            {
                "role": "user",
                "content":
                [
                    {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
                ]
                +
                [
                    {"type": "text", "text": "Here is a demonstration:\n Image:"}
                ]
                +
                [
                    {"type": "image", "image": demo_image_path}
                ]
                +
                [
                    {"type": "text", "text": f"Text:\n {demo_text}"}
                ]
                +
                [
                    {"type": "text", "text": "Answer: The overall sentiment expressed by the combination is negative. Although the picture is just the classic Windows XP wallpaper, which conveys a plastic, sensitive feel, the text expresses a sense of negativity due to concerns about fire hazards on the landscape of the wallpaper. Therefore, the overall sentiment conveyed by the combination should be considered negative."}
                ]
                +
                [
                    {"type": "text", "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral). "}
                ]
                +
                [
                    {"type": "text", "text": "Image: \n"}
                ]
                +
                [
                    {"type": "image", "image": image_path}
                ]
                +
                [
                    {"type": "text", "text": "Text: \n"}
                ]
                +
                [
                    {"type": "text", "text": txt}
                ]
            }
        ]
        qwenllm.update(message = message)
        qwenllm.infer()
        print(qwenllm.output_text)

        # 保存image_path，txt，sd_label，ground_truth到json文件中
        result = {
            "image_path": image_path,
            "txt": txt,
            "sd_label": sd_label,
            "ground_truth": ground_truth,
            "qwen_output": qwenllm.output_text
        }
        with open(result_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

def calculate_metrics_llava(
        result_file= "llava_result_infer_0307.json",
        sentiment_position = "first",):
    # 打开并读取 .json 文件
    with open(result_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 定义情感关键词
    sentiment_keywords = ['neutral', 'positive', 'negative']
    # 用正则表达式创建一个模式，匹配所有情感关键词
    sentiment_pattern = r'\b(?:' + '|'.join(sentiment_keywords) + r')\b'

    # 存储情感标签结果和真实标签
    sentiment_results = []
    ground_truth = []
    sd_labels = []

    # 存储每个类别的样本数量
    category_counts = Counter()

    # 遍历每一条数据
    for entry in data:
        llava_output = entry.get('llava_output', "")
        sd_label = entry.get('sd_label', None)

        if sd_label is not None:
            sd_labels.append(sd_label)

        if sentiment_position == "last":
            # 处理情感标签
            matched_sentiments = []  # 存储所有匹配的情感结果

            if llava_output:
                matches = re.finditer(sentiment_pattern, llava_output.lower())  # 使用 finditer 获取所有匹配项
                for match in matches:
                    if match.lastindex:
                        matched_sentiments.append(match.group(match.lastindex))  # 获取最后一个捕获组
                    else:
                        matched_sentiments.append(match.group(0))  # 获取整个匹配项

                # 如果你只需要最后一个匹配的情感
                matched_sentiment = matched_sentiments[-1] if matched_sentiments else None
                # 存储结果
                sentiment_results.append(matched_sentiment)
            else:
                sentiment_results.append(None)
        elif sentiment_position == "first":
            # 处理情感标签
            matched_sentiments = []  # 存储所有匹配的情感结果

            if llava_output:
                matches = re.finditer(sentiment_pattern, llava_output.lower())  # 使用 finditer 获取所有匹配项
                for match in matches:
                    if match.lastindex:
                        matched_sentiments.append(match.group(match.lastindex))  # 获取最后一个捕获组
                    else:
                        matched_sentiments.append(match.group(0))  # 获取整个匹配项

                # 如果你只需要第一个匹配的情感
                matched_sentiment = matched_sentiments[0] if matched_sentiments else None
                # 存储结果
                sentiment_results.append(matched_sentiment)
            else:
                sentiment_results.append(None)



        # 存储真实标签
        if entry.get('ground_truth', None) is not None:
            ground_truth.append(sentiment_keywords[entry['ground_truth']])
        else:
            ground_truth.append(None)

        # 更新类别计数
        if sd_label is not None:
            category_counts[sd_label] += 1

    # 计算每个 sd_label 分组的 F1 和 Accuracy
    for label in set(sd_labels):
        # 过滤出该 sd_label 对应的样本
        filtered_results = [(sr, gt) for sr, gt, sd in zip(sentiment_results, ground_truth, sd_labels) if sd == label]
        if filtered_results:
            sentiment_results_filtered, ground_truth_filtered = zip(*filtered_results)
            f1 = f1_score(ground_truth_filtered, sentiment_results_filtered, average='weighted', labels=sentiment_keywords)
            accuracy = accuracy_score(ground_truth_filtered, sentiment_results_filtered)
            print(f"Label {label} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Label {label} - No valid data to compute metrics.")

    # 计算所有样本的 F1 和 Accuracy
    valid_results = [(sr, gt) for sr, gt in zip(sentiment_results, ground_truth) if sr is not None and gt is not None]
    if valid_results:
        sentiment_results_cleaned, ground_truth_cleaned = zip(*valid_results)
        overall_f1 = f1_score(ground_truth_cleaned, sentiment_results_cleaned, average='weighted', labels=sentiment_keywords)
        overall_accuracy = accuracy_score(ground_truth_cleaned, sentiment_results_cleaned)
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
    else:
        print("No valid data to compute overall metrics.")

    # 输出每个类别的样本个数
    print(f"Sample counts by sd_label: {dict(category_counts)}")

def calculate_metrics_qwen_retrieval(
        result_file= "qwen_results.json",
        sentiment_position = "first",):
    # 打开并读取 .json 文件
    with open(result_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 定义情感关键词
    sentiment_keywords = ['neutral', 'positive', 'negative']
    # 用正则表达式创建一个模式，匹配所有情感关键词
    sentiment_pattern = r'\b(?:' + '|'.join(sentiment_keywords) + r')\b'

    # 存储情感标签结果和真实标签
    sentiment_results = []
    ground_truth = []
    sd_labels = []

    # 存储每个类别的样本数量
    category_counts = Counter()

    # 遍历每一条数据
    for entry in data:
        qwen_output = entry.get('qwen_output', "")
        sd_label = entry.get('sd_label', None)

        if sd_label is not None:
            sd_labels.append(sd_label)

        if sentiment_position == "last":
            # 处理情感标签
            matched_sentiments = []  # 存储所有匹配的情感结果

            if qwen_output:
                matches = re.finditer(sentiment_pattern, qwen_output.lower())  # 使用 finditer 获取所有匹配项
                for match in matches:
                    if match.lastindex:
                        matched_sentiments.append(match.group(match.lastindex))  # 获取最后一个捕获组
                    else:
                        matched_sentiments.append(match.group(0))  # 获取整个匹配项

                # 如果你只需要最后一个匹配的情感
                matched_sentiment = matched_sentiments[-1] if matched_sentiments else None
                # 存储结果
                sentiment_results.append(matched_sentiment)
            else:
                sentiment_results.append(None)
        elif sentiment_position == "first":
            # 处理情感标签
            matched_sentiments = []  # 存储所有匹配的情感结果

            if qwen_output:
                matches = re.finditer(sentiment_pattern, qwen_output.lower())  # 使用 finditer 获取所有匹配项
                for match in matches:
                    if match.lastindex:
                        matched_sentiments.append(match.group(match.lastindex))  # 获取最后一个捕获组
                    else:
                        matched_sentiments.append(match.group(0))  # 获取整个匹配项

                # 如果你只需要第一个匹配的情感
                matched_sentiment = matched_sentiments[0] if matched_sentiments else None
                # 存储结果
                sentiment_results.append(matched_sentiment)
            else:
                sentiment_results.append(None)

        # 存储真实标签
        if entry.get('ground_truth', None) is not None:
            ground_truth.append(entry['ground_truth'].lower())
        else:
            ground_truth.append(None)

        # 更新类别计数
        if sd_label is not None:
            category_counts[sd_label] += 1

    # 计算每个 sd_label 分组的 F1 和 Accuracy
    for label in set(sd_labels):
        # 过滤出该 sd_label 对应的样本
        filtered_results = [(sr, gt) for sr, gt, sd in zip(sentiment_results, ground_truth, sd_labels) if sd == label]
        if filtered_results:
            sentiment_results_filtered, ground_truth_filtered = zip(*filtered_results)
            f1 = f1_score(ground_truth_filtered, sentiment_results_filtered, average='weighted', labels=sentiment_keywords)
            accuracy = accuracy_score(ground_truth_filtered, sentiment_results_filtered)
            print(f"Label {label} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Label {label} - No valid data to compute metrics.")

    # 计算所有样本的 F1 和 Accuracy
    valid_results = [(sr, gt) for sr, gt in zip(sentiment_results, ground_truth) if sr is not None and gt is not None]
    if valid_results:
        sentiment_results_cleaned, ground_truth_cleaned = zip(*valid_results)
        overall_f1 = f1_score(ground_truth_cleaned, sentiment_results_cleaned, average='weighted', labels=sentiment_keywords)
        overall_accuracy = accuracy_score(ground_truth_cleaned, sentiment_results_cleaned)
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
    else:
        print("No valid data to compute overall metrics.")

    # 输出每个类别的样本个数
    print(f"Sample counts by sd_label: {dict(category_counts)}")

# def find_understand():
#     # 读取文件，每一行为一条 JSON 数据
#     with open("qwen_result_infer.json", "r", encoding="utf-8") as f:
#         data = [json.loads(line) for line in f]

#     # 定义情感关键词列表（下标与真实标签数字对应：0 -> neutral, 1 -> positive, 2 -> negative）
#     sentiment_keywords = ['neutral', 'positive', 'negative']

#     fully_understood_indices = []  # 存储“全部理解”的样本索引
#     fully_understood_entries = []    # 存储“全部理解”的完整条目
#     fully_understood_count = Counter()  # 按组合真实标签统计“全部理解”样本数量

#     for i, entry in enumerate(data):
#         # 获取真实标签（数字形式），并转换为情感字符串
#         gt_comb = entry.get("ground_truth", None)
#         gt_image = entry.get("ground_truth_image", None)
#         gt_text = entry.get("ground_truth_text", None)

#         gt_comb_str = sentiment_keywords[gt_comb] if gt_comb is not None else None
#         gt_image_str = sentiment_keywords[gt_image] if gt_image is not None else None
#         gt_text_str = sentiment_keywords[gt_text] if gt_text is not None else None

#         # 获取 qwen_output 列表中的文本（假设列表中第一个元素包含完整的预测信息）
#         qwen_output = entry.get("qwen_output", [])
#         if not qwen_output:
#             continue
#         output_str = qwen_output[0]

#         # 分别提取 Image、Text 和 Combination 的预测结果（忽略大小写）
#         image_match = re.search(r'Image:\s*([A-Za-z]+)', output_str, re.IGNORECASE)
#         text_match = re.search(r'Text:\s*([A-Za-z]+)', output_str, re.IGNORECASE)
#         comb_match = re.search(r'Combination:\s*([A-Za-z]+)', output_str, re.IGNORECASE)

#         pred_image = image_match.group(1).lower() if image_match else None
#         pred_text = text_match.group(1).lower() if text_match else None
#         pred_comb = comb_match.group(1).lower() if comb_match else None

#         # 只有当三个预测值以及对应的真实标签均存在时，才进行比较
#         if (pred_image is not None and pred_text is not None and pred_comb is not None and
#             gt_image_str is not None and gt_text_str is not None and gt_comb_str is not None):
#             if pred_image == gt_image_str and pred_text == gt_text_str and pred_comb == gt_comb_str:
#                 fully_understood_indices.append(i)
#                 fully_understood_count[gt_comb_str] += 1
#                 fully_understood_entries.append(entry)

#     total_samples = len(data)
#     num_fully_understood = len(fully_understood_indices)
#     print(f"Fully understood samples: {num_fully_understood} out of {total_samples} total samples.")
#     for sentiment in sentiment_keywords:
#         print(f"Label '{sentiment}': {fully_understood_count[sentiment]} fully understood samples.")

#     # 将完全理解的条目写入新的文件，每一行为一条 JSON 数据
#     with open("fully_understood_entries.json", "w", encoding="utf-8") as out_file:
#         for entry in fully_understood_entries:
#             out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
def main():
    # qwen_generate_img_txt_comb()
    # calculate_metrics_qwen(result_file="qwen_result_infer_0307-3.json",sentiment_position="last")
    # calculate_metrics_llava(result_file="llava_result_infer_0307.json",sentiment_position="last")
    # qwen_generate(result_file="qwen_result_infer_0307-3.json")
    # find_understand()
    # llava_generate()
    # qwen_generate_retrieval_json(
    #     result_file= "retri_all_feature.json",
    #     retrieval_json_file= "retrieval_results_all_shot_1_normalize_2.json",
    #     sd_label= 1,
    # )
    calculate_metrics_qwen_retrieval(
        result_file= "retri_all_feature.json",
        sentiment_position= "first",)

if __name__ == '__main__':
    main()