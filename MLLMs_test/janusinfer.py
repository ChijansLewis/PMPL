from Janusfiles import JanusInfer

if __name__ == "__main__":
    # 构造示例对话，包含图片和文本提示
    conversation = [
        {
            "role": "User",
            "content": ["<image_placeholder>\n"
                        "<image_placeholder>\n"
                        "Describe these images."],
            "images": ["dataset/twitter/data_image/0001.jpg",
                       "dataset/twitter/data_image/0002.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]
    # 初始化 JanusInfer 类
    janus_infer = JanusInfer(conversation=conversation, max_new_tokens=512)
    janus_infer.initialize(model_path="/home/STU/ljq/LLMs/DeepSeek/Janus-Pro-7B")
    
    # 运行推理，并打印生成的回答
    answer = janus_infer.infer()
    print(answer)
