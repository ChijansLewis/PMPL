import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import torch
from transformers import AutoModelForCausalLM

from .janus.models import MultiModalityCausalLM, VLChatProcessor
from .janus.utils.io import load_pil_images

class JanusInfer:
    """
    使用 Janus 模型进行多模态推理的类，支持图文对话示例。
        conversation: 对话示例列表，示例格式如下：
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nDescribe this image.",
                    "images": ["path/to/image.jpg"],
                },
                {"role": "Assistant", "content": ""}
            ]
        max_new_tokens: 生成文本的最大 token 数量。
    """
    def __init__(self, conversation: list[dict], max_new_tokens: int = 512):
        self.conversation = conversation
        self.max_new_tokens = max_new_tokens
        self.output_text = ''

    def update(self, **kwargs):
        """
        通过关键字参数更新推理时的参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @torch.inference_mode()
    def initialize(self, model_path: str):
        """
        加载 Janus 模型和 VLChatProcessor 权重，准备推理
        """
        # 加载处理器及其对应的 tokenizer
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        # 加载模型，并转为 bfloat16 模式，设置评估模式
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map='auto',
        )
        self.model = self.model.to(torch.bfloat16).eval()

    @torch.inference_mode()
    def infer(self):
        """
        运行推理，生成回答文本
        """
        # 加载图片，并准备对话输入
        pil_images = load_pil_images(self.conversation)
        prepare_inputs = self.processor(
            conversations=self.conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self.model.device)

        # 获取输入嵌入
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # 使用生成器生成回答，参数参考示例代码
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        # 解码输出为文本
        self.output_text = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return self.output_text

if __name__ == "__main__":
    # 构造示例对话，包含图片和文本提示
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nDescribe this image.",
            "images": ["dataset/twitter/data_image/0001.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]
    # 初始化 JanusInfer 类
    janus_infer = JanusInfer(conversation=conversation, max_new_tokens=512)
    janus_infer.initialize(model_path="/home/STU/ljq/LLMs/DeepSeek/Janus-Pro-7B")
    
    # 运行推理，并打印生成的回答
    answer = janus_infer.infer()
    print(answer)
