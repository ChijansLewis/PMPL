import re
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from accelerate import Accelerator

class LlavaInfer:
    '''
    使用llava模型进行支持多组图文示例的多卡推理模型
        message 需要参考示例来写
        resize_size 图片裁剪尺寸，0为不裁剪
        max_new_tokens (int, optional)
    '''
    def __init__(self, 
                 message:list[dict]=[{}], 
                 resize_size=224, 
                 max_new_tokens=200):
        self.resize_size = resize_size
        self.max_new_tokens = max_new_tokens
        self.output_text = ''
        self.message = message
    def initialize(self, model_id="llava-hf/llava-1.5-7b-hf"):
        '''
        用于加载模型权重，需要在初始化后运行
        '''
        self.accelerator = Accelerator()  # Initialize Accelerator
        # Load the model and processor with the Accelerator
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map='auto',  # Automatically distribute model across available GPUs
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Prepare the model for multi-GPU (Accelerator handles this)
        self.model = self.accelerator.prepare(model)
    def update(self, **kwargs):
        """
        以**kwargs方式更新推理时的参数
        """
        # Iterate over the keyword arguments and update the attributes
        for key, value in kwargs.items():
            if hasattr(self, key):  # Check if the attribute exists
                setattr(self, key, value)

    def infer(self) -> str:
        '''
        运行initialize之后进行推理，如变更参数需要调用update方法
        '''

        imgs, prompt = self.apply_chat_template(self.message)

        # Process the image and text for model input
        inputs = self.processor(images=imgs, text=prompt, return_tensors='pt').to(self.accelerator.device, torch.float16)

        # Generate the output
        with torch.no_grad():  # Disable gradients for inference
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        # Decode the output and return as a string
        result = self.processor.decode(output[0], skip_special_tokens=True)
        prompt = re.sub('<image>', '', prompt)
        self.output_text = result[len(prompt):].lstrip('\n')
        return self.output_text
    
    def apply_chat_template(self, message):
        '''
        自定义message的转化方法，<image>\n嵌入图像
        '''
        imgs = []
        prompt = ''
        for msg in message:
            for content in msg['content']:
                if content['type'] == 'text':
                    prompt += content['text']
                    prompt += "\n"
                elif content['type'] == 'image':
                    img = Image.open(content['image'])
                    if self.resize_size != 0:
                        img = img.resize((self.resize_size, self.resize_size))
                    imgs.append(img)
                    prompt += '<image>'
                   
        return imgs, prompt


# message = [
#             {
#                 "role": "user",
#                 "content": 
#                 [
#                     {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
#                 ] 
#                 +
#                 [
#                     {"type": "text", "text": "Please determine the sentiment expressed by the following combination (positive, negative, or neutral). "}
#                 ]
#                 +
#                 [
#                     {"type": "text", "text": "Image: \n"}
#                 ]
#                 +
#                 [
#                     {"type": "image", "image": image_path}
#                 ]
#                 +
#                 [
#                     {"type": "text", "text": "Text: \n"}
#                 ]
#                 +
#                 [
#                     {"type": "text", "text": txt}
#                 ]
#                 +
#                 [
#                     {"type": "text", "text": "Your answer should be: Image:(sentiment), Text:(sentiment), Combination:(sentiment)"}
#                 ]
#             }
#         ]

if __name__ == "__main__":
    images = [
        "dataset/twitter/data_image/0001.jpg",
        "dataset/twitter/data_image/0002.jpg",
    ]
    texts = [
        "How Jake Paul is changing the influencer game:",
        "Chris Brown and his crew were kicked off a plane after allegedly hot boxing it",
    ]
    prompt = '''Below are two pictures and two paragraphs of text. Please analyze the emotions of the first set of pictures and text and the second set of pictures and text respectively.'''
    message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": texts[i]} for i in range(len(texts))
                ] + [
                    {"type": "image", "image": images[i]} for i in range(len(images))
                ] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    model = LlavaInfer(resize_size=0)
    model.initialize(model_id="models/llava-1.5-7b-hf")
    model.update(message=message)
    model.infer()
    print(model.output_text)
