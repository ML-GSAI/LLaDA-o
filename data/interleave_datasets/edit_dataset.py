# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import re
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset, WebdatasetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        if "image_list" in row and "instruction_list" in row:
            image_list = row["image_list"]
            instruction_list = row["instruction_list"]
        elif "src_img" in row and "edited_img" in row and "edited_prompt_list" in row:
            image_list = [row["src_img"]["bytes"], row["edited_img"]["bytes"]]
            instruction_list = [row["edited_prompt_list"]]
        else:
            raise ValueError("Unknown data format in row")

        image_num = len(image_list)
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(image_list[start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(instruction_list[idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(image_list[end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(instruction_list[idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(image_list[idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(image_list[idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data

class UnifiedEditWebdatasetIterableDataset(InterleavedBaseIterableDataset, WebdatasetStandardIterableDataset):
    def count_input_images(self, sample):
        """Count the number of input images by looking for keys in image_x.jpg format"""
        max_image_num = 0
        for key in sample.keys():
            if key.startswith("image_") and key.endswith(".jpg"):
                try:
                    # Extract the numeric part, e.g., extract "1" from "image_1.jpg"
                    num_str = key.replace("image_", "").replace(".jpg", "")
                    num = int(num_str)
                    max_image_num = max(max_image_num, num)
                except ValueError:
                    # Skip this key if it cannot be converted to a number
                    continue
        return max_image_num
    
    def parse_sample(self, sample):
        if ("output.jpg" in sample and "txt" in sample and any(key.startswith("image_") and key.endswith(".jpg") for key in sample.keys())):
            input_image_count = self.count_input_images(sample)
            
            if input_image_count > 0:
                txt_content = sample["txt"]
                if re.search(r'<img><\|image_\d+\|></img>', txt_content):
                    data = self._init_data()
                    
                    # Use regex to split text, preserving delimiters
                    pattern = r'(<img><\|image_(\d+)\|></img>)'
                    parts = re.split(pattern, txt_content)
                    
                    for i in range(0, len(parts), 3):  # Every 3 elements form a group: text, full tag, image number
                        if i < len(parts):
                            # Add text part (if not empty)
                            text_part = parts[i].strip()
                            if text_part:
                                data = self._add_text(data, text_part, need_loss=False)
                            
                            # Check if there's an image tag
                            if i + 2 < len(parts):
                                image_num = parts[i + 2]  # image number
                                image_key = f"image_{image_num}.jpg"
                                if image_key in sample:
                                    # Add corresponding image
                                    data = self._add_image(
                                        data, 
                                        sample[image_key],
                                        need_loss=True, 
                                        need_vae=True, 
                                        need_vit=True,
                                    )
                    
                    # Add output image
                    data = self._add_image(
                        data,
                        sample["output.jpg"],
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
                    return data
                    
                else:
                    input_image_list = []
                    for i in range(1, input_image_count + 1):
                        image_key = f"image_{i}.jpg"
                        if image_key in sample:
                            input_image_list.append(sample[image_key])
                    
                    data = self._init_data()
                    # Add input images
                    for input_img in input_image_list:
                        data = self._add_image(
                            data, 
                            input_img,
                            need_loss=False, 
                            need_vae=True, 
                            need_vit=True,
                        )
                    
                    # Add text
                    data = self._add_text(data, sample["txt"].replace("<image>", "").rstrip(), need_loss=False)
                    
                    # Add output image
                    data = self._add_image(
                        data,
                        sample["output.jpg"],
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
                    return data

            else:
                raise ValueError("No valid input images found in multi-image format")
        else:
            if "src_img.jpg" in sample and "edited_img.jpg" in sample and "edited_prompt.txt" in sample:
                image_list = [sample["src_img.jpg"], sample["edited_img.jpg"]]
                # Process text, remove <image> tags
                prompt_text = sample["edited_prompt.txt"].replace("<image>", "")
                instruction_list = [[prompt_text]]
            elif "1.0.jpg" in sample and "2.jpg" in sample and "json" in sample and "1.1.jpg" not in sample:
                image_list = [sample["1.0.jpg"], sample["2.jpg"]]
                # Process text, remove <image> tags
                prompt_text = sample["json"]['instruction'].replace("<image>", "")
                instruction_list = [[prompt_text]]
            elif "1.0.jpg" in sample and "2.jpg" in sample and "json" in sample and "1.1.jpg" in sample:
                # Return empty dict to indicate this data cannot be parsed
                return {}
            else:
                raise ValueError("Unknown data format in sample")

            image_num = len(image_list)
            # randomly choose start and end, return [0, 1] when only two images
            start_idx = random.choice(range(image_num - 1))
            max_end = min(start_idx + 3, image_num)
            end_idx = random.choice(range(start_idx + 1, max_end))

            data = self._init_data()
            data = self._add_image(
                data, 
                image_list[start_idx],
                need_loss=False, 
                need_vae=True, 
                need_vit=True, 
            )

            if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
                if end_idx == image_num - 1:
                    end_idx -= 1

                instruction = ""
                for idx in range(start_idx + 1, end_idx + 1):
                    instruction += random.choice(instruction_list[idx-1]) + ". "
                data = self._add_text(data, instruction.rstrip(), need_loss=False)
                data = self._add_image(
                    data, 
                    image_list[end_idx],
                    need_loss=True, 
                    need_vae=False, 
                    need_vit=False,
                )
            else:
                for idx in range(start_idx + 1, end_idx + 1):
                    instruction = random.choice(instruction_list[idx-1])
                    data = self._add_text(data, instruction, need_loss=False)
                    if idx != end_idx:
                        data = self._add_image(
                            data, 
                            image_list[idx],
                            need_loss=True, 
                            need_vae=True, 
                            need_vit=True,
                        )
                    else:
                        data = self._add_image(
                            data, 
                            image_list[idx],
                            need_loss=True, 
                            need_vae=False, 
                            need_vit=False,
                        )
            return data
