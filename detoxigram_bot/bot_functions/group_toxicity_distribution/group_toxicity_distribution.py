import os 
from PIL import Image, ImageDraw, ImageFont

class GroupToxicityDistribution:
    def __init__(self, base_dir):
        self.positions = [
                            (438, 828, 841),
                            (1159, 828, 1558), 
                            (442, 1189, 841), 
                            (1161, 1187, 1558),
                            (440, 1548, 839),
                            (1163, 1547, 1558), 
                        ]
        self.font_path = os.path.join(os.path.dirname(__file__), 'IBMPlexSans-Regular.ttf')
        self.template_path = os.path.join(os.path.dirname(__file__), 'template.png')
        self.base_dir = base_dir


    def get_toxicity_graph(self, channel_name, toxicity_vector):
        '''
        Pre: None
        Post: saves the toxicity distribution of the channel in a png file
        '''
        with Image.open(self.template_path) as img:
            draw = ImageDraw.Draw(img)
            font_size = 36 * 4  
            font = ImageFont.truetype(self.font_path, font_size)
            text_width = draw.textlength(channel_name, font=font)
            x = (img.width - text_width) // 2
            y = 120
            
            draw.text((x, y), channel_name, font=font, fill='#DC312C')

            for value, (x_start, y, x_end) in zip(toxicity_vector, self.positions):
                x_position = x_start + float(value) * (x_end - x_start)
                radius = 25
                draw.ellipse((x_position - radius, y - radius, x_position + radius, y + radius), fill='#F8F1E4', outline='#DC312C', width=3)
            
            img.save(f'{channel_name}_toxicity_distribution.png')
            return os.path.join(self.base_dir, f'{channel_name}_toxicity_distribution.png')
        