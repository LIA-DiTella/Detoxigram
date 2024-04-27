from PIL import Image, ImageDraw, ImageFont
import os

class GroupToxicityDistribution:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.gauge_images = {
            0: './empty_gauge.png',
            1: './gauge_1.png',
            2: './gauge_2.png',
            3: './gauge_3.png',
        }
        self.positions = [
            (926, 175, 1322, 371),  # Sarcastic
            (1443, 173, 1845, 369), # Antagonize
            (924, 617, 1322, 816),  # Generalization
            (1445, 615, 1843, 813), # Dismissive
        ]
        self.font_path = os.path.join(self.base_dir, 'IBMPlexSans-Regular.ttf')
        self.template_path = os.path.join(self.base_dir, 'template.png')

    def get_toxicity_graph(self, channel_name, toxicity_vector):
        try:
            with Image.open(self.template_path) as img:
                draw = ImageDraw.Draw(img)
                
                font_size = 36 * 4 
                font = ImageFont.truetype(self.font_path, font_size)
                
                name_coords = (97, 269)
                draw.text(name_coords, channel_name, font=font, fill='#DC312C')          
        
                for value, (x1, y1, x2, y2) in zip(toxicity_vector, self.positions):
                    scaled_value = 0 if value < 0.10 else 1 if value < 0.5 else 2 if value < 0.75 else 3
                    gauge_image_path = os.path.join(self.base_dir, self.gauge_images[scaled_value])
                    
                    with Image.open(gauge_image_path) as gauge_img:
                        gauge_width, gauge_height = gauge_img.size
                        gauge_x = x1 + (x2 - x1 - gauge_width) // 2
                        gauge_y = y1 + (y2 - y1 - gauge_height) // 2
                        img.paste(gauge_img, (gauge_x, gauge_y), gauge_img)
                
                output_path = os.path.join(self.base_dir, f'{channel_name}_toxicity_distribution.png')
                img.save(output_path)
                return output_path
        except IOError as e:
            print(f"Error opening image files: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while creating the toxicity distribution: {e}")
