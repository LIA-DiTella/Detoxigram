from PIL import Image, ImageDraw, ImageFont
import os

class group_toxicity_distribution:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.gauge_images = {
            'ok_sarcastic': 'ok_sarcastic.png',
            'moderately_sarcastic': 'moderately_sarcastic.png',
            'highly_sarcastic': 'highly_sarcastic.png',
            'ok_antagonizing': 'ok_antagonizing.png',
            'moderately_antagonizing': 'moderately_antagonizing.png',
            'highly_antagonizing': 'highly_antagonizing.png',
            'ok_stereotyping': 'ok_stereotyping.png',
            'moderately_stereotyping': 'moderately_stereotyping.png',
            'highly_stereotyping': 'highly_stereotyping.png',
            'ok_dismissive': 'ok_dismissive.png',
            'moderately_dismissive': 'moderately_dismissive.png',
            'highly_dismissive': 'highly_dismissive.png',
        }
        
        self.positions = [
            (527, 291),  # Sarcastic
            (1008, 419), # Antagonize
            (527, 419),  # Stereotyping
            (1008, 291), # Dismissive
        ]
        self.font_path = os.path.join(self.base_dir, 'IBMPlexSans-Regular.ttf')
        self.template_path = os.path.join(self.base_dir, 'template.png')
        self.robot_images = {
            'green': 'robot_green.png',
            'yellow': 'robot_yellow.png',
            'orange': 'robot_orange.png',
            'red': 'robot_red.png'
        }

    def get_toxicity_image(self, toxicity_level, dimension):
        if toxicity_level < 0.25:
            return self.gauge_images[f'ok_{dimension}']
        elif toxicity_level < 0.50:
            return self.gauge_images[f'moderately_{dimension}']
        elif toxicity_level < 0.75:
            return self.gauge_images[f'moderately_{dimension}']
        else:
            return self.gauge_images[f'highly_{dimension}']

    def get_robot_image(self, toxicity_vector):
        ok_count = sum(1 for level in toxicity_vector if level < 0.25)
        moderately_count = sum(1 for level in toxicity_vector if 0.25 <= level < 0.75)
        highly_count = sum(1 for level in toxicity_vector if level >= 0.75)
        
        if ok_count == 4 or (ok_count == 3 and moderately_count == 1) or (ok_count == 2 and moderately_count == 2):
            return self.robot_images['green']
        elif (ok_count == 3 and highly_count == 1) or (ok_count == 2 and moderately_count == 1 and highly_count == 1) or (moderately_count == 3 and ok_count == 1):
            return self.robot_images['yellow']
        elif (moderately_count == 4) or (moderately_count == 3 and highly_count == 1) or (ok_count == 2 and highly_count == 2) or (ok_count == 1 and moderately_count == 2 and highly_count == 1):
            return self.robot_images['orange']
        else:
            return self.robot_images['red']

    def get_toxicity_graph(self, channel_name, toxicity_vector):
        try:
            with Image.open(self.template_path) as img:
                draw = ImageDraw.Draw(img)

                # Add channel name
                font = ImageFont.truetype(self.font_path, 50)
                draw.text(((820,95,1455,151)), channel_name, font=font, fill="black")

                # Add toxicity levels
                dimensions = ['sarcastic', 'antagonizing', 'stereotyping', 'dismissive']
                for i, (x, y) in enumerate(self.positions):
                    toxicity_level = toxicity_vector[i]
                    dimension = dimensions[i]
                    toxicity_image_path = self.get_toxicity_image(toxicity_level, dimension)
                    toxicity_image = Image.open(os.path.join(self.base_dir, toxicity_image_path))
                    img.paste(toxicity_image, (x, y), toxicity_image)

                # Add robot image
                robot_image_path = self.get_robot_image(toxicity_vector)
                robot_image = Image.open(os.path.join(self.base_dir, robot_image_path))
                img.paste(robot_image, (65, 100), robot_image)

                # Save the result
                output_path = os.path.join(self.base_dir, 'output.png')
                img.save(output_path)
                print(f"Generated image saved to {output_path}")
                return output_path

        except Exception as e:
            print(f"Error generating image: {e}")

