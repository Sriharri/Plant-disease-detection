from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.dialog import MDDialog
from kivymd.uix.scrollview import MDScrollView
from PIL import Image as PilImage
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import numpy as np
import tensorflow as tf
from labels_dict import labels_dict
from kivy.graphics import Color, Rectangle, Ellipse, Line 

KV = '''
MDBoxLayout:
    orientation: 'vertical'
    spacing: dp(10)
    padding: dp(10)

    Image:
        id: image_label
        source: ''
        size_hint_y: None
        height: dp(400)
        allow_stretch: True

    MDRaisedButton:
        text: "Browse Image"
        font_size: dp(20)
        theme_text_color: "Custom"
        text_color: 1, 1, 1, 1 
        on_press: app.browse_image()
        pos_hint: {'center_x': 0.5}  
    MDLabel:
        id: prediction_result_label
        text: "Prediction:"
        font_size: dp(25)

    MDRaisedButton:
        text: "Predict"
        font_size: dp(20)
        theme_text_color: "Custom"
        text_color: 1, 1, 1, 1  
        on_press: app.predict()
        pos_hint: {'center_x': 0.5}  

    MDRaisedButton:
        text: "Solve"
        font_size: dp(20)
        theme_text_color: "Custom"
        text_color: 1, 1, 1, 1  
        on_press: app.solve()
        pos_hint: {'center_x': 0.5}  
'''



class PlantDiseaseClassifierApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.confirm_image_selection,
        )
        self.confirm_dialog = None  
        self.interpreter = self.load_tflite_model()
        self.input_shape = (224, 224)
        self.output_details = self.interpreter.get_output_details()

    def build(self):
        self.image_path = None
        self.class_labels_dict = labels_dict  

        return Builder.load_string(KV)

    def load_tflite_model(self):
        model_path = 'Plant_disease_model.tflite' 
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def browse_image(self):
        self.file_manager.show('/')

    def exit_manager(self, *args):
        self.file_manager.close()

    def confirm_image_selection(self, path):
        self.image_path = path
        self.confirm_dialog = MDDialog(
            title="Confirm",
            text="Select this image?",
            size_hint=(0.8, 0.4),
            buttons=[
                MDRaisedButton(text="No", on_press=self.cancel_selection),
                MDRaisedButton(text="Yes", on_press=self.select_image),
            ],
        )
        self.confirm_dialog.open()

    def cancel_selection(self, *args):
        self.confirm_dialog.dismiss()  

    def select_image(self, *args):
        self.confirm_dialog.dismiss()  
        self.exit_manager()
        self.load_image(self.image_path)

    def load_image(self, path):
        image = PilImage.open(path)
        image = image.resize(self.input_shape, PilImage.Resampling.LANCZOS)
        texture = Texture.create(size=(image.width, image.height), colorfmt='rgb')
        texture.blit_buffer(image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.root.ids.image_label.texture = texture

    def preprocess_image(self):
        image = PilImage.open(self.image_path)
        image = image.resize(self.input_shape, PilImage.Resampling.LANCZOS)
        image_array = np.asarray(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0)
        return image_array.astype(np.float32)

    def predict(self):
        if self.image_path is not None:
            image_array = self.preprocess_image()
            input_details = self.interpreter.get_input_details()

            self.interpreter.set_tensor(input_details[0]['index'], image_array)
            self.interpreter.invoke()
            predicted_probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            predicted_class_index = np.argmax(predicted_probabilities)
            predicted_class_label = self.class_labels_dict.get(predicted_class_index, "Unknown Class")
            self.root.ids.prediction_result_label.text = f"Predicted Class: {predicted_class_label}"
        else:
            self.root.ids.prediction_result_label.text = "No image selected."

    

    def solve(self):
        if self.image_path is not None:
            predicted_class_index = np.argmax(self.interpreter.get_tensor(self.output_details[0]['index'])[0])
            predicted_class_label = self.class_labels_dict.get(predicted_class_index, "Unknown Class")

            solution_file_path = 'solutions.txt'
            solution = self.get_solution_for_class(solution_file_path, predicted_class_label)
            self.show_solution_popup(solution)
        else:
            self.show_solution_popup("No image selected.")

    def get_solution_for_class(self, file_path, predicted_class_label):
        try:
            with open(file_path, "r") as file:
                for line in file:
                    class_name, class_solution = line.strip().split('-', 1)
                    if class_name.strip() == predicted_class_label:
                        return class_solution.strip()
            return "No solution found for this class."
        except FileNotFoundError:
            return "No solution file found."

    def show_solution_popup(self, solution):
        dialog = MDDialog(
            title="Solution",
            text=solution,
            size_hint=(0.8, 0.8),
            buttons=[{"text": "Close", "on_release": lambda *args: self.dialog.dismiss()}],
        )
        dialog.open()
        self.dialog = dialog

if __name__ == '__main__':
    PlantDiseaseClassifierApp().run()
