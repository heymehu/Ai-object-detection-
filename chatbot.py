import os
import json
import requests
import tkinter as tk
from tkinter import ttk, scrolledtext
from dotenv import load_dotenv
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load environment variables
load_dotenv()

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Assistant Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Initialize API settings
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENROUTER_API_KEY in your .env file")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/yourusername/chatbot",
            "X-Title": "Python Chatbot"
        }
        self.conversation_history = []

        # Initialize OpenCV
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = False
        self.detected_objects = set()

        # Load DETR model and processor
        try:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            print("DETR model loaded successfully!")
        except Exception as e:
            print(f"Error loading DETR model: {str(e)}")
            raise

        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left frame for video
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Create video label
        self.video_label = ttk.Label(self.left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Create camera control buttons
        self.camera_frame = ttk.Frame(self.left_frame)
        self.camera_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_button = ttk.Button(
            self.camera_frame,
            text="Start Camera",
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Create right frame for chat
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.right_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=("Arial", 10),
            bg="#ffffff",
            fg="#000000"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)

        # Create input frame
        self.input_frame = ttk.Frame(self.right_frame)
        self.input_frame.pack(fill=tk.X)

        # Create message input
        self.message_input = ttk.Entry(
            self.input_frame,
            font=("Arial", 10)
        )
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_input.bind("<Return>", self.send_message)

        # Create send button
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Welcome message
        self.display_message("Assistant", "Welcome to Vision Assistant Pro! I can detect various objects and animals. Click 'Start Camera' to begin.")

    def toggle_camera(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.start_button.config(text="Stop Camera")
            self.update_camera()
        else:
            self.is_capturing = False
            self.start_button.config(text="Start Camera")
            self.video_label.config(image='')

    def update_camera(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Process image with DETR
                inputs = self.processor(images=pil_image, return_tensors="pt")
                outputs = self.model(**inputs)
                
                # Get predictions
                target_sizes = torch.tensor([pil_image.size[::-1]])
                results = self.processor.post_process_object_detection(
                    outputs, 
                    target_sizes=target_sizes, 
                    threshold=0.5
                )[0]

                # Get current objects
                current_objects = set()
                
                # Draw boxes and labels
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [int(i) for i in box.tolist()]
                    
                    # Get class name
                    class_name = self.model.config.id2label[label.item()]
                    confidence = score.item()
                    
                    # Add to current objects
                    current_objects.add(class_name)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (box[0], box[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for new objects
                new_objects = current_objects - self.detected_objects
                if new_objects:
                    for obj in new_objects:
                        self.display_message("Assistant", f"I detected a {obj} in the frame!")
                    self.detected_objects = current_objects

                # Convert frame to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_camera)

    def display_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_display.insert(tk.END, f"{sender}: ", "user")
        else:
            self.chat_display.insert(tk.END, f"{sender}: ", "assistant")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def get_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        
        data = {
            "model": "deepseek/deepseek-r1-distill-qwen-7b",
            "messages": self.conversation_history
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            assistant_response = response.json()["choices"][0]["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def send_message(self, event=None):
        user_input = self.message_input.get().strip()
        if not user_input:
            return

        # Clear input field
        self.message_input.delete(0, tk.END)

        # Display user message
        self.display_message("You", user_input)

        # Disable input while processing
        self.message_input.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

        # Process response in a separate thread
        def process_response():
            response = self.get_response(user_input)
            self.root.after(0, lambda: self.display_message("Assistant", response))
            self.root.after(0, lambda: self.message_input.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.message_input.focus())

        threading.Thread(target=process_response, daemon=True).start()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ChatbotGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
