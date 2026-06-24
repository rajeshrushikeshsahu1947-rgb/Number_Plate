# 🔢 Number Plate Detection (YOLOv8)

## 👋 Introduction
This project is about **detecting vehicle number plates** using computer vision and deep learning.  
The idea is simple: feed in an image or video, and the system highlights the license plate region for further processing (like OCR or tracking).  

I built this to explore how AI can be applied to **traffic monitoring, smart parking, and law enforcement systems**.

---

## 🛠️ Tech Stack
- 🐍 **Python**  
- 🎥 **OpenCV** for image processing  
- 🧠 **YOLOv8** for object detection  
- 📓 **Jupyter Notebook** for experiments  
- ⚡ Pre‑trained weights (`yolov8_custom.pt`, `best.pt`)  

---

## ✨ Features
- 🚗 Detects number plates in images and video streams  
- 📸 Works with custom datasets (e.g., taxi images, Indian vehicles)  
- 🔧 Easy to retrain with your own dataset  
- 📊 Supports exporting detection results for analysis  

---

## 📂 Repository Structure
Number_Plate/
│── README.md          # Documentation
│── main.py            # Main detection script
│── args.yaml          # Configurations
│── best.pt            # Trained YOLOv8 model
│── yolov8_custom.pt   # Custom weights
│── taxi1.jpg          # Sample test image

Code

---

## 🚀 Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/rajeshsahu777/Number_Plate.git
   cd Number_Plate
Install dependencies:

bash
pip install -r requirements.txt
Run detection:

bash
python main.py --source taxi1.jpg --weights best.pt
📸 Example Output
✅ Bounding box around detected number plate

✅ Confidence score for detection

✅ Ready for OCR integration

🤝 Contribution
This is an open project — feel free to fork, experiment, and improve.
Ideas like OCR integration, dataset expansion, or real‑time video support are welcome!

📧 Contact
Rajesh Sahu  
📍 Pune, Maharashtra
✉️ rajeshrushikeshsahu1947@gmail.com
🔗 GitHub: rajeshsahu777 (github.com in Bing)  
🔗 LinkedIn: Rajesh Sahu (linkedin.com in Bing)

⚡ Fun Fact
This started with a simple photo of a taxi 🚕 — now it’s a step toward smarter traffic systems!
