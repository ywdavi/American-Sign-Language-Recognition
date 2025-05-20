# Real-Time American Sign Language Recognition

430 million people, over 5% of the world's population, had disabling hearing loss in 2023. The World Health Organization estimates that by 2050, nearly 2.5 billion people will have some degree of hearing loss. People who are deaf or hard of hearing often face social exclusion due to communication challenges. To address this issue, we developed a web app that recognizes and translates ASL gestures in real-time using image classification techniques. Our app supports three different languages, suggests words through a language model, and converts recognized text into speech. This tool aims to help ASL users communicate more easily, promoting greater accessibility and social integration.

## Demo

https://github.com/user-attachments/assets/d3531e78-f5a3-4c20-9653-4d599bdf3d61

## ASL_YOLO_training.ipynb 
This notebook is used to load and preprocess the original _ASL Alphabet dataset_ and train a YOLO model for classification. The notebook was run on the Kaggle framework to use the GPU accelerator.

## SIGN4ALL-dashboard
The SIGN4ALL-dashboard folder includes all the necessary files to run the SIGN4ALL dashboard, a web-based application designed to facilitate sign language recognition and translation.

To successfully run the dashboard, ensure the following folder structure is maintained:

<pre>
SIGN4ALL-dashboard/  
│   main.py  
│   requirements.txt  
│  
├── static/  
│   ├── images/  
│   │   ├── no_camera.png  
│   │   ├── sign4all.png  
│   │   ├── favicon.ico  
│   │   └── all_signs.png  
│   ├── css/  
│   │   └── styles.css  
│   ├── corpora/  
│   │   ├── spanish_words.txt  
│   │   └── 280000_parole_italiane.txt  
│   └── models/  
│       ├── YOLO.pt  
│       ├── kenLM_eng.binary  (KenLM models are missing in this repository)
│       ├── kenLM_ita.bin  
│       └── kenLM_es.binary  
│  
└── templates/  
    └── index.html  
</pre>

How to run the Dashboard
1.	Download the SIGN4ALL-dashboard folder
2.	Install the required packages using:
pip install -r requirements.txt
3.	Run the dashboard using the command:
python main.py

## Acknowledgments
This project was developed with the help and collaboration of **Simone Vaccari**.
