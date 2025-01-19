# AI Assistant – Emotion-Aware and Contextual Decision-Making 🤖💬 (Still in Progress)

## Project Overview
This project is an AI-powered assistant that uses fine-tuned models to improve emotional comprehension and personalized interactions. The core of the assistant is based on LLaMA 3.1 (fine-tuned as the base model) and BERT (fine-tuned as the reward model), trained using Reinforcement Learning from Human Feedback (RLHF) with DPO to enhance the assistant’s ability to understand and respond to emotions more effectively. 💡

## Current Features
* Base Model (LLaMA 3.1): The assistant utilizes a fine-tuned version of LLaMA 3.1 to provide robust natural language understanding, ensuring intelligent responses across various conversation topics. 🌐
* Reward Model (BERT): Fine-tuned BERT model to improve the assistant’s ability to identify emotions in user inputs and refine the overall dialogue quality. ❤️
* RLHF with DPO: Applied Reinforcement Learning from Human Feedback (RLHF) combined with DPO (Direct Preference Optimization) to refine emotional comprehension and personalize responses based on user interactions. 🔄
* Memory with RAG: The assistant is equipped with Retrieval-Augmented Generation (RAG), using ChromaDB as the memory backend, allowing it to recall previous conversations and provide a personalized experience. 🧠

## Future Enhancements
This project is still in progress, and several exciting features are planned for future updates:
* Image and Speech Recognition: The AI assistant will gain the ability to recognize and remember users' voices and images for a more personalized interaction. 📸🎤
* Integration with External Applications: Future versions will allow the assistant to access external services like Google Calendar, Microsoft Office, and Zoom to manage schedules, meetings, and tasks dynamically. 📅📞

## Installation ⚙️
To run this project locally, follow the instructions below:

### Prerequisites
* Python 3.10+ 🐍
* CUDA-enabled GPU (optional for improved performance) 🎮

### Setup
1. Clone the repository:

```bash
git clone https://github.com/your-username/ai-assistant.git
cd ai-assistant
```

2. Install additional dependencies (make sure to do this inside virtual environment):

```bash
pip install -r requirements.txt
```

### Running the Project
If you are running the code for the first time, make sure to run this first (only need to run once):
```bash
python -m src.__init__
```

To start the AI assistant, make sure to be in the root directory (before src), then enter:

```bash
python -m src.main
```
You will be prompted to interact with the assistant, and it will respond based on its emotional comprehension and previous interactions. 🗣️

## License 📜
This project is licensed under the APACHE 2.0 License - see the LICENSE file for details.

Contributing 🤝
A lot of appreciation for the open source model and dataset that are being used and will be used in this project, namely:
1. [Llama3.1 Unsloth](https://github.com/unslothai/unsloth?tab=readme-ov-file)
2. [Bert](https://huggingface.co/docs/transformers/en/model_doc/bert)
3. [Dair-AI Dataset](https://huggingface.co/datasets/dair-ai/emotion)
   
Feel free to fork this repository, submit issues, and create pull requests for bug fixes, features, or improvements!

