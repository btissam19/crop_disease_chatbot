
# Crop Disease Chatbot

## Project Overview and Use Cases
This project involves creating a model to recognize and classify crop diseases into 38 different categories. It can be used by anyone to predict the disease affecting their plants, provided they have some understanding of these 38 categories. Our model only predicts diseases that fall within these specified categories.

## Dataset
The dataset used for this project can be found on Kaggle: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## How to Execute This Project on Your Machine

### Requirements
- Python version 3.x
- Anaconda Prompt (recommended for managing versions and dependencies)

### Steps
1. **Create Your Project Environment**: Open your Anaconda Prompt and run:
    ```bash
    conda create -n envname
    ```

2. **Activate Your Environment**: In the same shell, run:
    ```bash
    conda activate envname
    ```

3. **Navigate to the Project Directory**: In the same shell, go to the directory where you downloaded the project files and open your text editor.

4. **Choose the Python Interpreter**: Set the Python interpreter to the environment `envname`.

5. **Download the Required Packages**: Run the following command to install the necessary packages:
    ```bash
    pip install streamlit tensorflow transformers
    ```

6. **Run the Application**: After installation, run your app using:
    ```bash
    streamlit run main.py
    ```

## Additional Information
To ensure smooth execution, consider the following:
- Make sure your text editor (e.g., VSCode, PyCharm) is configured to use the Python interpreter from the `envname` environment.
- If you encounter any issues with package installations, check your internet connection and try running the command again.
- For detailed documentation on Streamlit, TensorFlow, and Transformers, refer to their official documentation:
  - [Streamlit Documentation](https://docs.streamlit.io/)
  - [TensorFlow Documentation](https://www.tensorflow.org/learn)
  - [Transformers Documentation](https://huggingface.co/transformers/)

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.
