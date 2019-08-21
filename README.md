# Interactive Q&A Demo for BERT 
## Usage
`pip install -r requirements.txt`  
`export FLASK_APP=app/server.py`  
`flask run`  
Input priority is given to text-based prompts.
## Screenshots
### Home Page
<img src="https://raw.githubusercontent.com/henryzxu/interactive-transformers/develop/assets/home.png" alt="home" width="400"/>
  
### Results  
<img src="https://raw.githubusercontent.com/henryzxu/interactive-transformers/develop/assets/results.png" alt="results" width="400"/>

## Issues
- Removing files via the Dropzone UI is currently not supported by the server. For now, all session uploads are cleared upon
successful in-memory storage of the latest file-based prompt.
## Credits
Credit to [Hugging Face](https://github.com/huggingface/pytorch-transformers) for their PyTorch implementation of
BERT, which I modified to streamline inference and extraction of additional prediction data. 
