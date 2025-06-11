## Running the Project
 
 python -m venv venv

.\venv\Scripts\activate


1. **Navigate to the repository directory**
 
   ```bash
   cd PATH_TO_CLONED_REPO
 
2. **Activate the virtual environment**
   ```bash
   .\venv\Scripts\activate
3. **Install dependencies**
    ```bash
    pip install -r .\requirements.txt
4. **Run the uvicorn server**
    ```bash
   uvicorn app.main:app --reload  
5. **Access the application**
    ```bash
   Open your browser and navigate to http://127.0.0.1:8000/docs to view the swagger documentation
