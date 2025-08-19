# AI Presentation Generator

This project has been refactored into a separate frontend and backend.

## Project Structure

-   `backend/`: Contains the FastAPI application, all the core logic for presentation generation, and dependencies.
-   `frontend/`: Contains the HTML, CSS, and JavaScript files for the user interface.

## How to Run

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Install the dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend server will be running at `http://localhost:8000`.

### Frontend

1.  **Open the `index.html` file in your browser:**
    You can do this by simply double-clicking the `frontend/index.html` file, or by using a simple HTTP server for local development. For example, using Python's built-in server:
    ```bash
    cd frontend
    python -m http.server
    ```
    Then, navigate to `http://localhost:8000` in your browser (or a different port if 8000 is taken by the backend). Note that if you run the frontend from a different port than the backend, you might encounter CORS issues. The provided `script.js` assumes the backend is at `http://localhost:8000`.

## How to Use

1.  Make sure both the backend and frontend are running.
2.  Open the frontend in your browser.
3.  Enter a topic in the input field.
4.  Click the "Generate Presentation" button.
5.  The presentation will be generated and displayed on the screen. You can navigate through the slides using the previous and next buttons.
