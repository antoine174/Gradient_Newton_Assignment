from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.data_handler import DataHandler
from src.optimizers import Optimizers

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data once when the server starts
handler = DataHandler(file_path="data/AmesHousing.csv")
x, y = handler.load_and_preprocess()

@app.get("/api/run-optimization")
def run_optimization(method: str, alpha: float, a0_init: float, a1_init: float, max_iter: int = 1000):
    opts = Optimizers(x, y)
    
    if method == "gd":
        result = opts.gradient_descent(alpha=alpha, a0_init=a0_init, a1_init=a1_init, max_iter=max_iter)
    elif method == "newton":
        result = opts.newtons_method(alpha=alpha, a0_init=a0_init, a1_init=a1_init, max_iter=max_iter)
    else:
        return {"error": "Invalid method selected."}
    
    return result