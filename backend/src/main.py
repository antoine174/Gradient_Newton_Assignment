from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from data_handler import DataHandler
# from experiments import ExperimentRunner

app = FastAPI()

# TODO (Antoine): Setup CORS middleware so the React frontend can talk to this API.

@app.get("/api/run-optimization")
def run_optimization(method: str, alpha: float, a0_init: float, a1_init: float):
    # TODO (Antoine): Initialize DataHandler and load data.
    # TODO (Antoine): Initialize ExperimentRunner with the data.
    # TODO (Antoine): Route the request to the correct optimizer based on 'method' (gd or newton).
    # TODO (Antoine): Return a JSON payload containing the iterations, loss history, and weight path.
    pass