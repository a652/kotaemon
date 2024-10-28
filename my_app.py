import os

from theflow.settings import settings as flowsettings

KH_APP_DATA_DIR = getattr(flowsettings, "KH_APP_DATA_DIR", ".")
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", None)
# override GRADIO_TEMP_DIR if it's not set
if GRADIO_TEMP_DIR is None:
    GRADIO_TEMP_DIR = os.path.join(KH_APP_DATA_DIR, "gradio_tmp")
    os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR


# from ktem.main import App  # noqa

# app = App()
# demo = app.make()
# demo.queue().launch(
#     favicon_path=app._favicon,
#     inbrowser=True,
#     allowed_paths=[
#         "libs/ktem/ktem/assets",
#         GRADIO_TEMP_DIR,
#     ],
# )

# app.py
from fastapi import FastAPI, Request  # Import FastAPI and Request class
from fastapi.responses import JSONResponse  # Import JSONResponse for API responses
import gradio as gr  # Import Gradio for UI integration
import uvicorn  # Import Uvicorn to run the FastAPI application
from ktem.main import App  # Import Gradio application from ktem.main

# Utility function to find the index of an object in a list by matching a key-value pair
def index_of_obj(objects, key, value):
    for index in objects:
        if getattr(objects[index], key) == value:  # Check if the object's key matches the value
            return index
    return -1  # Return -1 if no matching object is found

# Initialize key-index mapping for Gradio functions
def init_ktem_constants(demo):
    func_names = ["chat_fn", "list_file"]  # List of function names to be mapped
    func_indices = {}  # Dictionary to store function indices

    # Map each function name to its index in the Gradio app
    for func_name in func_names:
        func_indices[func_name] = index_of_obj(demo.fns, "name", func_name)
        print("func_name:", func_name, "func_indices:", func_indices[func_name])

    return func_indices  # Return the function index map

# Initialize and extend the API with custom and Gradio routes
def init_extend_api(demo):
    extendapi = FastAPI()  # Create a new FastAPI instance
    ktem_constants = init_ktem_constants(demo)  # Initialize function index map

    # Custom API route for testing
    @extendapi.get("/extendapi/test")
    async def get_test():
        return JSONResponse(content={"status": True, "message": "Hello from FastAPI!"})

    # Gradio API route to get a list of files
    @extendapi.get("/extendapi/file")
    async def get_extendapi_file(request: Request):
        # TODO: Replace with actual user_id loading logic
        user_id = 1
        list_file_func_index = ktem_constants["list_file"]  # Get the index for 'list_file' function
        file_list = demo.fns[list_file_func_index].fn(user_id)  # Call 'list_file' function with user_id
        return {"status": True, "message": "Success", "file_list": file_list[0]}

    @extendapi.post("/extendapi/chat")
    async def post_extendapi_chat(request: Request):
        # TODO: Replace with actual user_id loading logic
        user_id = 1
        chat_fn_func_index = ktem_constants["chat_fn"]
        res = next(demo.fns[chat_fn_func_index].fn(None,[[request.json["message"]]],demo._app.settings_state,demo._reasoning_type,demo._llm_type,demo.state_chat,demo._app.userid))
        result = []
        result.append(res)
        while True:
            try:
                result.append(next(res))
            except StopIteration:
                break
        return {"status": True, "message": "Success", "response": result}
    
    return extendapi  # Return the FastAPI instance with extended APIs

# Create an instance of the Gradio application
gradio_app = App().make()  # Create the Gradio app from the custom App class
extendapi = init_extend_api(gradio_app)  # Initialize the extended API

# Mount Gradio interface into FastAPI under the root path "/"
gr.mount_gradio_app(
    extendapi,
    gradio_app,
    path="/",  # Set the path for Gradio app
)

# Run FastAPI application with Gradio interface
if __name__ == "__main__":
    uvicorn.run(extendapi, host="0.0.0.0", port=7860)  # Launch the app on port 7860
