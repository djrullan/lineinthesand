import onnx
from onnx import helper
from onnx import TensorProto # Import TensorProto for data type conversion

def inspect_onnx_model(model_path):
    """
    Loads an ONNX model and prints its input and output details.

    Args:
        model_path (str): The file path to the ONNX model.
    """
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        print(f"Successfully loaded ONNX model from: {model_path}\n")

        # --- Inspect Inputs ---
        print("--- Model Inputs ---")
        if model.graph.input:
            for i, model_input in enumerate(model.graph.input):
                print(f"Input {i+1}:")
                print(f"  Name: {model_input.name}")

                # Get the shape and data type
                input_type = model_input.type.tensor_type
                if input_type.HasField("elem_type"):
                    # Correctly map ONNX data type enum to a readable string
                    data_type = TensorProto.DataType.Name(input_type.elem_type)
                    print(f"  Data Type: {data_type}")
                else:
                    print("  Data Type: Unknown")

                # Extract shape information
                shape = []
                if input_type.HasField("shape"):
                    for dim in input_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param) # Symbolic dimension
                        else:
                            shape.append("?") # Unknown dimension
                print(f"  Shape: {shape}")
                print("-" * 20)
        else:
            print("No inputs found for this model.")
        print("\n")

        # --- Inspect Outputs ---
        print("--- Model Outputs ---")
        if model.graph.output:
            for i, model_output in enumerate(model.graph.output):
                print(f"Output {i+1}:")
                print(f"  Name: {model_output.name}")

                # Get the shape and data type
                output_type = model_output.type.tensor_type
                if output_type.HasField("elem_type"):
                    # Correctly map ONNX data type enum to a readable string
                    data_type = TensorProto.DataType.Name(output_type.elem_type)
                    print(f"  Data Type: {data_type}")
                else:
                    print("  Data Type: Unknown")

                # Extract shape information
                shape = []
                if output_type.HasField("shape"):
                    for dim in output_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param) # Symbolic dimension
                        else:
                            shape.append("?") # Unknown dimension
                print(f"  Shape: {shape}")
                print("-" * 20)
        else:
            print("No outputs found for this model.")
        print("\n")

        # Removed the "--- Inspect Nodes (Operations) ---" section as per your request.

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Updated model path as per your request
    model_file = "models/yolo11n.onnx" 
    
    print(f"Attempting to inspect model: {model_file}")
    inspect_onnx_model(model_file)
    print("\nInspection complete.")

