"""
Helper script to generate a valid sample ONNX classification model for testing & CIA demo.
Takes input shape: [1, 3, 224, 224]
Produces output shape: [1, 2] representing [REAL_score, FAKE_score]
"""
import os
import onnx
from onnx import helper, TensorProto

def create_sample_deepfake_model(output_path="models/deepfake_model.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Define Model Input: [1, 3, 224, 224] (NCHW float tensor)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

    # 2. Define Model Output: [1, 2] (float tensor with Real/Fake logits)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    # 3. Global Average Pooling (flattens 224x224 spatial features to 1x1)
    gap_node = helper.make_node(
        'GlobalAveragePool',
        inputs=['input'],
        outputs=['gap_out']
    )

    # 4. Flatten node (1, 3, 1, 1) -> (1, 3)
    flatten_node = helper.make_node(
        'Flatten',
        inputs=['gap_out'],
        outputs=['flat_out'],
        axis=1
    )

    # 5. Linear weights: Shape [3, 2] and Bias: Shape [2]
    # Small weights to create realistic confidence scores based on RGB image statistics
    weights_tensor = helper.make_tensor(
        name='fc_weights',
        data_type=TensorProto.FLOAT,
        dims=[3, 2],
        vals=[0.6, -0.4, 0.3, -0.2, -0.5, 0.7] # weights for R, G, B channels
    )

    bias_tensor = helper.make_tensor(
        name='fc_bias',
        data_type=TensorProto.FLOAT,
        dims=[2],
        vals=[0.2, -0.2]
    )

    # 6. Gemm (Linear / Dense layer): Y = X * W + b -> Shape [1, 2]
    gemm_node = helper.make_node(
        'Gemm',
        inputs=['flat_out', 'fc_weights', 'fc_bias'],
        outputs=['output'],
        alpha=1.0,
        beta=1.0,
        transB=0
    )

    # 7. Construct Computation Graph
    graph = helper.make_graph(
        nodes=[gap_node, flatten_node, gemm_node],
        name='DeepfakeDetectorGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weights_tensor, bias_tensor]
    )

    # 8. Create Model with IR Version 8 (compatible with all ONNX Runtime versions)
    model = helper.make_model(
        graph,
        producer_name='DeepfakeDetectorApp',
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 15)]
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"[Success] Sample ONNX model generated at: {output_path}")

if __name__ == "__main__":
    create_sample_deepfake_model()
