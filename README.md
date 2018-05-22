# tf_helpers

## freeze graph

t.b.d.


## get information on a graph

Shell:
```
./summarize_graph --in_graph=/path/to/my/frozen/graph.pb
```

Example output:
```
Found 1 possible inputs: (name=image_tensor, type=uint8(4), shape=[?,?,?,3])
No variables spotted.
Found 4 possible outputs: (name=num_detections, op=Identity) (name=detection_classes, op=Identity) (name=detection_scores, op=Identity) (name=detection_boxes, op=Identity)
Found 16878731 (16.88M) const parameters, 0 (0) variable parameters, and 1548 control_edges
Op types used: 2572 Const, 549 Gather, 465 Identity, 452 Minimum, 371 Reshape, 360 Maximum, 344 Mul, 267 Sub, 261 Add, 211 Cast, 186 Greater, 180 Where, 180 Split, 165 Slice, 144 ConcatV2, 127 StridedSlice, 121 Pack, 116 Shape, 94 Unpack, 92 ZerosLike, 92 Squeeze, 90 NonMaxSuppressionV2, 64 Rsqrt, 55 Conv2D, 47 Relu6, 45 ExpandDims, 40 Fill, 37 Tile, 33 RealDiv, 30 Range, 29 Switch, 26 Enter, 21 DepthwiseConv2dNative, 14 Merge, 12 BiasAdd, 11 TensorArrayV3, 8 NextIteration, 6 Exit, 6 TensorArrayWriteV3, 6 TensorArraySizeV3, 6 TensorArrayGatherV3, 6 Sqrt, 5 TensorArrayReadV3, 5 TensorArrayScatterV3, 3 Equal, 3 Transpose, 3 Assert, 3 Rank, 2 Exp, 2 Less, 2 LoopCond, 1 All, 1 TopKV2, 1 Size, 1 Sigmoid, 1 ResizeBilinear, 1 Placeholder
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
```

## optimize graph using tensorflow

Using the graph_transform tool, the graph can be altered in order to reduce latency during inference. Here
[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms] following
is suggested:

Shell:

```
transform_graph \
--in_graph=/path/to/my/frozen/graph.pb \
--out_graph=/path/to/my/frozen/optimized_graph.pb \
--inputs='image_tensor' \
--outputs='num_detections,detection_classes,detection_scores,detection_boxes' \
--transforms='
  strip_unused_nodes(type=float, shape="1,299,299,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms
  sort_by_execution_order'
```

## check changes using summarize_graph

Shell:
```
./summarize_graph --in_graph=/path/to/my/frozen/optimized_graph.pb
```

Output:
```
Found 1 possible inputs: (name=image_tensor, type=uint8(4), shape=[?,?,?,3])
No variables spotted.
Found 4 possible outputs: (name=num_detections, op=Identity) (name=detection_classes, op=Identity) (name=detection_scores, op=Identity) (name=detection_boxes, op=Identity)
Found 16835655 (16.84M) const parameters, 0 (0) variable parameters, and 1536 control_edges
Op types used: 1837 Const, 549 Gather, 451 Minimum, 360 Maximum, 287 Reshape, 191 Sub, 183 Cast, 183 Greater, 180 Split, 180 Where, 173 Add, 119 StridedSlice, 117 Mul, 116 Shape, 109 Pack, 101 ConcatV2, 94 Unpack, 93 Slice, 92 ZerosLike, 92 Squeeze, 90 NonMaxSuppressionV2, 55 Conv2D, 47 Relu6, 29 Switch, 26 Enter, 21 DepthwiseConv2dNative, 14 Merge, 13 RealDiv, 12 BiasAdd, 12 Range, 11 TensorArrayV3, 8 ExpandDims, 8 NextIteration, 6 TensorArrayWriteV3, 6 TensorArraySizeV3, 6 Exit, 6 TensorArrayGatherV3, 5 TensorArrayScatterV3, 5 TensorArrayReadV3, 4 Identity, 4 Fill, 3 Transpose, 2 Assert, 2 LoopCond, 2 Less, 2 Exp, 2 Equal, 1 Size, 1 Sigmoid, 1 ResizeBilinear, 1 Placeholder, 1 Tile, 1 TopKV2
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=/home/jan/projects/opencv_tf_chain/optimized_graph.pb --show_flops --input_layer=image_tensor --input_layer_type=uint8 --input_layer_shape=-1,-1,-1,3 --output_layer=num_detections,detection_classes,detection_scores,detection_boxes
```

## create tensorrt graph

Python:
```
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile



def get_graph_def_from_pb(file):
    with gfile.FastGFile(file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def



trt_graph = trt.create_inference_graph(
                input_graph_def=get_graph_def_from_pb("/path/to/my/frozen/optimized_graph.pb"),
                outputs=["num_detections", "detection_classes", "detection_scores", "detection_boxes"],
                max_batch_size=1,
                max_workspace_size_bytes=500000000,
                precision_mode="FP32")


with gfile.FastGFile("/home/jan/projects/opencv_tf_chain/tensorrt_tf_optm_frozen_graph.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
```

**Important**: SSD mobilenet is not yet supported by TensorRT (https://github.com/tensorflow/tensorflow/issues/18744)

And it seems to be a good idea to have models with as few as possible dynamic shape dimensions.
