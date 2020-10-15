import tensorflow as tf

PATH_TO_CKPT = "face_detect/saved_model.pb"



if __name__ == "__main__":
    # converter = tf.lite.TFLiteConverter.from_saved_model("face_detect/")
    # # converter = tf.lite.TFLiteConverter.from_saved_model("mask_detector.model") # path to the SavedModel directory
    # maskNet = converter.convert()
    # # Save the model.
    # with open('face_detect.tflite', 'wb') as f:
    #   f.write(maskNet)
    # maskNet._make_predict_function()

    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.compat.v1.GraphDef()
    #     with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         # serialized_graph.decode("utf-8"))
    #         print(od_graph_def.ParseFromString(serialized_graph))
    #         print(tf.import_graph_def(od_graph_def, name=''))

    print("\n====== classifier model_dir, latest_checkpoint ===========")
    print(classifier.model_dir)
    print(classifier.latest_checkpoint())
    debug = False

    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        latest_checkpoint_path = classifier.latest_checkpoint()
        saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
        saver.restore(sess, latest_checkpoint_path)

        # Get the input and output tensors needed for toco.
        # These were determined based on the debugging info printed / saved below.
        input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
        input_tensor.set_shape([1, 4])
        out_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")
        out_tensor.set_shape([1, 3])

        # Pass the output node name we are interested in.
        # Based on the debugging info printed / saved below, pulled out the
        # name of the node for the logits (before the softmax is applied).
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=["dnn/logits/BiasAdd"])

        if debug is True:
            print("\nORIGINAL GRAPH DEF Ops ===========================================")
            ops = sess.graph.get_operations()
            for op in ops:
                if "BiasAdd" in op.name or "input_layer" in op.name:
                    print([op.name, op.values()])
            # save original graphdef to text file
            with open("estimator_graph.pbtxt", "w") as fp:
                fp.write(str(sess.graph_def))

            print("\nFROZEN GRAPH DEF Nodes ===========================================")
            for node in frozen_graph_def.node:
                print(node.name)
            # save frozen graph def to text file
            with open("estimator_frozen_graph.pbtxt", "w") as fp:
                fp.write(str(frozen_graph_def))

    tflite_model = tf.contrib.lite.toco_convert(frozen_graph_def, [input_tensor], [out_tensor])
    open("estimator_model.tflite", "wb").write(tflite_model)