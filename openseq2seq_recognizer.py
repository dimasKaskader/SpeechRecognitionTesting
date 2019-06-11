import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir, \
    create_logdir, create_model, get_interactive_infer_results

# Define the command line arguments that one would pass to run.py here

# A simpler version of what run.py does. It returns the created model and its saved checkpoint


def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None)
    return model, checkpoint


class OpenSeq2Seq:
    def __init__(self, model_path):
        self.args_S2T = ["--config_file=" + model_path + "/config.py",
            "--mode=interactive_infer",
            "--logdir=" + model_path + "/",
            "--batch_size_per_gpu=1",
            ]
        self.model_S2T, checkpoint_S2T = get_model(self.args_S2T, "S2T")
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=sess_config)
        vars_S2T = {}
        # vars_T2S = {}
        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            if "S2T" in v.name:
                vars_S2T["/".join(v.op.name.split("/")[1:])] = v
            '''if "T2S" in v.name:
                vars_T2S["/".join(v.op.name.split("/")[1:])] = v'''
        saver_S2T = tf.train.Saver(vars_S2T)
        saver_S2T.restore(self.sess, checkpoint_S2T)

    def recognize(self, wav_file):
        # Recognize speech
        results = get_interactive_infer_results(self.model_S2T, self.sess, model_in=[wav_file])
        english_recognized = results[0][0]

        return english_recognized
